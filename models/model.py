import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import re
from typing import Optional, List, Tuple

try:
    from torchvision.models import resnet34, ResNet34_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision library not available.")

from models.swin_umamba import VSSMEncoder


# ==============================================================================
#  基础模块
# ==============================================================================

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )


class SEBlockNoShortcut(nn.Module):
    def __init__(self, channels=512, mid_channels=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(channels, mid_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mid_channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        out = self.pool(x).view(B, C)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out).unsqueeze(-1)
        out = self.sigmoid(out)
        return out


# ==============================================================================
#  TMCA — 通道交互注意力
# ==============================================================================

class TMCA(nn.Module):
    """Channel_Exchange_Attention: 在CNN和Mamba特征之间交换通道信息"""
    def __init__(self, high_level_channel=512, low_level_channel=256):
        super().__init__()
        self.channel_self_attn1 = SEBlockNoShortcut(high_level_channel, low_level_channel)
        self.channel_self_attn2 = SEBlockNoShortcut(low_level_channel, high_level_channel)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C1, H1, W1 = x1.shape
        _, C2, _, _ = x2.shape

        x1_attn = self.channel_self_attn1(x1)
        x1_attn = x1_attn.transpose(1, 2)
        x2_attn = self.channel_self_attn2(x2)
        attn = x2_attn @ x1_attn

        attn1 = F.softmax(attn, dim=2)
        output1 = torch.einsum('bxy,byhw->bxhw', attn1, x1)

        attn2 = F.softmax(attn, dim=1)
        output2 = torch.einsum('bxy,bxhw->byhw', attn2, x2)

        return output1, output2


# ==============================================================================
#  SpatialAttentionFusion — 空间注意力 + 残差连接 动态融合
# ==============================================================================

class SpatialAttentionFusion(nn.Module):
    """
    空间注意力 + 残差连接 动态融合机制

    替换原有的 XFF 模块，提供更强的空间选择性融合:
      1. 通过拼接特征生成空间注意力图 — 学习每个位置两个分支的相对重要性
      2. 注意力调制后的特征通过残差连接保留原始信息流
      3. 融合特征通过投影层输出

    输入:
        - feat_a: 分支A特征 [B, C, H, W]  (Mamba/TMCA增强后)
        - feat_b: 分支B特征 [B, C, H, W]  (CNN/TMCA增强后)

    输出:
        - out: 融合特征 [B, C, H, W]
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels * 2, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.feat_a_mod = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.feat_b_mod = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.fusion_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([feat_a, feat_b], dim=1)
        spatial_weight = self.spatial_attn(concat)

        feat_a_mod = self.feat_a_mod(feat_a)
        feat_b_mod = self.feat_b_mod(feat_b)

        feat_a_out = feat_a + feat_a_mod * spatial_weight
        feat_b_out = feat_b + feat_b_mod * (1.0 - spatial_weight)

        fused = self.fusion_proj(torch.cat([feat_a_out, feat_b_out], dim=1))

        return fused


# ==============================================================================
#  CIM — 对比度增强模块
# ==============================================================================

class ContrastImprovementModule(nn.Module):
    """输入图像对比度/细节增强, 放在编码器之前"""
    def __init__(self, in_channels=1, scaling_factor=1.5, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.smooth_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, bias=False
        )
        self.alpha = nn.Parameter(torch.tensor(scaling_factor, dtype=torch.float32))
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.constant_(
            self.smooth_conv.weight,
            1.0 / (self.smooth_conv.kernel_size[0] ** 2)
        )

    def forward(self, x):
        smooth_image = self.smooth_conv(x)
        detail_layer = x - smooth_image
        output = smooth_image + self.alpha * detail_layer
        return output


# ==============================================================================
#  ResNet34 Encoder
# ==============================================================================

class ResNet34Encoder(nn.Module):
    """ResNet34编码器, 使用torchvision预训练模型"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for ResNet34Encoder")

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = resnet34(weights=weights)
        self.pretrained_loaded = pretrained

        if pretrained:
            print("Loaded ResNet34 with ImageNet pretraining")
        else:
            print("Loaded ResNet34 without pretraining")

        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Identity()

        self.layer0 = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1,
            self.resnet.relu, self.resnet.maxpool
        )
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.layer0(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        return [feat1, feat2, feat3, feat4]


# ==============================================================================
#  UNetDecoder
# ==============================================================================

# ==============================================================================
#  Swin-UMamba-style Decoder Blocks (BasicResBlock + UpBlock)
# ==============================================================================

class BasicResBlock(nn.Module):
    """Two Conv3x3 + InstanceNorm + LeakyReLU with optional residual connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, res_block: bool = True):
        super().__init__()
        self.res_block = res_block
        self.use_residual = res_block and (in_channels == out_channels) and (stride == 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size // 2, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=1, padding=kernel_size // 2, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if not self.use_residual and self.res_block:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                       stride=stride, bias=False)
            self.skip_norm = nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.use_residual:
            out = out + residual
        elif self.res_block:
            residual = self.skip_conv(residual)
            residual = self.skip_norm(residual)
            out = out + residual

        out = self.act2(out)
        return out


class UpBlock(nn.Module):
    """ConvTranspose2d upsample → concat skip → BasicResBlock."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 upsample_kernel_size: int = 2, res_block: bool = True):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=upsample_kernel_size, stride=upsample_kernel_size, bias=False
        )
        self.conv_block = BasicResBlock(
            out_channels + out_channels, out_channels,
            kernel_size=kernel_size, stride=1, res_block=res_block
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


# ==============================================================================
#  SwinUMambaDecoder — 含 raw input skip + 深监督
# ==============================================================================

class SwinUMambaDecoder(nn.Module):
    """Swin-UMamba style decoder with raw input skip and optional deep supervision.

    Feature sizes (feat_sizes): [stem_ch, s1, s2, s3, bottleneck]
                                [48,      64,  128, 256, 512]

    Skip hierarchy:
      enc_raw:  raw_image (3ch)  → BasicResBlock → 48@224
      enc_stem: stem_out (48ch)  → BasicResBlock → 48@112
      skip2:    fused[0] (64ch)  → BasicResBlock → 64@56
      skip3:    fused[1] (128ch) → BasicResBlock → 128@28
      skip4:    fused[2] (256ch) → BasicResBlock → 256@14
      bn_skip:  fused[3] (512ch) → BasicResBlock → 512@7

    Decoder path:
      bn(512@7)  → UpBlock(512,256) + skip4  → d4(256@14)
      d4(256@14) → UpBlock(256,128) + skip3  → d3(128@28)
      d3(128@28) → UpBlock(128,64)  + skip2  → d2(64@56)
      d2(64@56)  → UpBlock(64,48)   + enc_stem → d1(48@112)
      d1(48@112) → UpBlock(48,48)   + enc_raw → d0(48@224)
      d0(48@224) → BasicResBlock(48,48) → refined(48@224)
    """
    def __init__(self, feat_sizes: List[int] = None, out_chans: int = 1,
                 deep_supervision: bool = False):
        super().__init__()
        if feat_sizes is None:
            feat_sizes = [48, 64, 128, 256, 512]
        fs = feat_sizes  # [stem, s1, s2, s3, bottleneck]
        self.deep_supervision = deep_supervision

        # Skip processing blocks
        self.enc_raw = BasicResBlock(3, fs[0])           # raw input → stem_ch
        self.enc_stem = BasicResBlock(fs[0], fs[0])      # stem → stem_ch (identity-like)
        self.skip2 = BasicResBlock(fs[1], fs[1])         # s1 → s1
        self.skip3 = BasicResBlock(fs[2], fs[2])         # s2 → s2
        self.skip4 = BasicResBlock(fs[3], fs[3])         # s3 → s3
        self.bn_skip = BasicResBlock(fs[4], fs[4])       # bottleneck → bottleneck

        # Decoder UpBlocks (bottleneck → full resolution + raw skip)
        self.up4 = UpBlock(fs[4], fs[3])    # 512→256
        self.up3 = UpBlock(fs[3], fs[2])    # 256→128
        self.up2 = UpBlock(fs[2], fs[1])    # 128→64
        self.up1 = UpBlock(fs[1], fs[0])    # 64→48
        self.up0 = UpBlock(fs[0], fs[0])    # 48→48

        # Final refinement
        self.refine = BasicResBlock(fs[0], fs[0])

        # Output conv
        self.out_conv = nn.Conv2d(fs[0], out_chans, kernel_size=1)

        # Deep supervision output layers (4 multi-scale predictions)
        if deep_supervision:
            self.out_layers = nn.ModuleList([
                nn.Conv2d(fs[0], out_chans, 1),  # refined(48@224)
                nn.Conv2d(fs[1], out_chans, 1),  # d2(64@56)
                nn.Conv2d(fs[2], out_chans, 1),  # d3(128@28)
                nn.Conv2d(fs[3], out_chans, 1),  # d4(256@14)
            ])

    def forward(self, fused_features: List[torch.Tensor],
                stem_out: torch.Tensor, raw_image: torch.Tensor):
        # Process skip connections
        s_raw = self.enc_raw(raw_image)           # 3→48@224
        s_stem = self.enc_stem(stem_out)          # 48→48@112
        s2 = self.skip2(fused_features[0])        # 64→64@56
        s3 = self.skip3(fused_features[1])        # 128→128@28
        s4 = self.skip4(fused_features[2])        # 256→256@14
        bn = self.bn_skip(fused_features[3])      # 512→512@7

        # Decoder with skip connections
        d4 = self.up4(bn, s4)          # 512→256@14
        d3 = self.up3(d4, s3)          # 256→128@28
        d2 = self.up2(d3, s2)          # 128→64@56
        d1 = self.up1(d2, s_stem)      # 64→48@112
        d0 = self.up0(d1, s_raw)       # 48→48@224
        d_refined = self.refine(d0)    # 48→48@224

        if self.deep_supervision:
            return [
                self.out_layers[0](d_refined),   # main (48@224)
                self.out_layers[1](d2),           # aux  (64@56)
                self.out_layers[2](d3),           # aux  (128@28)
                self.out_layers[3](d4),           # aux  (256@14)
            ]
        else:
            return self.out_conv(d_refined)


# ==============================================================================
#  主模型 — 双分支分割网络 (ResNet34 + VMamba-Tiny Encoder)
# ==============================================================================

class TextGuidedBreastUSSegmentation(nn.Module):
    """
    双分支医学图像分割模型

    分支1: ResNet34 (CNN, 局部细节)
    分支2: VMamba-Tiny Encoder (VSSMEncoder, SS2D扫描, 全局上下文)
           — 使用作者原版 SS2D / VSSMEncoder, 可加载 VMamba-tiny 预训练权重

    融合: TMCA通道交互 → SpatialAttentionFusion空间注意力融合 → SwinUMambaDecoder
         (含 raw input skip + BasicResBlock + UpBlock + 深监督)
    """

    # VMamba-Tiny 各阶段输出通道数
    MAMBA_OUT_CHANNELS = [96, 192, 384, 768]

    def __init__(self,
                 img_size: int = 224,
                 in_channels: int = 3,
                 load_pretrained: bool = True,
                 pretrained_ckpt_path: str = "./data/pretrained/vmamba/vmamba_tiny_e292.pth",
                 freeze_mamba_encoder: bool = True,
                 use_cim: bool = True,
                 cim_scaling_factor: float = 0.5,
                 cim_kernel_size: int = 3,
                 foreground_prior: float = 0.0,
                 deep_supervision: bool = False):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.use_cim = use_cim
        self.freeze_mamba_encoder = freeze_mamba_encoder
        self.deep_supervision = deep_supervision

        # 0. CIM 输入增强
        if self.use_cim:
            self.cim = ContrastImprovementModule(
                in_channels=in_channels,
                scaling_factor=cim_scaling_factor,
                kernel_size=cim_kernel_size
            )
            print(f"CIM enabled (scaling_factor={cim_scaling_factor}, kernel_size={cim_kernel_size})")
        else:
            self.cim = None
            print("CIM disabled")

        # 1. 双分支Encoder

        # 1a. CNN分支: ResNet34
        self.cnn_encoder = ResNet34Encoder(pretrained=load_pretrained)

        # 1b. Mamba分支: VMamba-Tiny (作者原版 SS2D / VSSMEncoder)
        #     Stem → VSSMEncoder(patch_size=2, depths=[2,2,9,2], dims=[96,192,384,768])
        self.mamba_stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(48, eps=1e-5, affine=True),
        )
        self.mamba_encoder = VSSMEncoder(
            patch_size=2, in_chans=48,
            depths=[2, 2, 9, 2], dims=[96, 192, 384, 768],
            d_state=16, drop_path_rate=0.2,
        )
        self.mamba_pretrained_loaded = False

        # 加载 VMamba-Tiny 预训练权重
        if load_pretrained:
            self._load_vmamba_pretrained(pretrained_ckpt_path)
            if self.freeze_mamba_encoder:
                self._set_mamba_encoder_trainable(False)

        # 2. 特征投影层 (对齐 Mamba 和 ResNet 的通道数)
        #    VMamba-Tiny:  [96, 192, 384, 768]  →  ResNet34: [64, 128, 256, 512]
        mamba_ch = self.MAMBA_OUT_CHANNELS
        target_ch = [64, 128, 256, 512]
        self.trans_proj1 = nn.Conv2d(mamba_ch[0], target_ch[0], kernel_size=1)
        self.trans_proj2 = nn.Conv2d(mamba_ch[1], target_ch[1], kernel_size=1)
        self.trans_proj3 = nn.Conv2d(mamba_ch[2], target_ch[2], kernel_size=1)
        self.trans_proj4 = nn.Conv2d(mamba_ch[3], target_ch[3], kernel_size=1)
        print(f"VMamba-Tiny output channels: {mamba_ch} -> projected to {target_ch}")

        # 3. TMCA 通道交互注意力
        self.tmca1 = TMCA(64, 64)
        self.tmca2 = TMCA(128, 128)
        self.tmca3 = TMCA(256, 256)
        self.tmca4 = TMCA(512, 512)

        # 4. SpatialAttentionFusion 空间注意力融合
        self.xff1 = SpatialAttentionFusion(channels=64)
        self.xff2 = SpatialAttentionFusion(channels=128)
        self.xff3 = SpatialAttentionFusion(channels=256)
        self.xff4 = SpatialAttentionFusion(channels=512)

        # 5. Swin-UMamba风格解码器 (BasicResBlock + UpBlock + raw input skip + 深监督)
        self.decoder = SwinUMambaDecoder(
            feat_sizes=[48, 64, 128, 256, 512],
            out_chans=1,
            deep_supervision=deep_supervision
        )
        print(f"SwinUMambaDecoder: feat_sizes=[48,64,128,256,512], "
              f"deep_supervision={deep_supervision}")

        # 6. Bias-prior 初始化
        if foreground_prior > 0.0 and self.decoder.out_conv.bias is not None:
            bias_value = math.log(foreground_prior / (1.0 - foreground_prior))
            nn.init.constant_(self.decoder.out_conv.bias, bias_value)
            print(f"Bias-prior init: pi={foreground_prior:.4f}, b0={bias_value:.4f}")
        else:
            print("Bias-prior init disabled")

    # ------------------------------------------------------------------
    #  VMamba-Tiny 预训练权重加载
    # ------------------------------------------------------------------

    def _load_vmamba_pretrained(self, ckpt_path: str):
        """加载 VMamba-Tiny ImageNet 预训练权重到 mamba_encoder (VSSMEncoder)。"""
        if not os.path.exists(ckpt_path):
            print(f"[Pretrained] Checkpoint not found: {ckpt_path}")
            print(f"[Pretrained] Mamba encoder uses random initialization")
            return

        print(f"[Pretrained] Loading VMamba-Tiny weights from: {ckpt_path}")

        # 需要跳过的键 (patch_embed 和 norm 参数与我们的 stem 不匹配)
        skip_params = [
            "norm.weight", "norm.bias", "head.weight", "head.bias",
            "patch_embed.proj.weight", "patch_embed.proj.bias",
            "patch_embed.norm.weight", "patch_embed.norm.weight"
        ]

        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_dict = self.mamba_encoder.state_dict()
        loaded = 0
        skipped = 0
        debug_printed = False  # 首次不匹配时打印所有 model keys

        for k, v in ckpt['model'].items():
            if k in skip_params:
                skipped += 1
                continue
            kr = k  # model_dict = mamba_encoder.state_dict()，key 不需要加前缀
            if "downsample" in kr:
                i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
                kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            if kr in model_dict.keys():
                if v.shape != model_dict[kr].shape:
                    print(f"[Pretrained] Shape mismatch for {kr}: ckpt{v.shape} vs model{model_dict[kr].shape}, skipping")
                    skipped += 1
                    continue
                model_dict[kr] = v
                loaded += 1
            else:
                if not debug_printed and ("downsample" in k or "patch_embed" in k or "norm" in k):
                    print(f"[Pretrained] Key mapping failed for: {k} -> {kr}")
                    print(f"[Pretrained] Model keys containing similar pattern:")
                    for mk in sorted(model_dict.keys()):
                        if any(w in mk for w in ["downsample", "patch_embed", "norm", "reduction"]):
                            print(f"  {mk}")
                    debug_printed = True
                skipped += 1

        self.mamba_encoder.load_state_dict(model_dict)
        self.mamba_pretrained_loaded = True
        print(f"[Pretrained] Loaded {loaded} params into mamba_encoder, skipped {skipped}")

    # ------------------------------------------------------------------
    #  冻结 / 解冻 Mamba encoder
    # ------------------------------------------------------------------

    def _set_mamba_encoder_trainable(self, trainable: bool):
        for param in self.mamba_encoder.parameters():
            param.requires_grad = trainable
        status = "frozen" if not trainable else "unfrozen"
        print(f"[Pretrained] Mamba encoder {status}")

    @torch.no_grad()
    def freeze_mamba(self):
        self._set_mamba_encoder_trainable(False)

    @torch.no_grad()
    def unfreeze_mamba(self):
        self._set_mamba_encoder_trainable(True)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor):
        # Save raw image for skip connection (before CIM)
        raw_image = image

        # 1. CIM增强
        if self.use_cim and self.cim is not None:
            image = self.cim(image)

        # 2a. CNN分支编码
        cnn_features = self.cnn_encoder(image)

        # 2b. Mamba分支编码 (stem → VSSMEncoder)
        x_mamba = self.mamba_stem(image)
        vss_outs = self.mamba_encoder(x_mamba)
        # VSSMEncoder 返回 6 个特征图: [stem_input, s0, s1, s2, s3, s4]
        # 取 s0~s3 (索引1~4), 空间分辨率与 ResNet34 各层对齐
        mamba_features = vss_outs[1:5]

        # 3. 通道对齐
        aligned_mamba_features = [
            self.trans_proj1(mamba_features[0]),
            self.trans_proj2(mamba_features[1]),
            self.trans_proj3(mamba_features[2]),
            self.trans_proj4(mamba_features[3])
        ]

        # 4. TMCA 通道交互
        cnn_tmca1, mamba_tmca1 = self.tmca1(aligned_mamba_features[0], cnn_features[0])
        cnn_tmca2, mamba_tmca2 = self.tmca2(aligned_mamba_features[1], cnn_features[1])
        cnn_tmca3, mamba_tmca3 = self.tmca3(aligned_mamba_features[2], cnn_features[2])
        cnn_tmca4, mamba_tmca4 = self.tmca4(aligned_mamba_features[3], cnn_features[3])

        # 5. 空间注意力融合
        fused_feat1 = self.xff1(mamba_tmca1, cnn_tmca1)
        fused_feat2 = self.xff2(mamba_tmca2, cnn_tmca2)
        fused_feat3 = self.xff3(mamba_tmca3, cnn_tmca3)
        fused_feat4 = self.xff4(mamba_tmca4, cnn_tmca4)

        # 6. Swin-UMamba风格解码 (含 raw input skip + stem skip)
        fused_features = [fused_feat1, fused_feat2, fused_feat3, fused_feat4]
        output = self.decoder(fused_features, x_mamba, raw_image)

        return output


# 兼容性别名
BreastUSSegmentation = TextGuidedBreastUSSegmentation


# ==============================================================================
#  测试入口
# ==============================================================================

if __name__ == '__main__':
    from utils import SegmentationLoss

    print("=" * 80)
    print("Test: TextGuidedBreastUSSegmentation (ResNet34 + VMamba-Tiny Encoder)")
    print("=" * 80)

    # Test 1: 不加载预训练权重，无深监督
    print("\n--- Test 1: Random Init (no deep supervision) ---")
    model = TextGuidedBreastUSSegmentation(
        load_pretrained=False, use_cim=True, deep_supervision=False
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}  (single tensor)")

    target = torch.randint(0, 2, (2, 1, 224, 224)).float()
    criterion = SegmentationLoss()
    loss = criterion(out, target)
    print(f"Loss value: {loss.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # 各模块参数统计
    print("\n  Module Breakdown:")
    print("-" * 80)
    print(f"{'Module':40s} {'Total':>12s} {'Trainable':>12s}")
    print("-" * 80)
    for name, m in [
        ('CIM', model.cim),
        ('CNN Encoder (ResNet34)', model.cnn_encoder),
        ('Mamba Stem', model.mamba_stem),
        ('Mamba Encoder (VSSM)', model.mamba_encoder),
        ('TMCA + XFF', nn.Sequential(
            model.tmca1, model.tmca2, model.tmca3, model.tmca4,
            model.xff1, model.xff2, model.xff3, model.xff4,
        )),
        ('Decoder (SwinUMamba)', model.decoder),
    ]:
        if m is not None:
            t = sum(p.numel() for p in m.parameters())
            tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"{name:40s} {t:>12,} {tr:>12,}")

    # Test 2: 深监督模式
    print("\n" + "=" * 80)
    print("Test 2: With Deep Supervision")
    print("=" * 80)
    model_ds = TextGuidedBreastUSSegmentation(
        load_pretrained=False, use_cim=True, deep_supervision=True
    )
    out_ds = model_ds(x)
    print(f"Output type:  list (deep supervision)")
    for i, o in enumerate(out_ds):
        print(f"  out[{i}] shape: {o.shape}")
    total_ds = sum(p.numel() for p in model_ds.parameters())
    print(f"Total params: {total_ds:,} ({total_ds/1e6:.2f}M)")

    # Test 3: 尝试加载预训练权重
    print("\n" + "=" * 80)
    print("Test 3: With Pretrained Weights (if available)")
    print("=" * 80)
    model3 = TextGuidedBreastUSSegmentation(
        load_pretrained=True, use_cim=True, freeze_mamba_encoder=True,
        deep_supervision=False
    )
    out3 = model3(x)
    print(f"Output shape: {out3.shape}")
    total3 = sum(p.numel() for p in model3.parameters())
    trainable3 = sum(p.numel() for p in model3.parameters() if p.requires_grad)
    frozen3 = total3 - trainable3
    print(f"Total: {total3:,} ({total3/1e6:.2f}M)")
    print(f"Trainable: {trainable3:,} ({trainable3/1e6:.2f}M)")
    print(f"Frozen: {frozen3:,} ({frozen3/1e6:.2f}M)")
    print("=" * 80)
