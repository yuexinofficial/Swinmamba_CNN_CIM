"""
Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining.

Based on: Liu et al., "Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining", 2024
VMamba pretrained encoder from: https://github.com/MzeroMiko/VMamba

This implementation:
  - Uses pure PyTorch selective_scan when mamba-ssm is unavailable (Windows-compatible)
  - Implements lightweight U-Net decoder blocks (no MONAI dependency)
  - Supports loading VMamba-Tiny ImageNet pretrained weights
"""

import math
import re
from functools import partial
from typing import Optional, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from timm.layers import DropPath, trunc_normal_

# Try to import mamba-ssm's optimized selective scan; fall back to pure PyTorch
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_SSM = True
    print("Using CUDA-accelerated mamba_ssm selective_scan")
except ImportError:
    HAS_MAMBA_SSM = False
    print("mamba_ssm not available, using pure-PyTorch selective_scan (slower but functional)")


# ==============================================================================
#  Pure-PyTorch Selective Scan (fallback when mamba-ssm unavailable)
# ==============================================================================

def selective_scan_pytorch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """
    Pure PyTorch implementation matching mamba_ssm's selective_scan_fn interface.

    Args:
        u:  (B, D_total, L)   input sequence
        delta: (B, D_total, L)   delta (timestep) parameter
        A: (D_total, N)  state matrix (pre-neg-exp'd, i.e., -exp(A_log))
        B: (B, K, N, L)  input-dependent B projection
        C: (B, K, N, L)  input-dependent C projection
        D: (D_total,)     skip-connection parameter (optional)
        z: unused (kept for interface compatibility)
        delta_bias: (D_total,)  bias added to delta before softplus
        delta_softplus: whether to apply softplus to (delta + delta_bias)

    Returns:
        out: (B, D_total, L)  or  (out, last_state) if return_last_state
    """
    B_batch, D_total, L = u.shape
    K = B.shape[1]          # number of scan directions (typically 4)
    N = B.shape[2]          # d_state
    D_ch = D_total // K     # channels per scan direction

    # Apply delta bias + softplus
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    # Reshape for per-direction processing
    u = u.view(B_batch, K, D_ch, L)               # (B, K, D_ch, L)
    delta = delta.view(B_batch, K, D_ch, L)        # (B, K, D_ch, L)
    A = A.view(K, D_ch, N)                         # (K, D_ch, N)
    if D is not None:
        D = D.view(K, D_ch)                        # (K, D_ch)

    # Allocate output & hidden state
    y = torch.zeros(B_batch, K, D_ch, L, device=u.device, dtype=torch.float32)
    h = torch.zeros(B_batch, K, D_ch, N, device=u.device, dtype=torch.float32)

    for t in range(L):
        # A_disc = exp(Δ_t · A)
        delta_t = delta[:, :, :, t]                 # (B, K, D_ch)
        A_disc = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, K, D_ch, N)

        # B_disc = Δ_t · B_t
        B_t = B[:, :, :, t].unsqueeze(2)            # (B, K, 1, N)
        u_t = u[:, :, :, t].unsqueeze(3)            # (B, K, D_ch, 1)
        B_disc_u = delta_t.unsqueeze(-1) * B_t * u_t  # (B, K, D_ch, N)

        # h_t = A_disc · h_{t-1} + Δ_t · B_t · u_t
        h = A_disc * h + B_disc_u

        # y_t = C_t^T @ h_t + D · u_t
        C_t = C[:, :, :, t].unsqueeze(2)            # (B, K, 1, N)
        y_t = (C_t * h).sum(dim=-1)                  # (B, K, D_ch)
        if D is not None:
            y_t = y_t + D.unsqueeze(0) * u[:, :, :, t]
        y[:, :, :, t] = y_t

    out = y.view(B_batch, -1, L)

    if return_last_state:
        return out, h
    return out


# ==============================================================================
#  Patch Embedding / Merging (from Swin-UMamba)
# ==============================================================================

class PatchEmbed2D(nn.Module):
    """Image to Patch Embedding (2D).

    Args:
        patch_size: Patch token size. Default: 4.
        in_chans: Number of input image channels. Default: 3.
        embed_dim: Number of linear projection output channels. Default: 96.
        norm_layer: Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)  # BCHW → BHWC
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    """Patch Merging Layer (2x downsample, 2x channel expansion)."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ==============================================================================
#  SS2D — 2D Selective Scan (the core Mamba operator for vision)
# ==============================================================================

class SS2D(nn.Module):
    """2D Selective Scan module — 4-directional scanning over feature maps."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # x_proj: 4 directions, each projects d_inner → (dt_rank + d_state*2)
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # dt_projs: 4 directions, each projects dt_rank → d_inner
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        # Use mamba_ssm if available, else fallback to pure PyTorch
        if HAS_MAMBA_SSM:
            self.selective_scan = selective_scan_fn
        else:
            self.selective_scan = selective_scan_pytorch

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # 4-directional scanning: H→W, W→H, H→W(reversed), W→H(reversed)
        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, K, D, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        # Reconstruct 2D from 4-directional scans
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ==============================================================================
#  VSS Blocks & Layers (the encoder building blocks)
# ==============================================================================

class VSSBlock(nn.Module):
    """Single VSS (Visual State Space) block: LayerNorm → SS2D → DropPath + residual."""

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        return input + self.drop_path(self.self_attention(self.ln_1(input)))


class VSSLayer(nn.Module):
    """A VSS layer for one encoder stage — multiple VSSBlocks + optional downsampling."""

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        # Initialize out_proj.weight with kaiming_uniform
        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSMEncoder(nn.Module):
    """VMamba-based hierarchical encoder (the "Swin-Mamba" backbone).

    Produces multi-scale features for U-Net skip connections.
    Default: VMamba-Tiny (depths=[2,2,9,2], dims=[96,192,384,768])
    """

    def __init__(self, patch_size=4, in_chans=3, depths=(2, 2, 9, 2),
                 dims=(96, 192, 384, 768), d_state=16, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        self.ape = False
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        """Returns list of feature maps: [stem_out, stage0, stage1, stage2, stage3, stage4]"""
        x_ret = []
        x_ret.append(x)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 3, 1, 2))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x_ret


# ==============================================================================
#  Lightweight Decoder Blocks (replacing MONAI dependencies)
# ==============================================================================

class BasicResBlock(nn.Module):
    """Lightweight equivalent of MONAI's UnetrBasicBlock.

    Two Conv3x3 + InstanceNorm + LeakyReLU with optional residual connection.
    """
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
    """Lightweight equivalent of MONAI's UnetrUpBlock.

    Upsample → Conv3x3 → Concat skip → Two Conv3x3 + residual.
    """
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
#  Swin-UMamba — Main Model
# ==============================================================================

class SwinUMamba(nn.Module):
    """Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining.

    Architecture:
        Input → Stem (Conv7x7 s2) → VSSMEncoder (4 stages, SS2D scan)
        → Conv Blocks (skip processing) → UpBlocks (conv decoder) → Output Conv

    Args:
        in_chans: Input image channels (default 3 for RGB, 1 for grayscale)
        out_chans: Output segmentation classes (default 1 for binary)
        feat_size: Feature dimensions per stage [stem, s1, s2, s3, bottleneck]
        drop_path_rate: Stochastic depth rate
        deep_supervision: Enable deep supervision (multi-scale outputs)
        pretrained: Load VMamba ImageNet pretrained weights
        pretrained_ckpt_path: Path to VMamba checkpoint
    """

    def __init__(
        self,
        in_chans: int = 3,
        out_chans: int = 1,
        feat_size: List[int] = None,
        drop_path_rate: float = 0.2,
        deep_supervision: bool = False,
        pretrained: bool = True,
        pretrained_ckpt_path: str = "./data/pretrained/vmamba/vmamba_tiny_e292.pth",
    ) -> None:
        super().__init__()

        if feat_size is None:
            feat_size = [48, 96, 192, 384, 768]
        self.feat_size = feat_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.deep_supervision = deep_supervision

        # Stem: initial downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, feat_size[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(feat_size[0], eps=1e-5, affine=True),
        )

        # VSSM Encoder (VMamba-Tiny architecture)
        self.vssm_encoder = VSSMEncoder(
            patch_size=2, in_chans=feat_size[0],
            depths=[2, 2, 9, 2], dims=feat_size[1:],
            drop_path_rate=drop_path_rate,
        )

        # Encoder conv blocks for skip connections
        self.encoder1 = BasicResBlock(in_chans, feat_size[0])
        self.encoder2 = BasicResBlock(feat_size[0], feat_size[1])
        self.encoder3 = BasicResBlock(feat_size[1], feat_size[2])
        self.encoder4 = BasicResBlock(feat_size[2], feat_size[3])
        self.encoder5 = BasicResBlock(feat_size[3], feat_size[4])

        # Decoder: UpBlock = Upsample + Concat + Conv
        self.decoder6 = UpBlock(feat_size[4], feat_size[4])       # bottleneck → stage4
        self.decoder5 = UpBlock(feat_size[4], feat_size[3])       # stage4 → stage3
        self.decoder4 = UpBlock(feat_size[3], feat_size[2])       # stage3 → stage2
        self.decoder3 = UpBlock(feat_size[2], feat_size[1])       # stage2 → stage1
        self.decoder2 = UpBlock(feat_size[1], feat_size[0])       # stage1 → stem
        self.decoder1 = BasicResBlock(feat_size[0], feat_size[0]) # final refinement

        # Output layers (with optional deep supervision)
        self.out_layers = nn.ModuleList()
        for i in range(4):
            self.out_layers.append(nn.Conv2d(feat_size[i], out_chans, kernel_size=1))

        # Load pretrained weights
        self.pretrained_loaded = False
        if pretrained:
            self._load_pretrained(pretrained_ckpt_path)

    def _load_pretrained(self, ckpt_path: str):
        """Load ImageNet pretrained VMamba-Tiny weights into VSSMEncoder."""
        import os
        if not os.path.exists(ckpt_path):
            print(f"[SwinUMamba] Pretrained checkpoint not found at: {ckpt_path}")
            print(f"[SwinUMamba] Training from scratch (random initialization)")
            return

        print(f"[SwinUMamba] Loading pretrained weights from: {ckpt_path}")
        skip_params = [
            "norm.weight", "norm.bias", "head.weight", "head.bias",
            "patch_embed.proj.weight", "patch_embed.proj.bias",
            "patch_embed.norm.weight", "patch_embed.norm.weight"
        ]

        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_dict = self.vssm_encoder.state_dict()
        loaded = 0
        skipped = 0

        for k, v in ckpt['model'].items():
            if k in skip_params:
                skipped += 1
                continue
            kr = f"vssm_encoder.{k}"
            if "downsample" in kr:
                i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
                kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
                assert kr in model_dict.keys(), f"Key not found: {kr}"
            if kr in model_dict.keys():
                assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
                model_dict[kr] = v
                loaded += 1
            else:
                skipped += 1

        self.vssm_encoder.load_state_dict(model_dict)
        self.pretrained_loaded = True
        print(f"[SwinUMamba] Loaded {loaded} parameters, skipped {skipped}")

    def forward(self, x_in):
        # Stem
        x1 = self.stem(x_in)                          # [B, 48, H/2, W/2]

        # VSSM Encoder: returns [stem_out(48), s0(96), s1(192), s2(384), s3(768), s4(768)]
        vss_outs = self.vssm_encoder(x1)

        # Process skip features through conv blocks
        enc1 = self.encoder1(x_in)                    # input-level skip
        enc2 = self.encoder2(vss_outs[0])             # stage0 → 96
        enc3 = self.encoder3(vss_outs[1])             # stage1 → 192
        enc4 = self.encoder4(vss_outs[2])             # stage2 → 384
        enc5 = self.encoder5(vss_outs[3])             # stage3 → 768
        enc_hidden = vss_outs[4]                      # bottleneck → 768

        # Decoder with skip connections
        dec4 = self.decoder6(enc_hidden, enc5)        # 768→768, upsample 2x
        dec3 = self.decoder5(dec4, enc4)              # 768→384, upsample 2x
        dec2 = self.decoder4(dec3, enc3)              # 384→192, upsample 2x
        dec1 = self.decoder3(dec2, enc2)              # 192→96, upsample 2x
        dec0 = self.decoder2(dec1, enc1)              # 96→48, upsample 2x
        dec_out = self.decoder1(dec0)                 # 48→48, refine

        if self.deep_supervision:
            feat_out = [dec_out, dec1, dec2, dec3]
            out = []
            for i in range(4):
                pred = self.out_layers[i](feat_out[i])
                out.append(pred)
        else:
            out = self.out_layers[0](dec_out)

        return out

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


# ==============================================================================
#  Pretrained Weights Download Utility
# ==============================================================================

PRETRAINED_URL = (
    "https://github.com/MzeroMiko/VMamba/releases/download/"
    "%2320240218/vssmtiny_dp01_ckpt_epoch_292.pth"
)


def download_pretrained_weights(save_dir: str = "./data/pretrained/vmamba") -> str:
    """Download VMamba-Tiny pretrained weights.

    Returns:
        Path to downloaded checkpoint file.
    """
    import os
    import sys

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vmamba_tiny_e292.pth")

    if os.path.exists(save_path):
        print(f"Pretrained weights already exist at: {save_path}")
        return save_path

    print(f"Downloading VMamba-Tiny pretrained weights...")
    print(f"URL: https://github.com/MzeroMiko/VMamba/releases/download/20240218/vssmtiny_dp01_ckpt_epoch_292.pth")

    try:
        import urllib.request
        # The actual URL (without URL encoding for the tag)
        url = "https://github.com/MzeroMiko/VMamba/releases/download/20240218/vssmtiny_dp01_ckpt_epoch_292.pth"
        print(f"Downloading to: {save_path}")

        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                sys.stdout.write(f"\r  Downloading: {percent:.1f}% "
                                 f"({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, save_path, report_progress)
        print(f"\nDownload complete: {save_path}")
        return save_path

    except Exception as e:
        print(f"\nDownload failed: {e}")
        print(f"Please download manually from:")
        print(f"  https://github.com/MzeroMiko/VMamba/releases/tag/20240218")
        print(f"  File: vssmtiny_dp01_ckpt_epoch_292.pth")
        print(f"  Save to: {save_path}")
        return ""
