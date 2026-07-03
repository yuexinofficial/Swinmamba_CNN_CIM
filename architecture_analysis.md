# SwinMamba + ResNet34 + CIM + SwinDecoder 架构解析

## 📋 目录

1. [项目概览](#1-项目概览)
2. [整体框架](#2-整体框架)
3. [核心创新模块详解](#3-核心创新模块详解)
4. [完整数据流](#4-完整数据流)
5. [训练策略](#5-训练策略)
6. [损失函数与评估指标](#6-损失函数与评估指标)
7. [代码模块组织](#7-代码模块组织)

---

## 1. 项目概览

本项目是一个**双分支医学图像分割模型**，面向乳腺超声（BreastUS）和细胞核（MoNuSeg）分割任务。核心思路是**融合 CNN 的局部细节提取能力与 Mamba (SS2D) 的全局上下文建模能力**，通过通道交互注意力和空间注意力机制实现高效的多模态特征融合。

### 关键技术栈

| 组件 | 技术选型 | 作用 |
|------|---------|------|
| **CNN 编码器** | ResNet34 (ImageNet 预训练) | 提取局部纹理和细节特征 |
| **Mamba 编码器** | VMamba-Tiny + SS2D (4方向选择性扫描) | 建模全局空间依赖 |
| **特征交互** | TMCA (通道交互注意力) | CNN ↔ Mamba 通道级信息交换 |
| **特征融合** | SpatialAttentionFusion (空间注意力+残差) | 动态空间权重融合 |
| **解码器** | SwinUMambaDecoder (BasicResBlock + UpBlock) | 渐进式上采样 + 跳跃连接 |
| **输入增强** | CIM (对比度增强模块) | 增强输入图像的细节/对比度 |
| **损失函数** | Focal Tversky Loss | 处理类别不平衡 |
| **学习率调度** | WarmupPolyLR (Warm-up + Poly 衰减) | 稳定训练收敛 |

---

## 2. 整体框架

### 2.1 宏观架构图

```
                              ┌──────────────────────────────┐
                              │         输入图像 (3×224×224)     │
                              └──────────────┬───────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │   CIM 对比度增强  │  ← 创新点①
                                    │  (可选, 默认开启)  │
                                    └────────┬────────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          │                                     │
                 ┌────────▼────────┐                  ┌────────▼────────┐
                 │  ResNet34       │                  │  Mamba Stem     │
                 │  (CNN 分支)     │                  │  Conv7×7 s=2    │
                 │  ImageNet预训练  │                  │  → 48ch @ 112   │
                 └────────┬────────┘                  └────────┬────────┘
                          │                                     │
                 ┌────────▼────────┐                  ┌────────▼────────┐
                 │ Layer1 → 64ch   │                  │ VSSMEncoder     │
                 │ Layer2 → 128ch  │                  │ (VMamba-Tiny)   │
                 │ Layer3 → 256ch  │                  │ SS2D 4方向扫描   │  ← 创新点②
                 │ Layer4 → 512ch  │                  │ [96,192,384,768]│
                 └────────┬────────┘                  └────────┬────────┘
                          │                                     │
                          │                            ┌────────▼────────┐
                          │                            │ Conv1×1 投影层   │
                          │                            │ [96→64, 192→128,│
                          │                            │  384→256,768→512]│
                          │                            └────────┬────────┘
                          │                                     │
                          └──────────────┬──────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    TMCA 通道交互注意力  │  ← 创新点③
                              │  CNN特征 ↔ Mamba特征  │
                              │  (4个尺度, 一一对应)    │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ SpatialAttentionFusion│  ← 创新点④
                              │  空间注意力 + 残差融合  │
                              │  (4个尺度, 一一对应)    │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  SwinUMambaDecoder  │  ← 创新点⑤
                              │  (含 raw input skip) │
                              │  BasicResBlock      │
                              │  + UpBlock          │
                              │  + Deep Supervision │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   输出 (1×224×224)    │
                              │   二值分割掩膜         │
                              └──────────────────────┘
```

### 2.2 核心类关系

```
TextGuidedBreastUSSegmentation (主模型)
├── CIM (ContrastImprovementModule)            ← 输入增强
├── ResNet34Encoder (cnn_encoder)              ← CNN分支编码器
├── mamba_stem + VSSMEncoder (mamba_encoder)   ← Mamba分支编码器
├── trans_proj1~4 (Conv1×1)                    ← 通道投影对齐
├── tmca1~4 (TMCA)                              ← 通道交互注意力
├── xff1~4 (SpatialAttentionFusion)             ← 空间注意力融合
└── SwinUMambaDecoder (decoder)                ← Swin风格解码器
    ├── enc_raw (BasicResBlock)                → raw image skip
    ├── enc_stem (BasicResBlock)               → stem skip
    ├── skip2~4 + bn_skip (BasicResBlock)      → 多尺度跳跃连接
    ├── up0~4 (UpBlock)                         → 转置卷积上采样
    ├── refine (BasicResBlock)                  → 最终细化
    └── out_conv / out_layers (Conv1×1)         → 输出层
```

---

## 3. 核心创新模块详解

### 3.1 CIM — 对比度增强模块 (创新点①)

**文件位置**: [models/model.py](models/model.py#L150-L172)

**设计原理**:
CIM 放置在编码器之前，对输入图像进行对比度和细节增强。其核心思想是将图像分解为 **平滑层 (smooth)** 和 **细节层 (detail)**，再以可学习的缩放因子 α 重新组合。

```
输入图像 x
    │
    ├──→ smooth_conv(x) ──→ smooth_image     (均值滤波/平滑)
    │
    └──→ x - smooth_image ──→ detail_layer    (细节 = 原图 - 平滑)
    
输出 = smooth_image + α × detail_layer        (α 可学习)
```

**核心代码逻辑**:
```python
class ContrastImprovementModule(nn.Module):
    def __init__(self, in_channels=1, scaling_factor=1.5):
        # smooth_conv: 可学习的平滑卷积核
        self.smooth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=False)
        # α: 可学习的缩放因子 (初始值=scaling_factor)
        self.alpha = nn.Parameter(torch.tensor(scaling_factor))

    def forward(self, x):
        smooth_image = self.smooth_conv(x)         # 局部平滑
        detail_layer = x - smooth_image             # 提取细节
        output = smooth_image + self.alpha * detail_layer
        return output
```

**创新价值**:
- 自适应增强：α 可在训练中学习最适合当前数据的增强强度
- 保留原始信息：平滑卷积核初始化为均值滤波，保证训练初期不会破坏原始信息
- 即插即用：完全独立于后续编码器，可通过 `use_cim` 灵活开关

---

### 3.2 SS2D — 四方向选择性扫描 (创新点②)

**文件位置**: [models/swin_umamba.py](models/swin_umamba.py#L181-L360)

**设计原理**:
SS2D 是 VMamba 的核心操作，它将 Mamba（状态空间模型）从 1D 序列扩展到 2D 视觉任务。通过在 **4 个方向** 对特征图进行选择性扫描，建立全局空间依赖关系。

```
B×H×W×C 特征图
    │
    ├─→ 扫描方向1: H→W (左上到右下)
    ├─→ 扫描方向2: W→H (转置后扫描)
    ├─→ 扫描方向3: H→W 反向
    └─→ 扫描方向4: W→H 反向
    
每个方向: 
  展开为序列 → in_proj → Conv2d(depthwise) → SiLU → 
  x_proj (Δ, B, C) → selective_scan → 4方向求和 → out_norm → out_proj
```

**状态空间方程** (discrete form):
```
h_t = exp(Δ_t · A) · h_{t-1} + Δ_t · B_t · u_t
y_t = C_t^T · h_t + D · u_t
```

其中:
- `u_t`: 输入 token
- `h_t`: 隐藏状态 (d_state=16)
- `Δ_t`: 输入依赖的时间步长
- `A`: 状态转移矩阵 (可学习)
- `B_t, C_t`: 输入依赖的投影
- `D`: 跳跃连接参数

**跨平台兼容**:
```python
# 优先使用 CUDA 加速的 mamba_ssm
if HAS_MAMBA_SSM:
    self.selective_scan = selective_scan_fn  # CUDA 实现
else:
    self.selective_scan = selective_scan_pytorch  # 纯PyTorch回退
```

**创新价值**:
- 线性复杂度：相比 Transformer 的 O(N²)，SS2D 为 O(N)，适合高分辨率图像
- 全局感受野：4 方向扫描覆盖所有空间位置对
- 跨平台：Windows 下自动回退到纯 PyTorch 实现

---

### 3.3 VSSMEncoder — VMamba-Tiny 层次化编码器

**文件位置**: [models/swin_umamba.py](models/swin_umamba.py#L438-L519)

**架构配置 (VMamba-Tiny)**:
```
Stem: PatchEmbed2D (patch_size=2) → 48ch @ H/2×W/2
    ↓
Stage0: VSSLayer ×2, dim=96  → PatchMerging2D → 96ch @ H/4×W/4
Stage1: VSSLayer ×2, dim=192 → PatchMerging2D → 192ch @ H/8×W/8
Stage2: VSSLayer ×9, dim=384 → PatchMerging2D → 384ch @ H/16×W/16
Stage3: VSSLayer ×2, dim=768 → 768ch @ H/32×W/32
```

输出 6 个特征图: `[stem_input, stage0, stage1, stage2, stage3, stage4]`

---

### 3.4 TMCA — 通道交互注意力 (创新点③)

**文件位置**: [models/model.py](models/model.py#L57-L79)

**设计原理**:
TMCA (Transformer-style Mutual Channel Attention) 实现 CNN 和 Mamba 两个分支之间的**通道级信息交换**。它不是简单地将特征拼接或相加，而是通过注意力机制让两个分支相互"查询"。

```
CNN特征 x1 (B, C, H, W)          Mamba特征 x2 (B, C, H, W)
        │                                  │
        ├── SEBlock(x1) ──→ attn1 (B, C, 1) │
        │                                  ├── SEBlock(x2) ──→ attn2 (B, C, 1)
        │                                  │
        └──────────┬───────────────────────┘
                   │
          attn = attn2^T @ attn1   ← 通道交互矩阵 (C₂×C₁)
                   │
        ┌──────────┴──────────┐
        │                     │
   softmax(dim=2)       softmax(dim=1)
        │                     │
   output1 = attn1 @ x1  output2 = attn2^T @ x2
   (Mamba查询CNN)        (CNN查询Mamba)
```

**核心代码**:
```python
def forward(self, x1, x2):
    # SE注意力
    x1_attn = self.channel_self_attn1(x1)  # (B, C1, 1)
    x2_attn = self.channel_self_attn2(x2)  # (B, C2, 1)
    
    # 通道交互矩阵
    attn = x2_attn @ x1_attn^T              # (B, C2, C1)
    
    # 双向信息交换
    output1 = softmax(attn, dim=2) ⊗ x1     # Mamba增强CNN特征
    output2 = softmax(attn, dim=1)^T ⊗ x2   # CNN增强Mamba特征
    
    return output1, output2
```

**创新价值**:
- **双向感知**: 两个分支相互增强，而非单向融合
- **通道选择性**: 通过 SEBlock 筛选重要通道后再交互
- **参数量少**: 仅使用两个轻量 SEBlock

---

### 3.5 SpatialAttentionFusion — 空间注意力融合 (创新点④)

**文件位置**: [models/model.py](models/model.py#L86-L143)

**设计原理**:
在 TMCA 完成通道交互后，SpatialAttentionFusion 在**空间维度**上对两个增强后的特征进行自适应融合。通过生成空间注意力图来学习每个位置两个分支的相对重要性。

```
feat_a (Mamba增强后)           feat_b (CNN增强后)
        │                              │
        └────── concat ────────────────┘
                    │
            SpatialAttention
           (Conv→BN→ReLU→Conv→Sigmoid)
                    │
            spatial_weight (B, 1, H, W)
                    │
     ┌──────────────┼──────────────┐
     │              │              │
  feat_a_mod    feat_b_mod    (1 - spatial_weight)
  (DWConv)      (DWConv)
     │              │              │
     └──× weight ───┘              │
           │                       │
      feat_a_out             feat_b_out
           │                       │
           └────── + residual ─────┘
           │                       │
           └──── concat ───────────┘
                    │
               FusionProj (Conv1×1)
                    │
               输出 (B, C, H, W)
```

**关键公式**:
```
spatial_weight = Sigmoid(Conv(ReLU(BN(Conv([feat_a; feat_b])))))
feat_a_out = feat_a + DWConv(feat_a) × spatial_weight
feat_b_out = feat_b + DWConv(feat_b) × (1 - spatial_weight)
output = Conv([feat_a_out; feat_b_out])
```

**创新价值**:
- **空间自适应**: 不同空间位置自动决定哪个分支的特征更重要
- **残差保护**: `+ feat_a/b` 保留原始信息流，防止注意力破坏原始特征
- **互补权重**: `weight` 和 `1-weight` 形成互补关系，避免双倍激活

---

### 3.6 SwinUMambaDecoder — Swin风格解码器 (创新点⑤)

**文件位置**: [models/model.py](models/model.py#L291-L377)

**设计原理**:
解码器借鉴 Swin-UMamba 的设计理念，采用 UNet 结构，但使用 `BasicResBlock` 处理跳跃连接和 `UpBlock` (ConvTranspose2d + concat + BasicResBlock) 完成上采样。最关键的是引入了 **raw image skip connection**。

```
特征尺寸变化 (输入224×224):
    raw_image(3@224) ──BasicResBlock──→ s_raw(48@224)
    stem_out(48@112) ──BasicResBlock──→ s_stem(48@112)
    fused[0](64@56)   ──BasicResBlock──→ s2(64@56)
    fused[1](128@28)  ──BasicResBlock──→ s3(128@28)
    fused[2](256@14)  ──BasicResBlock──→ s4(256@14)
    fused[3](512@7)   ──BasicResBlock──→ bn(512@7)
    
解码路径:
    bn(512@7)  ──UpBlock(512→256)─→ + s4 ──→ d4(256@14)
    d4(256@14) ──UpBlock(256→128)─→ + s3 ──→ d3(128@28)
    d3(128@28) ──UpBlock(128→64)──→ + s2 ──→ d2(64@56)
    d2(64@56)  ──UpBlock(64→48)───→ + s_stem → d1(48@112)
    d1(48@112) ──UpBlock(48→48)───→ + s_raw  → d0(48@224)
    d0(48@224) ──BasicResBlock───→ refined(48@224)
    refined ────Conv1×1──────────→ output(1@224)
```

**深层监督 (Deep Supervision)**:
```python
if deep_supervision:
    return [
        out_layers[0](refined@224),   # 主输出 (w=1.0)
        out_layers[1](d2@56),         # 辅助输出 (w=0.5)
        out_layers[2](d3@28),         # 辅助输出 (w=0.5)
        out_layers[3](d4@14),         # 辅助输出 (w=0.5)
    ]
```

**BasicResBlock 结构**:
```
输入 x
  ├── Conv3×3 → InstanceNorm → LeakyReLU
  ├── Conv3×3 → InstanceNorm
  └── residual [+ 1×1卷积投影(通道不匹配时)]
  └── LeakyReLU → 输出
```

**UpBlock 结构**:
```
输入 x + skip
  ├── ConvTranspose2d (上采样×2)
  ├── concat([x_up, skip])
  └── BasicResBlock → 输出
```

**创新价值**:
- **Raw Image Skip**: 将未经任何处理的原始图像直接注入最浅层解码器，保留精细空间细节
- **三层跳跃连接**: raw_image → stem → encoder features，多级信息注入
- **轻量级**: 使用 InstanceNorm 替代 BatchNorm，BasicResBlock 比标准 ResBlock 更高效

---

### 3.7 双分支编码器设计

**CNN 分支 — ResNet34** ([models/model.py](models/model.py#L179-L213)):
```
输入 (3×224×224)
  → layer0: Conv7×7 + BN + ReLU + MaxPool  → 64ch @ 56×56 (feat1)
  → layer1: Bottleneck×3                    → 64ch @ 56×56 (feat2)
  → layer2: Bottleneck×4                    → 128ch @ 28×28 (feat3)
  → layer3: Bottleneck×6                    → 256ch @ 14×14 (feat4)
  → layer4: Bottleneck×3                    → 512ch @ 7×7
```

**Mamba 分支 — VMamba-Tiny**:
```
stem (Conv7×7 s=2 + IN) → 48ch @ 112×112
  → VSSMEncoder:
      PatchEmbed2D(s=2)→96ch @ 56×56
        → VSSLayer×2 → 96ch  (stage0)
        → PatchMerging → 192ch @ 28×28
        → VSSLayer×2 → 192ch (stage1)
        → PatchMerging → 384ch @ 14×14
        → VSSLayer×9 → 384ch (stage2)
        → PatchMerging → 768ch @ 7×7
        → VSSLayer×2 → 768ch (stage3)
```

**通道对齐** (通过 Conv1×1):
```
VMamba: [96, 192, 384, 768]
   ↓  Conv1×1  ↓
ResNet: [64, 128, 256, 512]
```

---

## 4. 完整数据流

### 4.1 训练阶段数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STEP 1: 数据加载                              │
├─────────────────────────────────────────────────────────────────────┤
│ train.py → Synapse_dataset / PreprocessedMoNuSegDataset             │
│    ├── .npz 文件 (BreastUS) / .npy 文件 (MoNuSeg patches)              │
│    ├── RandomGenerator: 随机旋转±15°、翻转、亮度/对比度/模糊/CoarseDropout│
│    ├── 弹性变形、染色增强、细胞尺度增强 (MoNuSeg专用)                    │
│    ├── Resize → 224×224                                              │
│    ├── 单通道→3通道复制                                               │
│    └── Min-Max归一化 → [0,1]                                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 2: 输入增强 (CIM)                              │
├─────────────────────────────────────────────────────────────────────┤
│ image (B,3,224,224) + raw_image (保存原始副本)                          │
│    ↓                                                                │
│ cim(image):                                                         │
│    smooth = Conv2d(image)           ← 可学习平滑                     │
│    detail = image - smooth                                           │
│    enhanced = smooth + α * detail   ← α 可学习 (初始0.5)             │
│    ↓                                                                │
│ image = enhanced  (后续编码器使用增强后的图像)                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│   STEP 3a: CNN分支编码         │   │   STEP 3b: Mamba分支编码       │
├───────────────────────────────┤   ├───────────────────────────────┤
│ cnn_encoder(image):           │   │ mamba_stem(image):             │
│   layer0 → 64@56              │   │   Conv7×7 s=2 → 48@112         │
│   layer1 → 64@56 (feat1)      │   │                                │
│   layer2 → 128@28 (feat2)     │   │ mamba_encoder(stem_out):       │
│   layer3 → 256@14 (feat3)     │   │   → stage0(S2D×2): 96@56       │
│   layer4 → 512@7  (feat4)     │   │   → stage1(S2D×2): 192@28      │
│                                │   │   → stage2(S2D×9): 384@14     │
│ [64,128,256,512]               │   │   → stage3(S2D×2): 768@7       │
│                                │   │                                │
│                                │   │ mamba_features = outs[1:5]     │
│                                │   │ trans_proj 1×1Conv对齐通道     │
│                                │   │   → [64,128,256,512]            │
└───────────────────────────────┘   └───────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                STEP 4: TMCA 通道交互 (每个尺度独立)                     │
├─────────────────────────────────────────────────────────────────────┤
│ For scale i in [1,2,3,4]:                                            │
│   cnn_tmca_i, mamba_tmca_i = TMCA_i(mamba_aligned_i, cnn_feat_i)     │
│                                                                      │
│   内部:                                                               │
│     attn1 = SEBlock(cnn_feat)     ← 压缩CNN通道                       │
│     attn2 = SEBlock(mamba_feat)   ← 压缩Mamba通道                     │
│     attn = attn2 @ attn1^T        ← 通道交互矩阵                      │
│     cnn_tmca  = softmax(attn, dim=2) ⊗ cnn_feat   (Mamba查询CNN)     │
│     mamba_tmca = softmax(attn, dim=1)^T ⊗ mamba_feat (CNN查询Mamba)  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│           STEP 5: SpatialAttentionFusion 空间融合 (每个尺度独立)        │
├─────────────────────────────────────────────────────────────────────┤
│ For scale i in [1,2,3,4]:                                            │
│   fused_i = XFF_i(mamba_tmca_i, cnn_tmca_i)                          │
│                                                                      │
│   内部:                                                               │
│     concat = [mamba_tmca; cnn_tmca]                                  │
│     spatial_w = Sigmoid(Conv(ReLU(BN(Conv(concat)))))  ← 空间权重     │
│     feat_a_out = mamba_tmca + DWConv(mamba_tmca) × spatial_w         │
│     feat_b_out = cnn_tmca + DWConv(cnn_tmca) × (1-spatial_w)         │
│     fused_i = Conv([feat_a_out; feat_b_out])                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                   fused_features = [fused1, fused2, fused3, fused4]
                   stem_out (48@112), raw_image (3@224)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                STEP 6: SwinUMambaDecoder 解码                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  skip处理:                                                            │
│    s_raw = BasicResBlock(raw_image)    → 48@224                      │
│    s_stem = BasicResBlock(stem_out)    → 48@112                      │
│    s2 = BasicResBlock(fused1)          → 64@56                       │
│    s3 = BasicResBlock(fused2)          → 128@28                      │
│    s4 = BasicResBlock(fused3)          → 256@14                      │
│    bn = BasicResBlock(fused4)          → 512@7                       │
│                                                                      │
│  逐层上采样:                                                           │
│    bn(512@7)  ──UpBlock──→ + s4 ──→ d4(256@14)                      │
│    d4(256@14) ──UpBlock──→ + s3 ──→ d3(128@28)                      │
│    d3(128@28) ──UpBlock──→ + s2 ──→ d2(64@56)                       │
│    d2(64@56)  ──UpBlock──→ + s_stem → d1(48@112)                    │
│    d1(48@112) ──UpBlock──→ + s_raw → d0(48@224)                     │
│    d0(48@224) ──BasicResBlock──→ refined(48@224)                     │
│                                                                      │
│  输出:                                                                │
│    无深监督: out_conv(refined) → (B, 1, 224, 224)                     │
│    有深监督: [out0(refined), out1(d2), out2(d3), out3(d4)]           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 7: 损失计算 & 反向传播                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  deep_supervision_loss(outputs, target):                              │
│    main_loss = FocalTverskyLoss(outputs[0], target) * 1.0            │
│    aux1_loss  = FocalTverskyLoss(upsample(outputs[1]), target) * 0.5 │
│    aux2_loss  = FocalTverskyLoss(upsample(outputs[2]), target) * 0.5 │
│    aux3_loss  = FocalTverskyLoss(upsample(outputs[3]), target) * 0.5 │
│    total = main + aux1 + aux2 + aux3                                  │
│                                                                      │
│  optimizer.step()  →  AdamW (lr=3e-4, wd=3e-5)                       │
│  scheduler.step()  →  WarmupPolyLR (10epoch warmup, power=0.9)       │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 推理阶段数据流

```
┌──────────────────────────────────────────────────────────────────┐
│                     测试/推理数据流                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  策略1: Sliding Window (高分辨率图像, 默认)                         │
│    ├── 不重叠滑窗 (stride = patch_size = 224)                       │
│    ├── 逐patch推理 → Sigmoid → 阈值0.5 → 拼接                       │
│    └── 计算指标 (Dice, HD95, ASD, IoU, Recall, Precision)          │
│                                                                   │
│  策略2: Direct Resize                                              │
│    ├── 整图Resize → 224×224                                        │
│    ├── 单次推理 → Sigmoid → 阈值0.5                                 │
│    └── 计算指标                                                     │
│                                                                   │
│  预处理 (与训练严格一致):                                            │
│    1. 加载原始图像 (.tif / .npz / .h5)                              │
│    2. 维度规范化 (C,H,W) 或 (H,W)                                   │
│    3. Resize到224×224                                              │
│    4. Min-Max归一化到[0,1]                                         │
│    5. 单通道→3通道复制                                              │
│    6. 转Tensor + 添加batch维度                                     │
│                                                                   │
│  后处理:                                                           │
│    logits → Sigmoid → >0.5 二值化 → 2D mask                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. 训练策略

### 5.1 优化器配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | AdamW | 带解耦权重衰减的Adam |
| 学习率 | 3e-4 | 初始学习率 |
| Weight Decay | 3e-5 | 权重衰减系数 |
| Betas | (0.9, 0.999) | Adam动量参数 |

### 5.2 学习率调度 — WarmupPolyLR

```
lr
 │
 │  Warmup阶段         Poly衰减阶段
 │  (10 epochs)        (power=0.9)
 │     ╱               ╲
 │    ╱                  ╲
 │   ╱                     ╲___________
 │  ╱                                   ╲_____
 │ ╱                                           ╲___
 └─────────────────────────────────────────────────→ iteration
     base_lr/1000                         min_lr=1e-6
```

文件位置: [trainer.py](trainer.py#L103-L139)

### 5.3 其他训练技巧

- **Bias-prior 初始化**: 根据训练集前景占比 π 初始化输出层 bias = `log(π/(1-π))`，加速前期收敛
- **早停 (EarlyStopping)**: 监控验证集 Dice，patience 个 epoch 无提升则停止并恢复最佳权重
- **确定性训练**: 固定随机种子 (seed=42)，关闭 cudnn benchmark 确保可复现
- **冻结Mamba编码器**: 可选冻结预训练的VMamba编码器，仅训练融合模块和解码器

---

## 6. 损失函数与评估指标

### 6.1 Focal Tversky Loss

**文件位置**: [utils.py](utils.py#L58-L91)

```
TI = TP / (TP + α·FP + β·FN)
L = (1 - TI)^γ
```

| 参数 | 默认值 | 作用 |
|------|--------|------|
| α | 0.6 | 假阳性惩罚 (α>β 更多惩罚误检) |
| β | 0.4 | 假阴性惩罚 |
| γ | 2.5 | Focal指数 (放大困难样本的loss) |

**设计理由**: 医学图像中前景(病变/细胞核)占比小，类别严重不平衡。Focal Tversky Loss 通过 α>β 对假阳性更敏感（减少误报），γ 让模型专注难分样本。

### 6.2 Dice + Focal Loss (备选)

```
L = 0.5 × DiceLoss + 0.5 × FocalLoss(α=0.8, γ=2.0)
```

### 6.3 评估指标 (MICCAI 标准)

| 指标 | 范围 | 含义 |
|------|------|------|
| **Dice** | [0, 1], ↑ | 重叠度系数 |
| **HD95** | ≥0, ↓ | 95% Hausdorff距离 (边界误差) |
| **ASD** | ≥0, ↓ | 平均表面距离 |
| **IoU** | [0, 1], ↑ | 交并比 (Jaccard) |
| **Recall** | [0, 1], ↑ | 敏感性 (TP/(TP+FN)) |
| **Precision** | [0, 1], ↑ | 精确度 (TP/(TP+FP)) |

---

## 7. 代码模块组织

```
Swinmamba_resnet34_CIM_swindecoder/
├── models/
│   ├── __init__.py            → 导出 BreastUSSegmentation
│   ├── model.py               → 主模型 TextGuidedBreastUSSegmentation
│   │   ├── ConvBNReLU                基础卷积块
│   │   ├── SEBlockNoShortcut         SE通道注意力
│   │   ├── TMCA                      通道交互注意力
│   │   ├── SpatialAttentionFusion    空间注意力融合
│   │   ├── ContrastImprovementModule 对比度增强模块 (CIM)
│   │   ├── ResNet34Encoder           CNN分支编码器
│   │   ├── BasicResBlock             Swin解码器基础块
│   │   ├── UpBlock                   Swin解码器上采样块
│   │   └── SwinUMambaDecoder         Swin风格解码器
│   │
│   └── swin_umamba.py         → VMamba-Tiny 编码器 + SS2D
│       ├── selective_scan_pytorch    纯PyTorch选择性扫描 (回退方案)
│       ├── PatchEmbed2D              图像→Patch嵌入
│       ├── PatchMerging2D            Patch合并 (下采样)
│       ├── SS2D                      2D选择性扫描 (核心算子)
│       ├── VSSBlock                  VSS基础块
│       ├── VSSLayer                  VSS层 (多VSSBlock)
│       ├── VSSMEncoder               VMamba层次化编码器
│       ├── BasicResBlock + UpBlock   (与model.py重复定义)
│       └── SwinUMamba                完整SwimUMamba模型 (参考实现)
│
├── datasets/
│   ├── __init__.py
│   ├── dataset_synapse.py      → 数据集加载 & 增强
│   │   ├── Synapse_dataset            .npz / .h5 数据集
│   │   ├── MoNuSeg_dataset            .tif + .xml 数据集
│   │   ├── RandomGenerator            训练增强pipeline
│   │   └── ValGenerator               验证预处理pipeline
│   │
│   └── preprocessed_monuseg.py → 预处理patches加载器
│       └── PreprocessedMoNuSegDataset
│
├── train.py                    → 训练入口 & 参数配置
├── trainer.py                  → 训练循环 & 验证 & 调度器
│   ├── trainer_synapse                主训练函数
│   ├── deep_supervision_loss          深监督损失
│   ├── validate_model                 验证函数
│   ├── WarmupPolyLR                   Warm-up+Poly学习率
│   └── EarlyStopping                  早停机制
│
├── test.py                     → 测试/推理脚本
│   ├── test_single_case              滑窗推理
│   ├── test_single_case_direct_resize 直接resize推理
│   └── main                          完整测试流程
│
├── utils.py                    → 损失函数 & 评估指标
│   ├── DiceLoss                       二分类Dice Loss
│   ├── FocalLoss                      二分类Focal Loss
│   ├── FocalTverskyLoss              Focal Tversky Loss
│   ├── SegmentationLoss               统一损失函数封装
│   ├── calculate_metric_percase       6项指标计算 (Dice/HD95/ASD/IoU/Recall/Precision)
│   └── test_single_volume            3D体数据推理
│
├── data/                       → 数据集 & 预训练权重
│   ├── BreastUS_BUSI/                乳腺超声数据
│   └── pretrained/vmamba/            VMamba-Tiny预训练权重
│
├── output/                     → 训练输出 (模型权重、日志、TensorBoard)
├── train_test.md               → 训练启动指南
└── architecture_analysis.md    → 本文档
```

---

## 附录: 参数量统计 (默认配置)

| 模块 | 参数量 | 是否可训练 |
|------|--------|-----------|
| **CIM** | ~18 | ✅ |
| **CNN Encoder (ResNet34)** | ~21.3M | ✅ (可冻结) |
| **Mamba Stem** | ~2.4K | ✅ |
| **Mamba Encoder (VSSM)** | ~13.4M | 默认冻结 |
| **通道投影层 (4×Conv1×1)** | ~0.55M | ✅ |
| **TMCA ×4** | ~1.1M | ✅ |
| **XFF ×4 (SpatialAttentionFusion)** | ~1.6M | ✅ |
| **SwinUMambaDecoder** | ~4.0M | ✅ |
| **总计** | **~41.8M** | 可训练 ~28.4M |

---

> **文档生成日期**: 2026-07-03
> **项目**: SwinMamba + ResNet34 + CIM + SwinDecoder — 双分支医学图像分割
> **适用数据集**: BreastUS (乳腺超声) / MoNuSeg (细胞核分割)
