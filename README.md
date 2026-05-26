# SwinMamba-ResNet34-CIM-SwinDecoder

Dual-branch medical image segmentation combining CNN local details with Mamba global context.

## Architecture

```
Input (224×224×3)
    │
    ├── CIM (Contrast Improvement Module) ── optional input enhancement
    │
    ├── CNN Branch: ResNet34 (ImageNet pretrained) ── local texture/detail
    │
    ├── Mamba Branch: VMamba-Tiny Encoder ── SS2D 4-directional selective scan, global context
    │
    ├── TMCA (Channel Exchange Attention) ── cross-branch feature interaction at 4 scales
    │
    ├── SpatialAttentionFusion ── learnable spatial gating + residual fusion
    │
    └── SwinUMambaDecoder ── UNet-style with raw input skip + deep supervision
            │
            └── Output (224×224×1) binary mask
```

### Key Components

| Module | Description |
|--------|-------------|
| **ResNet34 Encoder** | CNN branch for fine local textures, ImageNet pretrained |
| **VMamba-Tiny Encoder** | SS2D-based Mamba backbone with 4-directional scanning for global context |
| **CIM** | Contrast Improvement Module — enhances input detail before encoding |
| **TMCA** | Channel exchange attention — fuses CNN and Mamba features via cross-channel SE |
| **SpatialAttentionFusion** | Dynamic spatial gating with residual connections for multi-scale fusion |
| **SwinUMambaDecoder** | Upsampling decoder with raw image skip, InstanceNorm, deep supervision |
| **Bias-Prior Init** | Initializes output bias from foreground pixel ratio for faster convergence |
| **Focal Tversky Loss** | Handles class imbalance with tunable FP/FN penalty (α=0.6, β=0.4, γ=2.5) |

## Supported Datasets

| Dataset | Task | #Classes |
|---------|------|----------|
| **MoNuSeg** | Nuclei segmentation | 1 (binary) |
| **BreastUS (BUSI)** | Breast ultrasound tumor segmentation | 1 (binary) |
| **Synapse** | Multi-organ CT segmentation | 9 |

## Installation

```bash
# PyTorch (CUDA 11.8+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Dependencies
pip install numpy scipy medpy SimpleITK tensorboardX tqdm einops timm h5py opencv-python scikit-image Pillow

# Optional: mamba-ssm for CUDA-accelerated selective scan (Linux only)
pip install mamba-ssm
# On Windows, a pure-PyTorch fallback is used automatically
```

## Quick Start

```bash
# Test on CPU (pure PyTorch fallback, no GPU required)
python test.py --dataset MoNuSeg --split test --model <path_to_checkpoint> --test-mode sliding_window --deep-supervision 1

# Train on MoNuSeg with default settings
python train.py --dataset MoNuSeg --batch_size 16 --max_epochs 130 --load_pretrained True
```

## Training

```bash
# Standard training (recommended)
python train.py \
    --dataset MoNuSeg \
    --batch_size 16 \
    --max_epochs 130 \
    --base_lr 0.0003 \
    --load_pretrained True \
    --seed 42

# With deep supervision + focal tversky loss
python train.py \
    --dataset MoNuSeg \
    --batch_size 16 \
    --max_epochs 150 \
    --base_lr 3e-4 \
    --load_pretrained True \
    --use_cim 1 \
    --cim_scaling_factor 0.5 \
    --loss_type focal_tversky \
    --deep_supervision 1 \
    --early_stopping_patience 50

# Breast ultrasound
python train.py --dataset BreastUS --batch_size 8 --max_epochs 150 --load_pretrained True
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `BreastUS` | `MoNuSeg`, `BreastUS`, `Synapse` |
| `--batch_size` | `16` | Batch size per GPU |
| `--max_epochs` | `130` | Total training epochs |
| `--base_lr` | `0.0003` | Initial learning rate (AdamW) |
| `--img_size` | `224` | Input resolution |
| `--use_cim` | `1` | Enable contrast improvement |
| `--load_pretrained` | `False` | Load VMamba-Tiny ImageNet weights |
| `--loss_type` | `focal_tversky` | `dice_focal` or `focal_tversky` |
| `--deep_supervision` | `0` | Enable multi-scale auxiliary loss |
| `--early_stopping_patience` | `0` | Patience epochs (0=disabled) |
| `--foreground_prior` | `0.0` | Bias-prior init (0=auto-compute) |
| `--freeze_mamba_encoder` | `0` | Freeze VMamba after loading pretrained weights |

### Pretrained Weights

Download VMamba-Tiny ImageNet pretrained weights:

```bash
# Auto-download (in code)
# Or manually from:
# https://github.com/MzeroMiko/VMamba/releases/tag/20240218
# File: vssmtiny_dp01_ckpt_epoch_292.pth
# Place at: ./data/pretrained/vmamba/vmamba_tiny_e292.pth
```

## Testing

```bash
python test.py \
    --dataset MoNuSeg \
    --split test \
    --model ./output/TU_MoNuSeg224/TU_pretrain_VMambaTiny_.../best_model.pth \
    --test-mode sliding_window \
    --deep-supervision 1
```

## Output Structure

```
output/
└── TU_MoNuSeg224_pretrain_VMambaTiny_skip3_epo130_bs16_lr0.0003_224_s42_20260509_045946/
    ├── best_model.pth          # Best checkpoint (by validation Dice)
    ├── epoch_50.pth
    ├── epoch_100.pth
    ├── log.txt                 # Training log
    └── log/                    # TensorBoard events
```

## Evaluation Metrics

Dice, HD95, ASD (Average Surface Distance), IoU, Recall, Precision — all computed via `calculate_metric_percase()` in [utils.py](utils.py).

## Project Structure

```
├── train.py                  # Training entry point + argparse
├── trainer.py                # Training loop, WarmupPolyLR, EarlyStopping, validation
├── test.py                   # Test/inference script
├── utils.py                  # Loss functions, metrics (Dice, HD95, IoU)
├── models/
│   ├── model.py              # Dual-branch model: ResNet34 + VMamba-Tiny + TMCA + CIM + Decoder
│   └── swin_umamba.py        # SS2D, VSSBlock, VSSMEncoder, BasicResBlock, UpBlock, SwinUMamba
├── datasets/
│   ├── dataset_synapse.py    # Data loading, augmentations (BreastUS, Synapse, MoNuSeg)
│   └── preprocessed_monuseg.py  # Preprocessed MoNuSeg patch dataset
└── data/
    └── pretrained/
        └── vmamba/           # Place VMamba-Tiny checkpoint here
```

## Platform Support

- **Linux** — full support with CUDA-accelerated `mamba-ssm` selective scan
- **Windows** — supported via pure-PyTorch selective scan fallback (no `mamba-ssm` needed)

## References

- Swin-UMamba: [arXiv:2402.03302](https://arxiv.org/abs/2402.03302)
- VMamba: [https://github.com/MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba)
- Focal Tversky Loss: Abraham & Khan, ISBI 2019
