# 模型训练启动指南

## 🚀 快速启动

### 1️⃣ 最简单的启动方式（使用默认参数）
```bash
python train.py
```

测试：python test.py --dataset MoNuSeg --split test --model ./output/TU_MoNuSeg224/TU_pretrain_WaveFormer_epo150_bs16_lr0.0003_224_s42_20260509_045946/best_model.pth --test-mode sliding_window --deep-supervision 1 

### 2️⃣ 使用MoNuSeg数据集训练
```bash
python train.py --dataset MoNuSeg --batch_size 16 --max_epochs 130
```

### 3️⃣ 使用乳腺超声数据集训练
```bash
python train.py --dataset MoNuSeg --batch_size 16 --max_epochs 150 --load_pretrained True --preprocessed_dir preprocessed_data  --use_cim 1 --base_lr 3e-4 --use_cell_specific_aug 0 --cim_scaling_factor 0.5  --early_stopping_patience 50 --loss_type focal_tversky --deep_supervision 1 --foreground_prior 0.0
``` 
--loss_type	focal_tversky	损失类型选择
--tversky_alpha	0.6	FP 惩罚权重
--tversky_beta	0.4	FN 惩罚权重
--tversky_gamma	2.5	Focal 指数
--foreground_prior	0.0	前景占比（0=禁用 bias-prior）
---

## 📋 完整参数说明

### 数据相关参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--dataset` | `BreastUS` | 数据集名称: `BreastUS`, `Synapse`, `MoNuSeg` |
| `--root_path` | 根据数据集自动设置 | 训练数据根目录 |
| `--list_dir` | 根据数据集自动设置 | 数据列表目录 |
| `--preprocessed_dir` | `preprocessed_data` | 预处理数据目录 |
| `--val_split` | `val` | 验证集选择: `val` 或 `test` |

### 模型相关参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--img_size` | `224` | 输入图像大小 (224×224) |
| `--num_classes` | 自动 | 输出类数 (二分类为1) |
| `--load_pretrained` | `False` | 是否加载预训练权重 |
| `--text_model_name` | `emilyalsentzer/Bio_ClinicalBERT` | 文本编码器模型 |

### 训练相关参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--max_epochs` | `130` | 最大训练轮数 |
| `--max_iterations` | `30000` | 最大迭代次数 |
| `--batch_size` | `16` | 每GPU批大小 (推荐: 4-16) |
| `--base_lr` | `0.0003` | 初始学习率 |
| `--n_gpu` | `1` | GPU数量 |
| `--seed` | `42` | 随机种子 |
| `--deterministic` | `1` | 是否使用确定性训练 |

---

## 💡 常见训练配置

### 配置1️⃣: 快速测试（用于验证代码）
```bash
python train.py \
    --dataset MoNuSeg \
    --batch_size 4 \
    --max_epochs 10 \
    --base_lr 0.0003 \
    --load_pretrained False
```
- ⏱️ 预计时间: ~30分钟 (单GPU V100)
- 📊 用途: 验证代码是否正确运行

### 配置2️⃣: 标准训练（推荐）
```bash
python train.py \
    --dataset MoNuSeg \
    --batch_size 16 \
    --max_epochs 130 \
    --base_lr 0.0003 \
    --load_pretrained True \
    --seed 42
```
- ⏱️ 预计时间: ~6-8小时 (单GPU V100)
- 📊 用途: 完整的模型训练

### 配置3️⃣: 高精度训练（更好的效果）
```bash
python train.py \
    --dataset MoNuSeg \
    --batch_size 8 \
    --max_epochs 200 \
    --base_lr 0.0001 \
    --load_pretrained True \
    --deterministic 1 \
    --seed 42
```
- ⏱️ 预计时间: ~12-15小时 (单GPU V100)
- 📊 用途: 追求最高精度

### 配置4️⃣: 多GPU训练（分布式）
```bash
python train.py \
    --dataset MoNuSeg \
    --batch_size 16 \
    --max_epochs 130 \
    --base_lr 0.0003 \
    --n_gpu 2 \
    --load_pretrained True
```
- ⏱️ 预计时间: ~3-4小时 (双GPU V100)
- 📊 用途: 加速训练

---

## 🎯 推荐的参数组合

### 对于MoNuSeg数据集（细胞核分割）
```bash
# 基础配置
--dataset MoNuSeg
--batch_size 16        # MoNuSeg数据集较小，可用较大batch_size
--max_epochs 130
--base_lr 0.0003
--load_pretrained True
```

### 对于BreastUS数据集（乳腺超声分割）
```bash
# 基础配置
--dataset BreastUS
--batch_size 8         # 较小数据集，用较小batch_size
--max_epochs 150
--base_lr 0.0003
--load_pretrained True
```

### 对于Synapse数据集（器官分割）
```bash
# 基础配置
--dataset Synapse
--batch_size 24
--max_epochs 150
--base_lr 0.0005
--load_pretrained True
```

---

## ⚙️ 学习率选择建议

| Batch Size | 推荐学习率 | 说明 |
|-----------|----------|------|
| 4 | 0.0001 - 0.0002 | 小batch，用较小学习率 |
| 8 | 0.0002 - 0.0003 | 中等batch |
| 16 | 0.0003 - 0.0005 | 标准batch (推荐) |
| 24+ | 0.0005 - 0.001 | 大batch，可用较大学习率 |

---

## 🔧 高级参数调整

### 使用随机种子确保可重复性
```bash
python train.py \
    --dataset MoNuSeg \
    --seed 42 \
    --deterministic 1
```

### 调整模型输入大小
```bash
python train.py \
    --img_size 256 \
    --batch_size 8  # 注意：更大的img_size需要更小的batch_size
```

### 自定义数据目录
```bash
python train.py \
    --root_path /path/to/your/data \
    --list_dir /path/to/your/lists \
    --preprocessed_dir /path/to/preprocessed
```

---

## 📊 输出文件说明

训练完成后，会在以下目录生成输出：

```
model/
├── TU_MoNuSeg224_pretrain_R50-ViT-B_16_skip3_epo130_bs16/
│   ├── best_model.pth           # 最佳模型（验证集上的最优）
│   ├── epoch_50.pth             # 第50个epoch的检查点
│   ├── epoch_100.pth            # 第100个epoch的检查点
│   └── ...
├── TU_BreastUS224_pretrain_R50-ViT-B_16_skip3_epo130_bs16/
│   └── ...
```

---

## 🚨 常见问题

### Q1: 显存不足 (CUDA Out of Memory)
**解决方案：**
```bash
# 减小batch_size
python train.py --batch_size 4  # 从16改为4

# 或减小输入尺寸
python train.py --img_size 192 --batch_size 8

# 或使用梯度累积（trainer.py中支持）
```

### Q2: 预训练权重下载失败
**解决方案：**
```bash
# 方法1: 手动下载
python -c "from transformers import AutoModel; AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')"

# 方法2: 不使用预训练权重（需要更多epoch）
python train.py --load_pretrained False --max_epochs 200
```

### Q3: 如何使用自定义文本提示
在数据集中添加 `texts.json`:
```json
{
    "image_001": "hypoechoic breast lesion with irregular boundary",
    "image_002": "benign cyst with smooth margins"
}
```

### Q4: 如何恢复训练（从checkpoint恢复）
在 `trainer.py` 中添加：
```python
# 加载checkpoint
checkpoint = torch.load('model/path/epoch_50.pth')
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## 🎓 完整示例

### 示例1: 从头开始训练（10个epoch快速测试）
```bash
cd d:\Pycharm_project\NewAI3
python train.py \
    --dataset MoNuSeg \
    --batch_size 4 \
    --max_epochs 10 \
    --load_pretrained False
```

### 示例2: 使用预训练权重进行完整训练
```bash
cd d:\Pycharm_project\NewAI3
python train.py \
    --dataset MoNuSeg \
    --batch_size 16 \
    --max_epochs 130 \
    --base_lr 0.0003 \
    --load_pretrained True \
    --seed 42 \
    --deterministic 1
```

### 示例3: 在乳腺超声数据集上训练
```bash
cd d:\Pycharm_project\NewAI3
python train.py \
    --dataset BreastUS \
    --batch_size 8 \
    --max_epochs 150 \
    --base_lr 0.0003 \
    --load_pretrained True
```

---

## 📈 监控训练进程

训练过程中会输出：
```
Epoch 1/130 [=====>        ] 45% - Loss: 0.456, IoU: 0.82
Epoch 2/130 [==========>   ] 78% - Loss: 0.312, IoU: 0.88
...
```

监控的指标：
- **Loss**: 训练损失值（越低越好）
- **IoU**: Intersection over Union（越高越好，0-1）
- **Dice**: Dice系数（越高越好，0-1）

---

## ✅ 参数验证清单

启动训练前检查：
- [ ] 数据集路径正确
- [ ] GPU显存充足 (建议≥11GB)
- [ ] 网络连接正常（用于下载预训练模型）
- [ ] Python环境依赖已安装
- [ ] CUDA/cuDNN版本兼容

---

## 🔗 相关资源

- 论文配置：使用 `--max_epochs 130 --batch_size 16 --base_lr 0.0003`
- 数据集：MoNuSeg, BreastUS, Synapse
- 模型保存：`model/` 目录
- 日志记录：`logs/` 目录 (如果启用)

