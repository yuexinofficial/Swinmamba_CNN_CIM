"""
MoNuSeg 细胞核分割模型测试脚本 - 符合2024年最新医学图像分割评估标准

支持的数据集:
- BreastUS (乳腺超声图像)
- MoNuSeg (多器官细胞核分割)

使用方法:
1. 测试MoNuSeg测试集:
   python test.py --dataset MoNuSeg --split test --model checkpoints/best_model.pth

2. 测试MoNuSeg训练集:
   python test.py --dataset MoNuSeg --split train --model checkpoints/best_model.pth

3. 自定义参数:
   python test.py --dataset MoNuSeg --model path/to/model.pth --img-size 224 --output results/

4. 控制模型组件（必须与训练时保持一致）:
   python test.py --dataset MoNuSeg --model model.pth --use-cct 0 --use-cca 1 --use-cim 1
   # 关闭CCT但保留CCA和CIM

验证指标 (符合MICCAI标准):
- Dice Similarity Coefficient (DSC): 重叠度指标，范围[0,1]，越高越好
- Hausdorff Distance 95% (HD95): 边界距离指标，越低越好（对异常值鲁棒）
- Average Surface Distance (ASD): 平均表面距离，越低越好
- Intersection over Union (IoU): 交并比，范围[0,1]，越高越好
- Recall/Sensitivity: 召回率/敏感性
- Precision: 精确度

数据预处理流程（与训练时严格一致）:
1. 加载原始图像 (.tif) 和标注 (.xml)
2. Resize到指定尺寸 (默认224x224)
3. Min-Max归一化到[0,1]
4. 单通道复制为3通道（如需要）
5. 转换为Tensor并添加batch维度
6. 模型推理 + Sigmoid激活
7. 阈值0.5二值化得到预测结果
8. 计算各项指标（Dice, HD95, ASD, IoU, Recall, Precision）
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from tqdm import tqdm

from models.model import TextGuidedBreastUSSegmentation
from utils import calculate_metric_percase


def load_data(file_path, split='test', dataset='BreastUS'):
    """Load image and label from file (support both .h5 and .npz formats, and MoNuSeg .tif/.xml)
    
    Args:
        file_path: 文件路径（对于MoNuSeg可以是.tif或.xml）
        split: 数据分割 ('train' 或 'test')
        dataset: 数据集类型
        
    Returns:
        image: numpy array (C, H, W) or None
        label: numpy array (H, W) or None
    """
    try:
        if dataset == 'MoNuSeg':
            # MoNuSeg数据集：.tif图像和.xml标注
            # 确定文件路径
            if file_path.endswith('.xml'):
                annotation_path = file_path
                base_name = os.path.basename(file_path).replace('.xml', '')
                # 查找对应的.tif文件
                tif_path = file_path.replace('.xml', '.tif')
                if not os.path.exists(tif_path):
                    # 尝试在Tissue Images目录中查找
                    tif_path = os.path.join(
                        os.path.dirname(os.path.dirname(annotation_path)),
                        'Tissue Images',
                        base_name + '.tif'
                    )
            elif file_path.endswith('.tif'):
                tif_path = file_path
                base_name = os.path.basename(file_path).replace('.tif', '')
                annotation_path = file_path.replace('.tif', '.xml')
            else:
                logging.error(f"Unsupported file format for MoNuSeg: {file_path}")
                return None, None
            
            # 检查文件是否存在
            if not os.path.exists(tif_path):
                logging.error(f"Image file not found: {tif_path}")
                return None, None
            if not os.path.exists(annotation_path):
                logging.error(f"Annotation file not found: {annotation_path}")
                return None, None
            
            # 读取图像
            from PIL import Image
            image = Image.open(tif_path)
            image = np.array(image)
            
            # 如果图像是RGBA，去除alpha通道
            if image.ndim == 3 and image.shape[-1] == 4:
                image = image[:, :, :3]
            
            # 转换为 (C, H, W) 格式
            if image.ndim == 3 and image.shape[-1] == 3:
                image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            elif image.ndim == 2:
                # 灰度图扩展为3通道
                image = np.expand_dims(image, axis=0)
                image = np.repeat(image, 3, axis=0)
            
            # 解析XML标注生成二值mask
            import xml.etree.ElementTree as ET
            import skimage.draw
            
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            img_shape = (image.shape[1], image.shape[2])  # (H, W)
            binary_mask = np.zeros(img_shape, dtype=np.uint8)
            
            # 遍历所有Region
            for region in root.findall('.//Region'):
                vertices = []
                
                # 提取所有顶点坐标
                for vertex in region.findall('.//Vertex'):
                    x_str = vertex.get('X')
                    y_str = vertex.get('Y')
                    if x_str is not None and y_str is not None:
                        x = float(x_str)
                        y = float(y_str)
                        vertices.append([x, y])
                
                if len(vertices) >= 3:
                    vertices = np.array(vertices)
                    
                    # 使用skimage绘制多边形
                    rr, cc = skimage.draw.polygon(vertices[:, 1], vertices[:, 0], shape=img_shape)
                    binary_mask[rr, cc] = 1
            
            return image, binary_mask
            
        elif split == 'train' or file_path.endswith('.npz'):
            # Load .npz format (training data)
            data = np.load(file_path)
            image, label = data['image'], data['label']
            return image, label
        else:
            # Load .h5 format (test data)
            with h5py.File(file_path, 'r') as f:
                # Try to load image and label
                if 'image' in f and 'label' in f:
                    image = np.array(f['image'])
                    label = np.array(f['label'])
                    return image, label
                # If not, try alternative keys
                elif 'data' in f and 'seg' in f:
                    image = np.array(f['data'])
                    label = np.array(f['seg'])
                    return image, label
                else:
                    # List available keys
                    keys = list(f.keys())
                    logging.warning(f'Unknown H5 structure. Available keys: {keys}')
                    return None, None
    except Exception as e:
        logging.error(f'Error loading file {file_path}: {e}')
        import traceback
        traceback.print_exc()
        return None, None


def test_single_case(image, label, net, device, n_classes=2, img_size=224):
    """Test a single case using non-overlapping patches (与训练时一致)
    
    Args:
        image: numpy array - 输入图像 (C, H, W)
        label: numpy array - 真实标签 (H, W)
        net: 模型
        device: 设备
        n_classes: 类别数（对于二分类为2）
        img_size: patch尺寸
        
    Returns:
        metric_list: [(dice, hd95, asd, iou, recall, precision), ...]
        pred_map: 预测结果 (H, W) - 拼接后的完整预测图
    """
    # ==================== 数据预处理 ====================
    
    # 步骤1: 确保图像格式正确
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            # MoNuSeg预处理后已经是 (C, H, W) 格式
            pass
        elif image.ndim == 2:
            # 2D 灰度图 (H, W) -> (1, H, W)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3 and image.shape[-1] in [1, 3]:
            # (H, W, C) -> (C, H, W)
            image = np.moveaxis(image, -1, 0)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    c, h, w = image.shape
    
    # 步骤2: 转换为float32并归一化
    image = image.astype(np.float32)
    min_val = image.min()
    max_val = image.max()
    if max_val > min_val:
        image_normalized = (image - min_val) / (max_val - min_val)
    else:
        image_normalized = image - min_val
    
    # 步骤3: 单通道复制为三通道
    if image_normalized.shape[0] == 1:
        image_normalized = np.repeat(image_normalized, 3, axis=0)
    
    # ==================== 提取不重叠patches进行评估 ====================
    
    patch_size = img_size
    stride = img_size  # 不重叠，与训练时一致
    
    # 初始化预测图
    pred_map = np.zeros((h, w), dtype=np.uint8)
    
    metric_list = []
    patch_count = 0
    
    net.eval()
    
    # 遍历所有patch位置（不重叠）
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 确保patch在图像范围内
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            
            # 如果patch太小，跳过
            if y_end - y < patch_size // 2 or x_end - x < patch_size // 2:
                continue
            
            # 提取patch
            patch = image_normalized[:, y:y_end, x:x_end]
            raw_h = y_end - y
            raw_w = x_end - x
            
            # 如果patch尺寸不匹配，padding到patch_size
            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                padded_patch = np.zeros((c, patch_size, patch_size), dtype=np.float32)
                padded_patch[:, :patch.shape[1], :patch.shape[2]] = patch
                patch = padded_patch
            
            # 提取对应的label patch
            label_patch = label[y:y_end, x:x_end]
            if label_patch.shape[0] != patch_size or label_patch.shape[1] != patch_size:
                padded_label = np.zeros((patch_size, patch_size), dtype=label.dtype)
                padded_label[:label_patch.shape[0], :label_patch.shape[1]] = label_patch
                label_patch = padded_label
            
            # 转换为Tensor
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)  # [1, C, H, W]
            
            # 推理
            with torch.no_grad():
                output = net(patch_tensor)
                # 深监督模式下 output 是 list，取主输出 output[0]
                if isinstance(output, list):
                    output = output[0]
                output_prob = torch.sigmoid(output).squeeze().cpu().numpy()  # [H, W]
            
            # 二值化
            pred_patch = (output_prob > 0.5).astype(np.uint8)
            
            # 将预测结果放回完整图像
            pred_map[y:y_end, x:x_end] = pred_patch[:raw_h, :raw_w]
            
            # 仅对真实图像区域计算patch指标，避免zero-padding影响
            pred_binary = (pred_patch[:raw_h, :raw_w] > 0).astype(np.uint8)
            label_binary = (label_patch[:raw_h, :raw_w] > 0).astype(np.uint8)
            
            dice, hd95, asd, iou, recall, precision = calculate_metric_percase(pred_binary, label_binary)
            metric_list.append((dice, hd95, asd, iou, recall, precision))
            
            patch_count += 1
    
    if patch_count == 0:
        # 如果没有有效的patch，使用直接resize方法作为fallback
        print("Warning: No valid patches found, falling back to direct resize")
        return test_single_case_direct_resize(image, label, net, device, n_classes, img_size)
    
    return metric_list, pred_map


def test_single_case_direct_resize(image, label, net, device, n_classes=2, img_size=224):
    """Test a single case by direct resizing (旧方法，保留作为备选)
    
    Args:
        image: numpy array - 输入图像
        label: numpy array - 真实标签
        net: 模型
        device: 设备
        n_classes: 类别数（对于二分类为2）
        img_size: 输入图像尺寸
        
    Returns:
        metric_list: [(dice, hd95, asd, iou, recall, precision), ...]
        pred_map: 预测结果
    """
    # ==================== 数据预处理（与训练时严格一致）====================
    
    # 步骤1: 确保图像格式正确
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            # MoNuSeg预处理后已经是 (C, H, W) 格式
            pass
        elif image.ndim == 2:
            # 2D 灰度图 (H, W) -> (1, H, W)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3 and image.shape[-1] in [1, 3]:
            # (H, W, C) -> (C, H, W)
            image = np.moveaxis(image, -1, 0)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # 步骤2: 获取尺寸并Resize到目标尺寸
    c, h, w = image.shape
    output_size = (img_size, img_size)
    
    if h != output_size[0] or w != output_size[1]:
        if c in [1, 3]:
            # (C, H, W) -> 只resize H和W维度
            zoom_factors = (1, output_size[0] / h, output_size[1] / w)
            image = zoom(image, zoom_factors, order=3)  # 双三次插值
            label = zoom(label, zoom_factors[1:], order=0)  # 最近邻插值，只处理空间维度
    
    # 步骤3: 转换为float32
    image = image.astype(np.float32)
    
    # 步骤4: 单通道灰度图复制为三通道（与训练时一致）
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    
    # 步骤5: Min-Max归一化到[0,1]
    min_val = image.min()
    max_val = image.max()
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = image - min_val  # 避免除以零
    
    # 步骤6: 转换为Tensor
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # [1, C, H, W]
    label_tensor = torch.from_numpy(label.astype(np.float32)).long().unsqueeze(0)  # [1, H, W]
    
    # 步骤7: 维度检查
    assert image_tensor.shape[1] == 3, f'Expected 3 channels, got {image_tensor.shape[1]}'
    assert image_tensor.shape[2:] == (img_size, img_size), \
        f'Expected size {img_size}, got {image_tensor.shape[2:]}'
    
    # 移动到设备
    image_tensor = image_tensor.to(device=device)
    label_tensor = label_tensor.to(device=device)
    
    # ==================== 推理 ====================
    net.eval()
    with torch.no_grad():
        output = net(image_tensor)
        # 深监督模式下 output 是 list，取主输出 output[0]
        if isinstance(output, list):
            output = output[0]

    # ==================== 后处理 ====================
    # Sigmoid激活 + 阈值分割
    output_prob = torch.sigmoid(output)  # [1, 1, H, W]
    pred_map = (output_prob > 0.5).cpu().numpy().astype(np.uint8)  # [1, 1, H, W]
    
    # 去除batch和channel维度，得到2D mask
    pred_2d = pred_map.squeeze()  # [H, W]
    label_2d = label_tensor.cpu().numpy().squeeze().astype(np.uint8)  # [H, W]
    
    # ==================== 计算指标（符合2024年最新标准）====================
    # 对于二分类任务（前景=1，背景=0），只计算前景类的指标
    metric_list = []
    
    # 计算前景类（class 1）的指标
    pred_binary = (pred_2d > 0).astype(np.uint8)
    label_binary = (label_2d > 0).astype(np.uint8)
    
    dice, hd95, asd, iou, recall, precision = calculate_metric_percase(pred_binary, label_binary)
    metric_list.append((dice, hd95, asd, iou, recall, precision))
    
    return metric_list, pred_2d


def get_args():
    parser = argparse.ArgumentParser(description='Test TextGuidedBreastUSSegmentation on BreastUS_BUSI dataset')
    parser.add_argument('--model', '-m', type=str, 
                        default='model/epoch_299.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--data-dir', '-d', type=str, default='data/BreastUS_BUSI/test_vol_h5/',
                        help='Path to test data directory')
    parser.add_argument('--list-file', '-l', type=str, default='data/BreastUS_BUSI/lists/lists_BreastUS/test.txt',
                        help='Path to test file list')
    parser.add_argument('--dataset', type=str, default='BreastUS', choices=['BreastUS', 'MoNuSeg'],
                        help='Dataset type')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Dataset split to evaluate (train or test)')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size / patch size')
    parser.add_argument('--output', '-o', type=str, default='test_results/',
                        help='Output directory for results')
    parser.add_argument('--test-mode', type=str, default='sliding_window', 
                        choices=['sliding_window', 'direct_resize'],
                        help='Testing strategy: sliding_window (recommended for high-res images) or direct_resize')
    parser.add_argument('--use-cim', type=int, default=1,
                        help='Whether to use Contrast Improvement Module (1/0)')
    parser.add_argument('--cim-scaling-factor', type=float, default=0.5,
                        help='Initial scaling factor for CIM')
    parser.add_argument('--cim-kernel-size', type=int, default=3,
                        help='Kernel size for CIM smoothing conv')
    parser.add_argument('--use-cell-specific-aug', type=int, default=0,
                        help='Whether to use cell-specific augmentations (1/0)')
    parser.add_argument('--deep-supervision', type=int, default=1,
                        help='Enable deep supervision (must match training config) 1/0')
    parser.add_argument('--freeze-mamba-encoder', type=int, default=0,
                        help='Freeze VMamba encoder (must match training config) 1/0')
    parser.add_argument('--pretrained-ckpt-path', type=str,
                        default='./data/pretrained/vmamba/vmamba_tiny_e292.pth',
                        help='Path to VMamba pretrained checkpoint')
    parser.add_argument('--foreground-prior', type=float, default=0.0,
                        help='Foreground pixel prior for bias initialization')

    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s',
                       handlers=[
                           logging.FileHandler('test.log', encoding='utf-8'),
                           logging.StreamHandler(sys.stdout)
                       ])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # 根据 split 参数自动设置数据路径
    if args.split == 'train':
        if args.dataset == 'MoNuSeg':
            args.data_dir = 'MoNuSeg/MoNuSeg 2018 Training Data/Tissue Images'
            args.list_file = 'MoNuSeg/MoNuSeg 2018 Training Data/Annotations'  # 实际上是annotation目录
        else:
            args.data_dir = 'data/BreastUS_BUSI/train_npz'
            args.list_file = 'data/BreastUS_BUSI/lists/lists_BreastUS/train.txt'
        args.output = 'train_results/'
        logging.info('Evaluating on TRAINING set')
    else:
        if args.dataset == 'MoNuSeg':
            args.data_dir = 'MoNuSeg/MoNuSegTestData'
            args.list_file = 'MoNuSeg/MoNuSegTestData'  # 测试集的annotation目录
        else:
            args.data_dir = 'data/BreastUS_BUSI/test_vol_h5'
            args.list_file = 'data/BreastUS_BUSI/lists/lists_BreastUS/test.txt'
        args.output = 'test_results/'
        logging.info('Evaluating on TEST set')
    
    # Convert relative paths to absolute paths
    if not os.path.isabs(args.model):
        args.model = os.path.abspath(os.path.join(os.getcwd(), args.model))
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(os.getcwd(), args.data_dir))
    if not os.path.isabs(args.list_file):
        args.list_file = os.path.abspath(os.path.join(os.getcwd(), args.list_file))
    if not os.path.isabs(args.output):
        args.output = os.path.abspath(os.path.join(os.getcwd(), args.output))
    
    # Validate paths
    if not os.path.exists(args.model):
        logging.error(f'Model file not found: {args.model}')
        sys.exit(1)
    
    if not os.path.exists(args.data_dir):
        logging.error(f'Data directory not found: {args.data_dir}')
        sys.exit(1)
    
    if not os.path.exists(args.list_file):
        logging.error(f'List file not found: {args.list_file}')
        sys.exit(1)
    
    # Create model - using TextGuidedBreastUSSegmentation
    # 重要：必须与训练时使用相同的初始化参数以确保架构一致
    logging.info('Initializing model architecture...')
    model = TextGuidedBreastUSSegmentation(
        img_size=args.img_size,
        in_channels=args.channels,
        load_pretrained=True,
        pretrained_ckpt_path=args.pretrained_ckpt_path,
        freeze_mamba_encoder=bool(args.freeze_mamba_encoder),
        use_cim=bool(args.use_cim),
        cim_scaling_factor=args.cim_scaling_factor,
        cim_kernel_size=args.cim_kernel_size,
        foreground_prior=args.foreground_prior,
        deep_supervision=bool(args.deep_supervision)
    )
    model.to(device=device)
    
    # Load checkpoint
    logging.info(f'Loading trained weights from: {args.model}')
    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    
    # 处理权重加载（支持直接加载 state_dict 或包含其他键的字典）
    if isinstance(state_dict, dict):
        # 如果state_dict中包含'state_dict'键，提取它
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # 如果包含'model'键，提取它
        elif 'model' in state_dict:
            state_dict = state_dict['model']
    
    # Remove module prefix if exists (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # 使用strict=False加载，因为可能存在细微差异（如BatchNorm的num_batches_tracked）
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    # 详细报告加载情况
    if missing_keys:
        logging.warning(f'Missing keys ({len(missing_keys)}):')
        # 只显示前10个缺失的键
        for key in missing_keys[:10]:
            logging.warning(f'  - {key}')
        if len(missing_keys) > 10:
            logging.warning(f'  ... and {len(missing_keys) - 10} more')
    
    if unexpected_keys:
        logging.warning(f'Unexpected keys ({len(unexpected_keys)}):')
        # 只显示前10个意外的键
        for key in unexpected_keys[:10]:
            logging.warning(f'  - {key}')
        if len(unexpected_keys) > 10:
            logging.warning(f'  ... and {len(unexpected_keys) - 10} more')
    
    # 验证权重是否成功加载
    loaded_count = len([k for k in new_state_dict.keys() if k not in unexpected_keys])
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Successfully loaded {loaded_count}/{len(new_state_dict)} weight tensors')
    logging.info(f'Total model parameters: {total_params:,}')
    
    # 检查关键层的权重统计信息，确认不是随机初始化
    sample_weights = []
    for name, param in model.named_parameters():
        if 'conv' in name and param.dim() == 4:
            sample_weights.append(param.data.std().item())
            if len(sample_weights) >= 5:
                break
    
    if sample_weights:
        avg_std = np.mean(sample_weights)
        logging.info(f'Sample conv layer weight std: {avg_std:.6f}')
        if avg_std < 0.02:
            logging.error('Weight std is too low (<0.02), model may NOT be properly loaded!')
        elif avg_std < 0.05:
            logging.info('Weight std is moderate (0.02-0.05): typical for a well-converged trained model')
        else:
            logging.info('Weight std is high (>0.05): typical for randomly initialized or early-stage model')
    
    logging.info('Model loaded successfully!')
    
    # Load test file list
    if args.dataset == 'MoNuSeg':
        # 对于MoNuSeg，直接扫描目录中的文件
        import glob
        
        if args.split == 'train':
            # 训练集：从Tissue Images目录加载.tif文件
            image_dir = os.path.join(args.data_dir, 'Tissue Images') if os.path.isdir(os.path.join(args.data_dir, 'Tissue Images')) else args.data_dir
            image_files = glob.glob(os.path.join(image_dir, '*.tif'))
            
            # 过滤出有对应标注的文件
            annotation_dir = os.path.join(os.path.dirname(args.data_dir), 'Annotations') if 'Training Data' in args.data_dir else args.data_dir
            test_cases = []
            for img_file in image_files:
                base_name = os.path.basename(img_file).replace('.tif', '')
                xml_file = os.path.join(annotation_dir, base_name + '.xml')
                if os.path.exists(xml_file):
                    test_cases.append(base_name)
        else:
            # 测试集：从测试目录加载.xml文件
            annotation_dir = args.data_dir
            xml_files = glob.glob(os.path.join(annotation_dir, '*.xml'))
            
            # 过滤出有对应图像的文件
            test_cases = []
            for xml_file in xml_files:
                base_name = os.path.basename(xml_file).replace('.xml', '')
                # 尝试在多个位置查找对应的.tif文件
                tif_paths = [
                    os.path.join(annotation_dir, base_name + '.tif'),
                    os.path.join(os.path.dirname(annotation_dir), 'Tissue Images', base_name + '.tif'),
                ]
                
                for tif_path in tif_paths:
                    if os.path.exists(tif_path):
                        test_cases.append(base_name)
                        break
    else:
        with open(args.list_file, 'r') as f:
            test_cases = [line.strip() for line in f.readlines()]
    
    logging.info(f'Found {len(test_cases)} test cases')
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metric collectors (6元组指标)
    all_metrics = {
        'dice': [],
        'hd95': [],
        'asd': [],
        'iou': [],
        'recall': [],
        'precision': []
    }
    case_results = []
    
    # Test
    model.eval()
    
    logging.info('='*80)
    logging.info('Starting testing...')
    logging.info('='*80)
    
    for case_name in tqdm(test_cases, desc='Testing'):
        # Load data - support both .npz (train) and .tif/.xml (MoNuSeg) formats
        if args.dataset == 'MoNuSeg':
            if args.split == 'train':
                file_path = os.path.join(args.data_dir, f'{case_name}.tif')
            else:
                file_path = os.path.join(args.data_dir, f'{case_name}.xml')
        elif args.split == 'train':
            file_path = Path(args.data_dir) / f'{case_name}.npz'
        else:
            file_path = Path(args.data_dir) / f'{case_name}.npy.h5'
        
        if not os.path.exists(file_path):
            logging.warning(f'Data file not found: {file_path}')
            continue
        
        image, label = load_data(str(file_path), split=args.split, dataset=args.dataset)
        
        if image is None or label is None:
            logging.warning(f'Failed to load data for {case_name}')
            continue
        
        # Test single case - 根据test_mode选择策略
        if args.test_mode == 'sliding_window':
            metric_list, pred_map = test_single_case(image, label, model, device, args.classes, args.img_size)
        else:  # direct_resize
            metric_list, pred_map = test_single_case_direct_resize(image, label, model, device, args.classes, args.img_size)
        
        # Collect metrics for each class (6元组)
        for class_idx, (dice, hd95, asd, iou, recall, precision) in enumerate(metric_list, start=1):
            all_metrics['dice'].append(dice)
            all_metrics['hd95'].append(hd95)
            all_metrics['asd'].append(asd)
            all_metrics['iou'].append(iou)
            all_metrics['recall'].append(recall)
            all_metrics['precision'].append(precision)
            
            case_results.append({
                'case': case_name,
                'class': class_idx,
                'dice': dice,
                'hd95': hd95,
                'asd': asd,
                'iou': iou,
                'recall': recall,
                'precision': precision
            })
        
        # Save prediction
        pred_output_path = output_dir / f'{case_name}_pred.npy'
        np.save(str(pred_output_path), pred_map)
        
        # Log per-case metrics
        if metric_list:
            mean_dice = np.mean([m[0] for m in metric_list])
            mean_hd95 = np.mean([m[1] for m in metric_list])
            mean_asd = np.mean([m[2] for m in metric_list])
            mean_iou = np.mean([m[3] for m in metric_list])
            mean_recall = np.mean([m[4] for m in metric_list])
            mean_precision = np.mean([m[5] for m in metric_list])
            
            logging.info(f'{case_name}: Dice={mean_dice:.6f}, HD95={mean_hd95:.6f}, '
                        f'ASD={mean_asd:.6f}, IoU={mean_iou:.6f}, '
                        f'Recall={mean_recall:.6f}, Precision={mean_precision:.6f}')
    
    # Save detailed results
    results_file = output_dir / 'test_results.txt'
    with open(results_file, 'w') as f:
        f.write('Case\tClass\tDice\tHD95\tASD\tIoU\tRecall\tPrecision\n')
        f.write('='*100 + '\n')
        for result in case_results:
            f.write(f'{result["case"]}\t{result["class"]}\t'
                   f'{result["dice"]:.6f}\t{result["hd95"]:.6f}\t'
                   f'{result["asd"]:.6f}\t{result["iou"]:.6f}\t'
                   f'{result["recall"]:.6f}\t{result["precision"]:.6f}\n')
    
    # Summary statistics (符合2024年最新标准)
    logging.info('\n' + '='*80)
    logging.info('TEST RESULTS SUMMARY (2024 Standard Metrics)')
    logging.info('='*80)
    
    if all_metrics['dice']:
        # Per-class statistics
        for class_idx in range(1, args.classes):
            class_indices = [i for i, r in enumerate(case_results) if r['class'] == class_idx]
            if class_indices:
                logging.info(f'\nClass {class_idx} Metrics:')
                class_dices = [all_metrics['dice'][i] for i in class_indices]
                class_hd95s = [all_metrics['hd95'][i] for i in class_indices]
                class_asds = [all_metrics['asd'][i] for i in class_indices]
                class_ious = [all_metrics['iou'][i] for i in class_indices]
                class_recalls = [all_metrics['recall'][i] for i in class_indices]
                class_precisions = [all_metrics['precision'][i] for i in class_indices]
                
                logging.info(f'  Dice:     {np.mean(class_dices):.6f} ± {np.std(class_dices, ddof=1):.6f}')
                logging.info(f'  HD95:     {np.mean(class_hd95s):.6f} ± {np.std(class_hd95s, ddof=1):.6f}')
                logging.info(f'  ASD:      {np.mean(class_asds):.6f} ± {np.std(class_asds, ddof=1):.6f}')
                logging.info(f'  IoU:      {np.mean(class_ious):.6f} ± {np.std(class_ious, ddof=1):.6f}')
                logging.info(f'  Recall:   {np.mean(class_recalls):.6f} ± {np.std(class_recalls, ddof=1):.6f}')
                logging.info(f'  Precision:{np.mean(class_precisions):.6f} ± {np.std(class_precisions, ddof=1):.6f}')
        
        # Overall statistics
        logging.info('\n' + '='*80)
        logging.info('OVERALL METRICS (All Classes) - 2024 Standard:')
        logging.info('='*80)
        logging.info(f'  Test Cases: {len(test_cases)}')
        logging.info(f'  Dice:      {np.mean(all_metrics["dice"]):.6f} ± {np.std(all_metrics["dice"], ddof=1):.6f}')
        logging.info(f'  HD95:      {np.mean(all_metrics["hd95"]):.6f} ± {np.std(all_metrics["hd95"], ddof=1):.6f}')
        logging.info(f'  ASD:       {np.mean(all_metrics["asd"]):.6f} ± {np.std(all_metrics["asd"], ddof=1):.6f}')
        logging.info(f'  IoU:       {np.mean(all_metrics["iou"]):.6f} ± {np.std(all_metrics["iou"], ddof=1):.6f}')
        logging.info(f'  Recall:    {np.mean(all_metrics["recall"]):.6f} ± {np.std(all_metrics["recall"], ddof=1):.6f}')
        logging.info(f'  Precision: {np.mean(all_metrics["precision"]):.6f} ± {np.std(all_metrics["precision"], ddof=1):.6f}')
        logging.info('='*80)
        
        # Save summary
        summary_file = output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write('='*80 + '\n')
            f.write('TEST RESULTS SUMMARY (2024 Standard Metrics)\n')
            f.write('='*80 + '\n\n')
            
            for class_idx in range(1, args.classes):
                class_indices = [i for i, r in enumerate(case_results) if r['class'] == class_idx]
                if class_indices:
                    f.write(f'Class {class_idx} Metrics:\n')
                    class_dices = [all_metrics['dice'][i] for i in class_indices]
                    class_hd95s = [all_metrics['hd95'][i] for i in class_indices]
                    class_asds = [all_metrics['asd'][i] for i in class_indices]
                    class_ious = [all_metrics['iou'][i] for i in class_indices]
                    class_recalls = [all_metrics['recall'][i] for i in class_indices]
                    class_precisions = [all_metrics['precision'][i] for i in class_indices]
                    
                    f.write(f'  Dice:     {np.mean(class_dices):.6f} ± {np.std(class_dices, ddof=1):.6f}\n')
                    f.write(f'  HD95:     {np.mean(class_hd95s):.6f} ± {np.std(class_hd95s, ddof=1):.6f}\n')
                    f.write(f'  ASD:      {np.mean(class_asds):.6f} ± {np.std(class_asds, ddof=1):.6f}\n')
                    f.write(f'  IoU:      {np.mean(class_ious):.6f} ± {np.std(class_ious, ddof=1):.6f}\n')
                    f.write(f'  Recall:   {np.mean(class_recalls):.6f} ± {np.std(class_recalls, ddof=1):.6f}\n')
                    f.write(f'  Precision:{np.mean(class_precisions):.6f} ± {np.std(class_precisions, ddof=1):.6f}\n\n')
            
            f.write('='*80 + '\n')
            f.write('OVERALL METRICS (All Classes) - 2024 Standard:\n')
            f.write('='*80 + '\n')
            f.write(f'Test Cases: {len(test_cases)}\n')
            f.write(f'Dice:      {np.mean(all_metrics["dice"]):.6f} ± {np.std(all_metrics["dice"], ddof=1):.6f}\n')
            f.write(f'HD95:      {np.mean(all_metrics["hd95"]):.6f} ± {np.std(all_metrics["hd95"], ddof=1):.6f}\n')
            f.write(f'ASD:       {np.mean(all_metrics["asd"]):.6f} ± {np.std(all_metrics["asd"], ddof=1):.6f}\n')
            f.write(f'IoU:       {np.mean(all_metrics["iou"]):.6f} ± {np.std(all_metrics["iou"], ddof=1):.6f}\n')
            f.write(f'Recall:    {np.mean(all_metrics["recall"]):.6f} ± {np.std(all_metrics["recall"], ddof=1):.6f}\n')
            f.write(f'Precision: {np.mean(all_metrics["precision"]):.6f} ± {np.std(all_metrics["precision"], ddof=1):.6f}\n')
            f.write('='*80 + '\n')
    else:
        logging.error('No test cases were processed successfully!')
        sys.exit(1)


if __name__ == '__main__':
    main()
