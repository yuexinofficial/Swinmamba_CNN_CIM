import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk


# ==============================================================================
#  Loss Functions
# ==============================================================================

class DiceLoss(nn.Module):
    """
    二分类 Dice Loss
    标准实现，按batch展平计算
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        inputs_flat = inputs.view(inputs.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (inputs_flat * target_flat).sum(dim=1)
        union = inputs_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice_coeff = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1. - dice_coeff
        return dice_loss.mean()


class FocalLoss(nn.Module):
    """
    二分类 Focal Loss
    基于 binary_cross_entropy_with_logits 实现
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        p_t = torch.sigmoid(inputs)
        pt = p_t * targets + (1 - p_t) * (1 - targets)
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        loss = focal_weight * bce_loss
        return loss.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (论文推荐参数: α=0.6, β=0.4, γ=2.5)

    Tversky 系数: TI = (TP) / (TP + α·FP + β·FN)
    α > β → 对假阳性惩罚更重（宁可漏检，不要误检）
    α < β → 对假阴性惩罚更重（宁可误检，不要漏检）
    Focal 项: (1 - TI)^γ 让模型关注困难样本

    参考: Abraham & Khan, "A Novel Focal Tversky Loss for Improved
          Unbalanced Biomedical Image Segmentation", ISBI 2019
    """
    def __init__(self, alpha: float = 0.6, beta: float = 0.4, gamma: float = 2.5,
                 smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(preds)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        batch_size = preds.size(0)
        preds_flat = preds.reshape(batch_size, -1)
        targets_flat = targets.reshape(batch_size, -1)
        tp = (preds_flat * targets_flat).sum(dim=1)
        fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
        fn = ((1 - preds_flat) * targets_flat).sum(dim=1)
        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = tp / denominator
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky.mean()


class SegmentationLoss(nn.Module):
    """
    统一分割损失函数，支持多种损失类型:

        loss_type='dice_focal':
            weight_dice * DiceLoss + weight_focal * FocalLoss

        loss_type='focal_tversky':
            FocalTverskyLoss(α, β, γ)
    """
    def __init__(self, loss_type: str = 'dice_focal',
                 dice_weight: float = 0.95, focal_weight: float = 0.05,
                 tversky_alpha: float = 0.6, tversky_beta: float = 0.4,
                 tversky_gamma: float = 2.5):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == 'dice_focal':
            self.dice_weight = dice_weight
            self.focal_weight = focal_weight
            self.dice_loss = DiceLoss()
            self.focal_loss = FocalLoss(alpha=0.8, gamma=2.0)
        elif loss_type == 'focal_tversky':
            self.tversky_loss = FocalTverskyLoss(
                alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. "
                             f"Choose 'dice_focal' or 'focal_tversky'.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'dice_focal':
            dice_loss = self.dice_loss(pred, target)
            if target.dim() == 3:
                target = target.unsqueeze(1)
            focal_loss = self.focal_loss(pred, target.float())
            return self.dice_weight * dice_loss + self.focal_weight * focal_loss
        elif self.loss_type == 'focal_tversky':
            return self.tversky_loss(pred, target)
def calculate_metric_percase(pred, gt):
    """计算分割指标（Dice, HD95, ASD, IoU, Recall, Precision）- 符合最新医学图像分割标准
    
    Args:
        pred: 预测的二值mask (numpy array)
        gt: 真实的二值mask (numpy array)
        
    Returns:
        tuple: (dice, hd95, asd, iou, recall, precision)
        
    Metrics:
        - Dice: Sørensen-Dice系数，范围[0,1]，越高越好
        - HD95: 95% Hausdorff距离，越低越好（对异常值鲁棒）
        - ASD: Average Surface Distance，平均表面距离，越低越好
        - IoU: Intersection over Union (Jaccard指数)，范围[0,1]
        - Recall: 敏感性/召回率，TP/(TP+FN)
        - Precision: 精确度，TP/(TP+FP)
    """
    import logging
    
    # 确保输入是二值的
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    
    # 检查维度
    if pred.shape != gt.shape:
        logging.warning(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
        # 尝试resize到相同尺寸
        if pred.size > 0 and gt.size > 0:
            zoom_factors = tuple(gt_dim / pred_dim for gt_dim, pred_dim in zip(gt.shape, pred.shape))
            pred = zoom(pred, zoom_factors, order=0)
    
    # 处理边界情况
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    
    if pred_sum == 0 and gt_sum == 0:
        # 真阴性：预测和真实都为空
        return 1.0, 0.0, 0.0, 1.0, 1.0, 1.0
    elif pred_sum == 0 or gt_sum == 0:
        # 假阳性或假阴性
        return 0.0, float('inf'), float('inf'), 0.0, 0.0, 0.0
    
    try:
        # 计算核心指标
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        
        # 计算ASD（Average Surface Distance）
        try:
            asd = metric.binary.asd(pred, gt)
        except Exception:
            # 如果ASD计算失败，使用HD95作为替代
            asd = hd95
        
        recall = metric.binary.recall(pred, gt)
        precision = metric.binary.precision(pred, gt)
        
        # 手动计算 Jaccard (IoU)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = intersection / union if union > 0 else 0
        
        return dice, hd95, asd, iou, recall, precision
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        # 返回默认值
        return 0.0, float('inf'), float('inf'), 0.0, 0.0, 0.0


def calculate_metrics(pred, gt):
    """兼容性函数，调用calculate_metric_percase"""
    return calculate_metric_percase(pred, gt)


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            
            # 处理单通道灰度图，转换为三通道以适配模型
            input = torch.from_numpy(slice).unsqueeze(0).float().cuda()  # (1, H, W)
            if input.shape[0] == 1:
                input = input.repeat(3, 1, 1)  # (3, H, W) 复制为三通道
            input = input.unsqueeze(0)  # (1, 3, H, W) 添加 batch 维度
            
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    
    metric_list = []
    detail_metrics = {}  # 存储详细指标
    
    for i in range(1, classes):
        metrics_tuple = calculate_metric_percase(prediction == i, label == i)
        metric_list.append(metrics_tuple)
        
        # 保存详细指标用于打印
        detail_metrics[f'class_{i}'] = {
            'dice': metrics_tuple[0],
            'hd95': metrics_tuple[1],
            'iou': metrics_tuple[2],
            'recall': metrics_tuple[3],
            'precision': metrics_tuple[4]
        }
    
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    
    return metric_list