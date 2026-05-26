import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SegmentationLoss
from torchvision import transforms


def deep_supervision_loss(outputs, target, criterion, ds_weights=None):
    """Compute loss with deep supervision.

    Args:
        outputs: single tensor or list [main, aux1, aux2, aux3]
        target: ground truth label [B, 1, H, W]
        criterion: loss function
        ds_weights: weights for [main, aux1, aux2, aux3] (default: 1.0, 0.5, 0.5, 0.5)

    Returns:
        scalar loss
    """
    if not isinstance(outputs, list):
        return criterion(outputs, target)

    if ds_weights is None:
        ds_weights = [1.0, 0.5, 0.5, 0.5]

    loss = ds_weights[0] * criterion(outputs[0], target)
    for i, aux_out in enumerate(outputs[1:], 1):
        if i < len(ds_weights):
            aux_up = F.interpolate(aux_out, size=target.shape[2:],
                                   mode='bilinear', align_corners=False)
            loss = loss + ds_weights[i] * criterion(aux_up, target)
    return loss


class EarlyStopping:
    """早停机制：当验证指标在连续 patience 个 epoch 内未改善时停止训练

    Args:
        patience: 容忍的 epoch 数（验证指标未提升的次数上限）
        min_delta: 最小改善阈值（超过此值才算有提升）
        mode: 'max' 表示指标越大越好（如 Dice），'min' 表示越小越好（如 Loss）
        restore_best: 是否在停止时恢复最佳模型权重
    """
    def __init__(self, patience: int = 20, min_delta: float = 0.0,
                 mode: str = 'max', restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None

    def step(self, score, model):
        """每轮验证后调用，返回是否应该停止"""
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            if self.restore_best:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        improved = (
            (self.mode == 'max' and score > self.best_score + self.min_delta) or
            (self.mode == 'min' and score < self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = score
            self.counter = 0
            if self.restore_best:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best and self.best_state_dict is not None:
                    model.load_state_dict(self.best_state_dict)
                return True

        return False


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """带 Warm-up 的 Poly 学习率调度器（完全匹配论文设置）
    
    Args:
        optimizer: 优化器
        warmup_epochs: Warm-up 轮数（论文设置：10）
        total_epochs: 总训练轮数
        iters_per_epoch: 每轮迭代次数
        power: Poly 指数（论文设置：0.9）
        base_lr: 基础学习率（论文设置：0.0003）
        min_lr: 最小学习率
    """
    def __init__(self, optimizer, warmup_epochs=10, total_epochs=130, iters_per_epoch=1, 
                 power=0.9, base_lr=0.0003, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = total_epochs * iters_per_epoch
        self.power = power
        self.base_lr = base_lr
        self.min_lr = min_lr
        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        current_iter = self.last_epoch
        
        # Warm-up 阶段：线性增加学习率
        if current_iter < self.warmup_iters:
            warmup_factor = current_iter / self.warmup_iters
            # 从 base_lr/1000 线性增加到 base_lr
            factor = max(1e-3, warmup_factor)
            return [self.base_lr * factor for _ in self.optimizer.param_groups]
        
        # Poly 阶段：多项式衰减
        else:
            progress = (current_iter - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            factor = (1 - progress) ** self.power
            return [max(self.base_lr * factor, self.min_lr) for _ in self.optimizer.param_groups]


def validate_model(model, valloader, criterion=None, device='cuda'):
    """在验证集上评估1通道二分类模型"""
    from utils import SegmentationLoss

    model.eval()
    if criterion is None:
        criterion = SegmentationLoss(loss_type='dice_focal', dice_weight=0.5, focal_weight=0.5)
    
    total_loss = 0
    total_dice = 0
    total_iou = 0
    num_batches = len(valloader)
    
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valloader):
            image, label = sampled_batch['image'], sampled_batch['label']
            image, label = image.to(device), label.to(device)
            
            # 前向传播（深监督时取主输出）
            output = model(image)
            if isinstance(output, list):
                output = output[0]  # take main output for validation metrics

            # 调整label形状以匹配1通道输出
            if label.dim() == 3:
                label = label.unsqueeze(1)  # [B, 1, H, W]
            
            # 计算损失
            loss = criterion(output, label)
            total_loss += loss.item()
            
            # 计算 Dice 和 IoU
            output_prob = torch.sigmoid(output)  # [B, 1, H, W]
            output_pred = (output_prob > 0.5).float()  # [B, 1, H, W]
            
            # Dice 系数
            intersect = (output_pred * label).sum()
            dice = 2. * intersect / ((output_pred.sum() + label.sum()) + 1e-6)
            total_dice += dice.item()
            
            # IoU
            union = ((output_pred > 0) | (label > 0)).sum()
            if union > 0:
                iou = intersect.float() / union
                total_iou += iou.item()
    
    model.train()
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    avg_iou = total_iou / max(num_batches, 1)
    
    return avg_loss, avg_dice, avg_iou


def trainer_synapse(args, model, snapshot_path):
    import sys
    import os
    
    # 添加项目根目录和 datasets 目录到 Python 路径
    root_path = os.getcwd()
    datasets_path = os.path.join(root_path, 'datasets')
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    if datasets_path not in sys.path:
        sys.path.insert(0, datasets_path)

    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator, MoNuSeg_dataset
    from datasets.preprocessed_monuseg import PreprocessedMoNuSegDataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes  # 现在是1（1通道二分类）
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    
    # 根据数据集类型加载不同的数据集
    if args.dataset == 'MoNuSeg':
        # 加载已经预处理好的MoNuSeg patch数据集
        db_train = PreprocessedMoNuSegDataset(
            preprocessed_dir=args.preprocessed_dir,
            split="train",
            transform=transforms.Compose([
                RandomGenerator(output_size=[args.img_size, args.img_size], dataset_type='MoNuSeg',
                                use_cell_specific_aug=bool(args.use_cell_specific_aug))
            ])
        )
        print("The length of MoNuSeg train set is: {}".format(len(db_train)))
        
        # 加载验证集
        from dataset_synapse import ValGenerator
        db_val = PreprocessedMoNuSegDataset(
            preprocessed_dir=args.preprocessed_dir,
            split="val",
            transform=transforms.Compose([
                ValGenerator(output_size=[args.img_size, args.img_size], dataset_type='MoNuSeg')
            ])
        )
        print("The length of MoNuSeg validation set is: {}".format(len(db_val)))
    else:
        # 加载训练集
        db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                   transform=transforms.Compose(
                                       [RandomGenerator(output_size=[args.img_size, args.img_size],
                                                        use_cell_specific_aug=bool(args.use_cell_specific_aug))]))
        print("The length of train set is: {}".format(len(db_train)))
        
        # 加载验证集（从测试集中分出的 50%）
        val_split = getattr(args, 'val_split', 'val')  # 默认使用 val.txt
        from dataset_synapse import ValGenerator
        db_val = Synapse_dataset(base_dir=args.root_path.replace('train_npz', 'test_vol_h5'), 
                                 list_dir=args.list_dir, 
                                 split=val_split,  # 使用 val.txt
                                 transform=transforms.Compose([ValGenerator(output_size=[args.img_size, args.img_size])]))
        print("The length of validation set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)  # 论文设置：DataLoader worker 初始化 seed=42

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    # 验证集 DataLoader（不 shuffle，不增强）
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    # 论文设置：AdamW, weight_decay=3e-5, betas=(0.9, 0.999)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=3e-5, betas=(0.9, 0.999))
    # 论文设置：Warm-up Poly 学习率调度，warmup_epochs=10, power=0.9
    iters_per_epoch = len(trainloader)

    # 根据命令行参数选择损失函数
    loss_type = getattr(args, 'loss_type', 'focal_tversky')
    if loss_type == 'focal_tversky':
        tversky_alpha = getattr(args, 'tversky_alpha', 0.6)
        tversky_beta = getattr(args, 'tversky_beta', 0.4)
        tversky_gamma = getattr(args, 'tversky_gamma', 2.5)
        criterion = SegmentationLoss(
            loss_type='focal_tversky',
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_gamma=tversky_gamma
        )
        logging.info(f"Using Focal Tversky Loss (alpha={tversky_alpha}, beta={tversky_beta}, gamma={tversky_gamma})")
    else:
        criterion = SegmentationLoss(loss_type='dice_focal', dice_weight=0.5, focal_weight=0.5)
        logging.info("Using Dice + Focal Loss (w_dice=0.5, w_focal=0.5)")

    scheduler = WarmupPolyLR(
        optimizer,
        warmup_epochs=10,           # 论文设置：10 个 epoch warm-up
        total_epochs=args.max_epochs,
        iters_per_epoch=iters_per_epoch,
        power=0.9,                  # 论文设置：power=0.9
        base_lr=base_lr,            # 0.0003
        min_lr=1e-6
    )
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    early_stopping_patience = getattr(args, 'early_stopping_patience', 0)
    if early_stopping_patience > 0:
        early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=1e-4,
                                      mode='max', restore_best=True)
        logging.info(f"EarlyStopping enabled: patience={early_stopping_patience}, mode=max (monitoring val_dice)")
    else:
        early_stopper = None
        logging.info("EarlyStopping disabled")
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # 确保 label_batch 是 4D [B, 1, H, W]
            if label_batch.dim() == 5:
                # 5D [B, 1, 1, H, W] -> 4D [B, 1, H, W]
                label_batch = label_batch.squeeze(2)
            elif label_batch.dim() == 3:
                # 3D [B, H, W] -> 4D [B, 1, H, W]
                label_batch = label_batch.unsqueeze(1)
            elif label_batch.dim() == 4 and label_batch.shape[1] != 1:
                # 4D [B, C, H, W] where C != 1 -> [B, 1, H, W]
                label_batch = label_batch[:, 0:1, :, :]
            
            outputs = model(image_batch)  # [B, 1, H, W] or list for deep supervision

            # 计算损失（支持深监督）
            loss = deep_supervision_loss(outputs, label_batch, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = optimizer.param_groups[0]['lr']

            # Poly 学习率调度器在每个 iteration 后更新
            scheduler.step()

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 20 == 0:
                # 使用安全的索引，避免 batch_size=1 时越界
                sample_idx = min(1, image_batch.size(0) - 1)
                image = image_batch[sample_idx, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                
                # 修改：将预测结果进行二值化处理
                outputs_vis_raw = outputs[0] if isinstance(outputs, list) else outputs
                outputs_vis = torch.sigmoid(outputs_vis_raw)
                # 添加二值化处理：output > 0.5
                binary_prediction = (outputs_vis > 0.5).float()
                # 将二值化结果缩放到0-50范围以便可视化
                binary_prediction_scaled = binary_prediction * 50
                
                # 修复：从batch中选取单个样本，去除batch维度
                binary_prediction_single = binary_prediction_scaled[sample_idx]
                writer.add_image('train/Prediction', binary_prediction_single, iter_num)
                
                labs = label_batch[sample_idx, ...] * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        
        # 每个 epoch 结束后在验证集上评估
        if (epoch_num + 1) % 1 == 0 or epoch_num == max_epoch - 1:
            val_loss, val_dice, val_iou = validate_model(model, valloader, criterion=criterion)
            logging.info(f'Epoch {epoch_num}: Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}, Val IoU={val_iou:.4f}')
            
            # TensorBoard 记录
            writer.add_scalar('val/loss', val_loss, epoch_num)
            writer.add_scalar('val/dice', val_dice, epoch_num)
            writer.add_scalar('val/iou', val_iou, epoch_num)
            
            # 保存最佳模型
            if val_dice > best_performance:
                best_performance = val_dice
                save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"save best model to {save_mode_path} with val_dice={val_dice:.4f}")

            # 早停检查
            if early_stopper is not None:
                stopped = early_stopper.step(val_dice, model)
                if stopped:
                    logging.info(
                        f"EarlyStopping triggered at epoch {epoch_num}! "
                        f"Best val_dice={early_stopper.best_score:.4f}, "
                        f"No improvement for {early_stopping_patience} epochs."
                    )
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    break

    writer.close()
    logging.info(f"Training Finished! Best Validation Dice: {best_performance:.4f}")
    return "Training Finished!"