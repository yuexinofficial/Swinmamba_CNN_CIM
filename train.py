import argparse
import logging
import math
import os
import random
import numpy as np
import torch

import torch.backends.cudnn as cudnn

# 解决 HuggingFace tokenizers fork 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from trainer import trainer_synapse
from models.model import TextGuidedBreastUSSegmentation


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BreastUS_BUSI/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='BreastUS', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='../data/BreastUS_BUSI/lists/lists_BreastUS', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network (1 for binary segmentation)')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=130, help='maximum epoch number to train')  # 论文设置：130 epochs
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')  # 论文设置：batch_size=16
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0003,  # 论文设置：初始学习率 0.0003
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')  # 论文设置：224×224
parser.add_argument('--seed', type=int,
                    default=42, help='random seed')  # 论文设置：seed=42
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--load_pretrained', type=bool, default=False,
                    help='whether to load pretrained weights')
parser.add_argument('--use_cim', type=int, default=1,
                    help='whether to use Contrast Improvement Module (1/0)')
parser.add_argument('--use_cell_specific_aug', type=int, default=0,
                    help='whether to use cell-specific augmentations (elastic deformation, stain, cell scale) 1/0')
parser.add_argument('--cim_scaling_factor', type=float, default=0.5,
                    help='initial scaling factor for CIM')
parser.add_argument('--cim_kernel_size', type=int, default=3,
                    help='kernel size for CIM smoothing conv')
parser.add_argument('--preprocessed_dir', type=str, default='preprocessed_data',
                    help='directory containing preprocessed MoNuSeg patches')
parser.add_argument('--val_split', type=str, default='val',
                    help='Validation split to use (val or test). "val" for validation set, "test" for original test set')
parser.add_argument('--early_stopping_patience', type=int, default=0,
                    help='Early stopping patience (epochs without improvement before stopping). 0=disabled. '
                         'Recommended: 20-30 for stable training.')
parser.add_argument('--loss_type', type=str, default='focal_tversky',
                    choices=['dice_focal', 'focal_tversky'],
                    help='Loss function type: dice_focal (Dice+Focal), '
                         'focal_tversky (Focal Tversky Loss)')
parser.add_argument('--tversky_alpha', type=float, default=0.6,
                    help='Tversky alpha (FP penalty weight). alpha > beta penalizes FP more. '
                         'Default 0.6 for nuclei segmentation.')
parser.add_argument('--tversky_beta', type=float, default=0.4,
                    help='Tversky beta (FN penalty weight). Default 0.4.')
parser.add_argument('--tversky_gamma', type=float, default=2.5,
                    help='Tversky gamma (focal exponent). Higher = focus on hard samples. '
                         'Default 2.5.')
parser.add_argument('--foreground_prior', type=float, default=0.0,
                    help='Foreground pixel ratio for bias-prior initialization. '
                         '0=disabled. MoNuSeg: ~0.10, BreastUS: ~0.25. '
                         'bias = log(prior/(1-prior)).')
parser.add_argument('--pretrained_ckpt', type=str, default='./data/pretrained/vmamba/vmamba_tiny_e292.pth',
                    help='Path to Swin-UMamba/VMamba pretrained checkpoint')
parser.add_argument('--deep_supervision', type=int, default=0,
                    help='Enable deep supervision (multi-scale auxiliary loss) 1/0')
parser.add_argument('--freeze_mamba_encoder', type=int, default=0,
                    help='Freeze VMamba encoder after loading pretrained weights (1/0). Default 0 = trainable.')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': 'data/Synapse/train_npz',
            'list_dir': 'data/Synapse/lists/lists_Synapse',
            'num_classes': 9,
        },
        'BreastUS': {
            'root_path': 'data/BreastUS_BUSI/train_npz',
            'list_dir': 'data/BreastUS_BUSI/lists/lists_BreastUS',
            'num_classes': 1,  # 二分类：1通道输出
        },
        'MoNuSeg': {
            'image_dir': 'MoNuSeg/MoNuSeg 2018 Training Data/Tissue Images',
            'annotation_dir': 'MoNuSeg/MoNuSeg 2018 Training Data/Annotations',
            'test_image_dir': 'MoNuSeg/MoNuSegTestData',
            'test_annotation_dir': 'MoNuSeg/MoNuSegTestData',
            'num_classes': 1,  # 二分类：细胞核分割
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    if dataset_name == 'MoNuSeg':
        args.root_path = dataset_config[dataset_name]['image_dir']
        args.list_dir = dataset_config[dataset_name]['annotation_dir']
    else:
        args.root_path = dataset_config[dataset_name]['root_path']
        args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    # 修复相对路径问题（用户传入的是相对路径时自动转绝对）
    if not os.path.isabs(args.root_path):
        args.root_path = os.path.abspath(os.path.join(os.getcwd(), args.root_path))
    if not os.path.isabs(args.list_dir):
        args.list_dir = os.path.abspath(os.path.join(os.getcwd(), args.list_dir))
    if not os.path.isabs(args.preprocessed_dir):
        args.preprocessed_dir = os.path.abspath(os.path.join(os.getcwd(), args.preprocessed_dir))

    if not os.path.exists(args.root_path):
        raise FileNotFoundError(f'root_path not found: {args.root_path}')
    if not os.path.exists(args.list_dir):
        raise FileNotFoundError(f'list_dir not found: {args.list_dir}')
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    # 使用项目内 output 目录（而非 ../model 指向项目外）
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_path = "./output/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_VMambaTiny'
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + '_' + timestamp  # 时间戳防止覆盖

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    print(f"Output directory: {snapshot_path}")

    # 自动计算前景像素占比 π（用于 Bias-prior Initialization）
    if args.foreground_prior <= 0.0:
        print("\nAuto-computing foreground pixel ratio (π) from training set...")
        try:
            if dataset_name == 'MoNuSeg':
                labels_dir = os.path.join(args.preprocessed_dir, 'train', 'labels')
                if os.path.exists(labels_dir):
                    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.npy')]
                    if label_files:
                        ratios = []
                        for f in label_files[:200]:  # 采样最多200个patch
                            label = np.load(os.path.join(labels_dir, f))
                            ratios.append(label.mean())
                        foreground_prior = float(np.mean(ratios))
                        print(f"  Sampled {len(label_files[:200])} patches, avg foreground ratio = {foreground_prior:.6f}")
                    else:
                        print("  WARNING: No .npy label files found, using default π=0.10")
                        foreground_prior = 0.10
                else:
                    print(f"  WARNING: Labels dir not found at {labels_dir}, using default π=0.10")
                    foreground_prior = 0.10
            elif dataset_name == 'BreastUS':
                train_list = os.path.join(args.list_dir, 'train.txt')
                if os.path.exists(train_list):
                    sample_names = open(train_list).readlines()
                    ratios = []
                    for name in sample_names[:200]:
                        npz_path = os.path.join(args.root_path, name.strip() + '.npz')
                        if os.path.exists(npz_path):
                            label = np.load(npz_path)['label']
                            ratios.append(label.mean())
                    if ratios:
                        foreground_prior = float(np.mean(ratios))
                        print(f"  Sampled {len(ratios)} npz files, avg foreground ratio = {foreground_prior:.6f}")
                    else:
                        print("  WARNING: No valid npz files, using default π=0.25")
                        foreground_prior = 0.25
                else:
                    print(f"  WARNING: train.txt not found, using default π=0.25")
                    foreground_prior = 0.25
            else:
                foreground_prior = 0.10  # 默认值
                print(f"  Unknown dataset, using default π={foreground_prior}")
        except Exception as e:
            print(f"  WARNING: Auto-computation failed ({e}), using default π=0.10")
            foreground_prior = 0.10
    else:
        foreground_prior = args.foreground_prior
        print(f"\nUsing manually specified foreground prior π={foreground_prior:.6f}")
    print(f"  -> Bias-prior init b0 = log(π/(1-π)) = {math.log(foreground_prior/(1-foreground_prior)):.4f}")

    # 创建分割模型
    use_pretrained = bool(args.load_pretrained)
    net = TextGuidedBreastUSSegmentation(
        img_size=args.img_size,
        in_channels=3,
        load_pretrained=use_pretrained,
        pretrained_ckpt_path=args.pretrained_ckpt,
        freeze_mamba_encoder=bool(args.freeze_mamba_encoder),
        use_cim=bool(args.use_cim),
        cim_scaling_factor=args.cim_scaling_factor,
        cim_kernel_size=args.cim_kernel_size,
        foreground_prior=foreground_prior,
        deep_supervision=bool(args.deep_supervision)
    ).cuda()

    # 打印模型信息
    print(f"\n{'='*60}")
    print(f"Created Dual-Branch Segmentation Model")
    print(f"  CNN Branch:  ResNet34 (ImageNet pretrained)")
    print(f"  Mamba Branch: VMamba-Tiny Encoder (SS2D)")
    print(f"  Input:  {args.img_size}x{args.img_size}x3")
    print(f"  Output: {args.img_size}x{args.img_size}x1 Binary Mask")
    print(f"  Pretrained: {use_pretrained}")
    print(f"  CIM: {bool(args.use_cim)}")

    print(f"\nComponent Loading Status Check:")
    # ResNet34
    if hasattr(net.cnn_encoder, 'resnet') and net.cnn_encoder.resnet is not None:
        if hasattr(net.cnn_encoder, 'pretrained_loaded') and net.cnn_encoder.pretrained_loaded:
            print(f"  [OK] ResNet34: Loaded (ImageNet pretrained)")
        else:
            print(f"  [WARN] ResNet34: Using random initialization")
    else:
        print(f"  [FAIL] ResNet34: Not loaded correctly")
    # VMamba-Tiny
    if net.mamba_pretrained_loaded:
        print(f"  [OK] VMamba-Tiny Encoder: Loaded (ImageNet pretrained)")
    else:
        print(f"  [WARN] VMamba-Tiny Encoder: Using random initialization")

    total_params = sum(p.numel() for p in net.parameters())
    print(f"\nTotal Parameters: {total_params/1e6:.1f}M")
    print(f"{'='*60}\n")

    trainer = {'Synapse': trainer_synapse, 'BreastUS': trainer_synapse, 'MoNuSeg': trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path)