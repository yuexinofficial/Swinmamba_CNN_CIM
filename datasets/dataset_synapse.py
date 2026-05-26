import os
import random
import h5py
import json
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import skimage.draw


def random_rot_flip(image, label):
    """随机旋转 90 度倍数和翻转"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(-2, -1))  # 在最后两个维度上旋转
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis + 1 if image.ndim == 3 else axis).copy()  # 翻转H或W维度
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """随机旋转 -15° 到 +15°"""
    angle = np.random.uniform(-15, 15)  # 论文设置：±15°
    if image.ndim == 3:
        # 对于3D图像，需要分别旋转每个通道
        rotated_image = np.zeros_like(image)
        for c in range(image.shape[0]):
            rotated_image[c] = ndimage.rotate(image[c], angle, order=3, reshape=False)
        image = rotated_image
    else:
        image = ndimage.rotate(image, angle, order=3, reshape=False)  # 双三次插值
    label = ndimage.rotate(label, angle, order=0, reshape=False)  # 最近邻插值
    return image, label


def parse_monuseg_xml(xml_path, img_shape):
    """
    解析MoNuSeg数据集的XML标注文件，生成二进制mask
    
    Args:
        xml_path: XML文件路径
        img_shape: 图像形状 (H, W)
        
    Returns:
        binary_mask: 二进制mask，细胞核区域为1，背景为0
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 创建空的mask
        binary_mask = np.zeros(img_shape, dtype=np.uint8)
        
        # 遍历所有Region
        for region in root.findall('.//Region'):
            vertices = []
            
            # 提取所有顶点坐标
            for vertex in region.findall('.//Vertex'):
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                vertices.append([x, y])
            
            if len(vertices) >= 3:  # 至少需要3个点才能形成多边形
                vertices = np.array(vertices)
                
                # 使用skimage绘制多边形
                rr, cc = skimage.draw.polygon(vertices[:, 1], vertices[:, 0], shape=img_shape)
                binary_mask[rr, cc] = 1
        
        return binary_mask
        
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return np.zeros(img_shape, dtype=np.uint8)


def random_brightness_contrast(image, probability=0.25):
    if random.random() < probability:
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-0.1, 0.1)
        image = image.astype(np.float32) * alpha + beta
        image = np.clip(image, 0, np.max(image) if np.max(image) > 0 else 1)
    return image


def random_color_scale(image, probability=0.25):
    if random.random() < probability and image.ndim == 3 and image.shape[0] == 3:
        scales = np.random.uniform(0.8, 1.2, size=3)
        for c in range(3):
            image[c] = image[c] * scales[c]
        image = np.clip(image, 0, np.max(image) if np.max(image) > 0 else 1)
    return image


def random_blur(image, probability=0.25):
    if random.random() < probability and image.ndim == 3:
        sigma = random.uniform(0.1, 1.5)
        blurred = np.zeros_like(image)
        for c in range(image.shape[0]):
            blurred[c] = ndimage.gaussian_filter(image[c], sigma=sigma)
        image = blurred
    return image


def random_coarse_dropout(image, probability=0.25, max_holes=5, max_size=32):
    if random.random() < probability and image.ndim == 3:
        _, h, w = image.shape
        for _ in range(random.randint(1, max_holes)):
            hole_h = random.randint(8, min(max_size, h // 2))
            hole_w = random.randint(8, min(max_size, w // 2))
            y1 = random.randint(0, h - hole_h)
            x1 = random.randint(0, w - hole_w)
            image[:, y1:y1 + hole_h, x1:x1 + hole_w] = 0
    return image


def elastic_deformation(image, label, probability=0.3, alpha=1000, sigma=50):
    """
    弹性变形增强 - 模拟组织切片的变形
    基于细胞分割论文中的标准实现

    Args:
        image: 输入图像 [C, H, W] 或 [H, W]
        label: 标签图像 [H, W]
        probability: 应用概率
        alpha: 变形强度 (越大变形越剧烈)
        sigma: 变形平滑度 (越大变形越平滑)
    """
    if random.random() > probability:
        return image, label

    if image.ndim == 3:
        c, h, w = image.shape
    else:
        h, w = image.shape
        c = 1

    # 生成随机位移场
    dx = np.random.randn(h, w) * alpha
    dy = np.random.randn(h, w) * alpha

    # 使用高斯滤波平滑位移场
    dx = ndimage.gaussian_filter(dx, sigma=sigma, mode='nearest')
    dy = ndimage.gaussian_filter(dy, sigma=sigma, mode='nearest')

    # 创建坐标网格
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # 计算变形后的坐标
    deformed_x = x_coords + dx
    deformed_y = y_coords + dy

    # 应用弹性变形
    if image.ndim == 3:
        deformed_image = np.zeros_like(image)
        for i in range(c):
            deformed_image[i] = ndimage.map_coordinates(
                image[i], [deformed_y, deformed_x], order=3, mode='nearest'
            )
    else:
        deformed_image = ndimage.map_coordinates(
            image, [deformed_y, deformed_x], order=3, mode='nearest'
        )

    # 对标签也应用相同的变形 (使用最近邻插值保持标签的离散性)
    deformed_label = ndimage.map_coordinates(
        label, [deformed_y, deformed_x], order=0, mode='nearest'
    )

    return deformed_image, deformed_label


def stain_augmentation(image, probability=0.25):
    """
    染色增强 - 模拟不同实验室的H&E染色差异
    基于细胞分割论文中的标准实现

    Args:
        image: 输入图像 [C, H, W] RGB格式
        probability: 应用概率
    """
    if random.random() > probability or image.ndim != 3 or image.shape[0] != 3:
        return image

    # 将图像转换到HSV空间进行染色调整
    image_rgb = np.transpose(image, (1, 2, 0))  # [H, W, C]
    image_hsv = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    # 随机调整色调和饱和度 (模拟染色差异)
    hue_shift = random.uniform(-10, 10)  # 色调偏移
    sat_scale = random.uniform(0.8, 1.2)  # 饱和度缩放

    image_hsv[:, :, 0] = (image_hsv[:, :, 0] + hue_shift) % 180  # HSV色调范围0-179
    image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * sat_scale, 0, 255)

    # 转换回RGB
    augmented_rgb = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    augmented_image = np.transpose(augmented_rgb, (2, 0, 1)).astype(np.float32) / 255.0

    return augmented_image


def cell_scale_augmentation(image, label, probability=0.2):
    """
    细胞尺度增强 - 针对不同大小的细胞核进行尺度变换
    基于细胞分割论文中的实现

    Args:
        image: 输入图像 [C, H, W]
        label: 标签图像 [H, W]
        probability: 应用概率
    """
    if random.random() > probability:
        return image, label

    # 计算细胞密度 (标签中前景像素的比例)
    cell_density = np.sum(label > 0) / label.size

    # 根据细胞密度决定尺度变换范围
    if cell_density > 0.1:  # 高密度区域
        scale_range = (0.9, 1.1)  # 小范围变换
    elif cell_density > 0.05:  # 中密度区域
        scale_range = (0.8, 1.2)  # 中等范围变换
    else:  # 低密度区域
        scale_range = (0.7, 1.3)  # 大范围变换

    scale_factor = random.uniform(*scale_range)

    if image.ndim == 3:
        c, h, w = image.shape
    else:
        h, w = image.shape
        c = 1

    # 计算缩放后的尺寸
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # 缩放图像
    if image.ndim == 3:
        scaled_image = np.zeros((c, new_h, new_w), dtype=image.dtype)
        for i in range(c):
            scaled_image[i] = cv2.resize(image[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 缩放标签 (使用最近邻插值)
    scaled_label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 如果缩放后尺寸不匹配，进行padding或crop
    if new_h != h or new_w != w:
        # 创建目标尺寸的图像
        if image.ndim == 3:
            final_image = np.zeros((c, h, w), dtype=image.dtype)
            final_label = np.zeros((h, w), dtype=label.dtype)
        else:
            final_image = np.zeros((h, w), dtype=image.dtype)
            final_label = np.zeros((h, w), dtype=label.dtype)

        # 计算复制区域
        copy_h = min(h, new_h)
        copy_w = min(w, new_w)

        if image.ndim == 3:
            final_image[:, :copy_h, :copy_w] = scaled_image[:, :copy_h, :copy_w]
        else:
            final_image[:copy_h, :copy_w] = scaled_image[:copy_h, :copy_w]

        final_label[:copy_h, :copy_w] = scaled_label[:copy_h, :copy_w]

        return final_image, final_label

    return scaled_image, scaled_label


class RandomGenerator(object):
    """数据增强类，参考另一个项目的丰富增强"""
    def __init__(self, output_size, dataset_type='Synapse', use_cell_specific_aug=True):
        self.output_size = output_size
        self.dataset_type = dataset_type
        self.use_cell_specific_aug = use_cell_specific_aug

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 随机 90° 旋转 + 翻转
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)

        # 随机旋转 ±15°
        if random.random() < 0.6:
            image, label = random_rotate(image, label)

        # 亮度/对比度扰动
        image = random_brightness_contrast(image, probability=0.25)

        # 颜色通道抖动
        image = random_color_scale(image, probability=0.25)

        # 模糊增强
        image = random_blur(image, probability=0.25)

        # 随机遮挡（CoarseDropout）
        image = random_coarse_dropout(image, probability=0.25)

        # === 新增：细胞特定增强方法 ===
        if self.use_cell_specific_aug:
            # 弹性变形增强 (对细胞分割最重要)
            image, label = elastic_deformation(image, label, probability=0.3, alpha=1000, sigma=50)

            # 染色增强 (模拟H&E染色差异)
            image = stain_augmentation(image, probability=0.25)

            # 细胞尺度增强 (根据细胞密度调整)
            image, label = cell_scale_augmentation(image, label, probability=0.2)

        # 获取图像尺寸
        if self.dataset_type == 'MoNuSeg':
            x, y = image.shape[1], image.shape[2]
        else:
            if image.ndim == 3:
                x, y = image.shape[1], image.shape[2]
            else:
                x, y = image.shape

        if x != self.output_size[0] or y != self.output_size[1]:
            if self.dataset_type == 'MoNuSeg':
                image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)
            else:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = image.astype(np.float32)

        if self.dataset_type == 'MoNuSeg':
            if image.shape[0] != 3:
                if image.shape[0] == 1:
                    image = np.repeat(image, 3, axis=0)
        else:
            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)

        image = image - image.min()
        max_val = image.max()
        if max_val > 0:
            image = image / max_val

        image = torch.from_numpy(image)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label}
        return sample


class ValGenerator(object):
    """验证集/测试集预处理类（无随机增强，但包含必要的预处理）
    
    根据项目规范：测试阶段的数据预处理必须与训练阶段保持一致
    包括：Resize、通道复制、Min-Max 归一化
    """
    def __init__(self, output_size, dataset_type='Synapse'):
        self.output_size = output_size
        self.dataset_type = dataset_type

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # 处理图像维度
        if self.dataset_type == 'MoNuSeg':
            # MoNuSeg: 图像应该是 (C, H, W) 格式，保持3D
            if image.ndim == 3:
                # 确保是 (C, H, W) 而不是 (H, W, C)
                if image.shape[0] not in [1, 3] and image.shape[2] in [1, 3]:
                    image = np.transpose(image, (2, 0, 1))
            elif image.ndim == 2:
                # 如果是2D灰度图，扩展为3通道
                image = np.expand_dims(image, axis=0)
                image = np.repeat(image, 3, axis=0)
        else:
            # 其他数据集：确保图像是 2D (H, W)
            if image.ndim == 3:
                # 如果是 (1, H, W)，压缩为 (H, W)
                if image.shape[0] == 1:
                    image = image[0]
                # 如果是 (H, W, 1)，压缩为 (H, W)
                elif image.shape[2] == 1:
                    image = image[:, :, 0]
        
        # 确保 label 是 2D (H, W)
        if label.ndim == 3:
            if label.shape[0] == 1:
                label = label[0]  # (1, H, W) -> (H, W)
            elif label.shape[2] == 1:
                label = label[:, :, 0]  # (H, W, 1) -> (H, W)
        
        # Resize 到目标尺寸
        if self.dataset_type == 'MoNuSeg':
            # MoNuSeg: 图像是 (C, H, W)，获取 H, W
            _, h, w = image.shape
        else:
            # 其他数据集：图像是 (H, W)
            h, w = image.shape
            
        if h != self.output_size[0] or w != self.output_size[1]:
            if self.dataset_type == 'MoNuSeg':
                # MoNuSeg: 对 (C, H, W) 进行resize
                image = zoom(image, (1, self.output_size[0] / h, self.output_size[1] / w), order=3)
            else:
                # 其他数据集：对 (H, W) 进行resize
                image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=3)
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)

        # 转换为 float32
        image = image.astype(np.float32)
        
        # 对于非MoNuSeg数据集，确保是3通道
        if self.dataset_type != 'MoNuSeg':
            # 兼容单通道灰度图，复制为三通道，适配默认三通道模型
            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)

        # Min-Max 归一化到 [0,1]（与训练集保持一致）
        image = image - image.min()
        max_val = image.max()
        if max_val > 0:
            image = image / max_val

        # 转换为 Tensor
        image = torch.from_numpy(image)
        label = torch.from_numpy(label.astype(np.int64))  # label 使用 int64，不需要 float32
        sample = {'image': image, 'label': label}  # label 保持 2D (H, W)
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        
        # 加载文本标签（如果存在）
        self.text_labels = None
        text_label_path = os.path.join('MoNuSeg', 'text_labels', f'{split}_text.json')
        if os.path.exists(text_label_path):
            try:
                with open(text_label_path, 'r', encoding='utf-8') as f:
                    self.text_labels = json.load(f)
                print(f"✅ Loaded {len(self.text_labels)} text labels from: {text_label_path}")
            except Exception as e:
                print(f"❌ Failed to load text labels: {e}")
        else:
            print(f"⚠️ Text labels not found at {text_label_path}, using default text")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx].strip('\n')
        
        if self.split == "train":
            data_path = os.path.join(self.data_dir, case_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = case_name
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = case_name
        
        # 添加文本标签
        if self.text_labels is not None and case_name in self.text_labels:
            sample['text'] = self.text_labels[case_name]
        else:
            # 使用默认文本提示
            sample['text'] = "breast lesion segmentation"
            
        return sample


class MoNuSeg_dataset(Dataset):
    """MoNuSeg数据集类，处理H&E染色图像和XML标注，支持patch提取"""
    
    def __init__(self, image_dir, annotation_dir, split, transform=None, patch_size=224, stride=112):
        """
        Args:
            image_dir: 图像文件夹路径
            annotation_dir: 标注文件夹路径  
            split: 数据集分割 ('train', 'val', 'test')
            transform: 数据变换
            patch_size: 图像块大小
            stride: 滑动窗口步长
        """
        self.transform = transform
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        
        # 获取所有图像文件
        self.image_files = []
        self.annotation_files = []
        
        # 检查是否有分割列表文件
        split_file = os.path.join(annotation_dir, f'{split}_files.txt')
        file_list = None
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                file_list = set(line.strip() for line in f.readlines() if line.strip())
            print(f"Loaded file list from {split_file} ({len(file_list)} files)")
        
        # 扫描图像和标注文件
        for file in sorted(os.listdir(image_dir)):
            if file.endswith('.tif'):
                base_name = file[:-4]  # 移除 .tif 扩展名
                
                # 如果有分割列表，只加载列表中的文件
                if file_list is not None and base_name not in file_list:
                    continue
                
                image_path = os.path.join(image_dir, file)
                annotation_path = os.path.join(annotation_dir, base_name + '.xml')
                
                if os.path.exists(annotation_path):
                    self.image_files.append(image_path)
                    self.annotation_files.append(annotation_path)
                else:
                    print(f"Warning: Annotation file not found for {base_name}")
        
        print(f"Found {len(self.image_files)} image-annotation pairs for {split}")
        
        # 计算每张图像的patches坐标
        self.patches_info = []  # 存储 (image_idx, x, y) 的列表
        
        for image_idx, image_path in enumerate(self.image_files):
            try:
                # 读取图像获取尺寸
                img = Image.open(image_path)
                img_array = np.array(img)
                h, w = img_array.shape[:2]
                
                # 计算patch网格
                for y in range(0, h - patch_size + 1, stride):
                    for x in range(0, w - patch_size + 1, stride):
                        self.patches_info.append((image_idx, x, y))
                        
            except Exception as e:
                print(f"Error reading image {image_path} for patch calculation: {e}")
                continue
        
        print(f"Total patches generated: {len(self.patches_info)} (avg {len(self.patches_info)/len(self.image_files):.1f} patches per image)")
        
        # 加载文本标签（如果存在）
        self.text_labels = None
        text_label_path = os.path.join('MoNuSeg', 'text_labels', f'{split}_text.json')
        if os.path.exists(text_label_path):
            try:
                with open(text_label_path, 'r', encoding='utf-8') as f:
                    self.text_labels = json.load(f)
                print(f"✅ Loaded {len(self.text_labels)} text labels from: {text_label_path}")
            except Exception as e:
                print(f"❌ Failed to load text labels: {e}")
        else:
            print(f"⚠️ Text labels not found at {text_label_path}, using default text")

    def __len__(self):
        return len(self.patches_info)

    def __getitem__(self, idx):
        # 获取patch信息
        image_idx, x, y = self.patches_info[idx]
        
        image_path = self.image_files[image_idx]
        annotation_path = self.annotation_files[image_idx]
        base_name = os.path.basename(image_path)[:-4]  # 移除 .tif 扩展名
        
        # 读取图像
        try:
            image = Image.open(image_path)
            image = np.array(image)
            
            # 如果图像是RGBA，去除alpha通道
            if image.shape[-1] == 4:
                image = image[:, :, :3]
            
            # 转换为 (C, H, W) 格式
            if image.shape[-1] == 3:
                image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回空白图像
            image = np.zeros((3, self.patch_size, self.patch_size), dtype=np.uint8)
        
        # 解析XML标注
        try:
            img_shape = (image.shape[1], image.shape[2])  # (H, W)
            label = parse_monuseg_xml(annotation_path, img_shape)
        except Exception as e:
            print(f"Error parsing annotation {annotation_path}: {e}")
            # 返回空白mask
            label = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
        
        # 提取patch
        patch_x_end = x + self.patch_size
        patch_y_end = y + self.patch_size
        
        # 确保patch不超出图像边界
        if patch_x_end > image.shape[2]:
            patch_x_end = image.shape[2]
        if patch_y_end > image.shape[1]:
            patch_y_end = image.shape[1]
            
        image_patch = image[:, y:patch_y_end, x:patch_x_end]
        label_patch = label[y:patch_y_end, x:patch_x_end]
        
        # 如果patch尺寸不匹配（边界情况），进行padding
        if image_patch.shape[1] != self.patch_size or image_patch.shape[2] != self.patch_size:
            # 创建目标尺寸的patch
            padded_image = np.zeros((3, self.patch_size, self.patch_size), dtype=image_patch.dtype)
            padded_label = np.zeros((self.patch_size, self.patch_size), dtype=label_patch.dtype)
            
            # 复制现有数据
            h_actual, w_actual = image_patch.shape[1], image_patch.shape[2]
            padded_image[:, :h_actual, :w_actual] = image_patch
            padded_label[:h_actual, :w_actual] = label_patch
            
            image_patch = padded_image
            label_patch = padded_label
        
        sample = {'image': image_patch, 'label': label_patch}
        
        if self.transform:
            sample = self.transform(sample)
            
        sample['case_name'] = f"{base_name}_patch_{idx}"
        
        # 添加文本标签
        if self.text_labels is not None and base_name in self.text_labels:
            sample['text'] = self.text_labels[base_name]
        else:
            # 使用默认文本提示
            sample['text'] = "nuclear segmentation in histopathology images"
            
        return sample
