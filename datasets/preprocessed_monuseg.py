import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class PreprocessedMoNuSegDataset(Dataset):
    """使用预处理patches的MoNuSeg数据集类"""

    def __init__(self, preprocessed_dir, split, transform=None):
        """
        Args:
            preprocessed_dir: 预处理数据目录
            split: 'train' 或 'val'
            transform: 数据变换
        """
        self.transform = transform
        self.split = split

        self.images_dir = os.path.join(preprocessed_dir, split, 'images')
        self.labels_dir = os.path.join(preprocessed_dir, split, 'labels')

        # 加载patch信息
        patch_info_path = os.path.join(preprocessed_dir, split, f'{split}_patches.json')
        with open(patch_info_path, 'r') as f:
            self.patch_info = json.load(f)

        print(f"Loaded {len(self.patch_info)} preprocessed patches for {split}")

        # 移除文本标签加载，因为模型不使用文本输入

    def __len__(self):
        return len(self.patch_info)

    def __getitem__(self, idx):
        patch_data = self.patch_info[idx]

        # 加载预处理的patch
        patch_name = patch_data['patch_name']
        image_path = os.path.join(self.images_dir, f"{patch_name}.npy")
        label_path = os.path.join(self.labels_dir, f"{patch_name}.npy")

        image = np.load(image_path)
        label = np.load(label_path)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = patch_name

        return sample