# datasets/SiamUAV.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json


class SiamUAVDataset(Dataset):
    """示例SiamUAV数据集 - 替换为你的实际数据集"""
    def __init__(self, root_dir="data/SiamUAV", split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 假设数据组织方式
        self.uav_dir = os.path.join(root_dir, split, "uav")
        self.sat_dir = os.path.join(root_dir, split, "satellite")
        self.annot_file = os.path.join(root_dir, split, "annotations.json")
        
        # 加载标注
        if os.path.exists(self.annot_file):
            with open(self.annot_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            # 生成虚拟数据用于测试
            self.annotations = []
            uav_files = [f for f in os.listdir(self.uav_dir) if f.endswith(('.jpg', '.png'))]
            for uav_file in uav_files[:10]:  # 只取前10个用于测试
                self.annotations.append({
                    'uav': uav_file,
                    'satellite': uav_file.replace('uav', 'sat'),
                    'x': np.random.uniform(0, 1),
                    'y': np.random.uniform(0, 1)
                })
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annot = self.annotations[idx]
        
        # 加载图像
        uav_path = os.path.join(self.uav_dir, annot['uav'])
        sat_path = os.path.join(self.sat_dir, annot['satellite'])
        
        # 如果文件不存在，创建虚拟图像
        if not os.path.exists(uav_path):
            uav_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            uav_img = cv2.imread(uav_path)
            
        if not os.path.exists(sat_path):
            sat_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            sat_img = cv2.imread(sat_path)
        
        # 转换为tensor
        uav_tensor = torch.from_numpy(uav_img.transpose(2, 0, 1)).float() / 255.0
        sat_tensor = torch.from_numpy(sat_img.transpose(2, 0, 1)).float() / 255.0
        
        # 获取坐标
        X = torch.tensor([annot['x']])
        Y = torch.tensor([annot['y']])
        
        return uav_tensor, sat_tensor, X, Y, uav_path, sat_path, idx, annot


class SiamUAV_test(SiamUAVDataset):
    """测试集"""
    def __init__(self, opt=None):
        if opt is None:
            opt = type('obj', (object,), {'data_root': 'data/SiamUAV'})
        super().__init__(
            root_dir=getattr(opt, 'data_root', 'data/SiamUAV'),
            split="test"
        )