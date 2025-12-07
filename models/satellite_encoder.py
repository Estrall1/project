# models/satellite_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SatelliteEncoder(nn.Module):
    """卫星视角流 """
    def __init__(self, feature_dim=512):
        super().__init__()
        
        # 多尺度编码器
        # 输入: [B, 3, 512, 512]
        
        # 尺度1: 1/2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 尺度2: 1/4
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 尺度3: 1/8
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 尺度4: 1/16
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(512, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.feature_dim = feature_dim
        
    def forward(self, x):
        """
        输入: [B, 3, 512, 512]
        输出: dense feature map [B, D, H/16, W/16]
        """
        # 多尺度特征提取
        x1 = self.conv1(x)  # [B, 64, 128, 128]
        x2 = self.conv2(x1)  # [B, 128, 64, 64]
        x3 = self.conv3(x2)  # [B, 256, 32, 32]
        x4 = self.conv4(x3)  # [B, 512, 16, 16]
        
        # 多尺度融合（这里简化为使用最深层的特征）
        fused = self.fusion(x4)  # [B, D, 16, 16]
        
        return fused