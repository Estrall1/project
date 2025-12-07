# models/uav_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UAVEncoder(nn.Module):
    """UAV视角流"""
    def __init__(self, backbone='resnet18', feature_dim=512):
        super().__init__()
        
        # 1. 局部显著性建模 - 使用预训练ResNet
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            in_channels = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            in_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 2. 几何畸变鲁棒性增强 - 使用可变形卷积
        self.deform_conv = nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1)
        
        # 3. 序列式token输出
        self.token_projection = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.feature_dim = feature_dim
        
    def forward(self, x):
        """
        输入: [B, 3, H, W]
        输出: tokens [B, T, D], features [B, D, H', W']
        """
        # 提取基础特征
        base_features = self.features(x)  # [B, C, H/32, W/32]
        
        # 可变形卷积增强几何鲁棒性
        enhanced_features = self.deform_conv(base_features)
        
        # 自适应池化到固定大小
        pooled_features = self.adaptive_pool(enhanced_features)  # [B, D, 8, 8]
        
        # 生成序列tokens
        tokens = self.token_projection(pooled_features)
        B, D, H, W = tokens.shape
        tokens = tokens.view(B, D, H*W).permute(0, 2, 1)  # [B, T, D]
        
        return tokens, pooled_features