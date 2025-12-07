# models/crossview_model_advanced.py
import torch
import torch.nn as nn
from models.uav_encoder import UAVEncoder
from models.satellite_encoder import SatelliteEncoder
from models.fusion_module import CrossViewCrossAttention
from models.matcher import MatchingHead

class AdvancedCrossViewGeolocator(nn.Module):
    """完整的跨视角地理定位模型"""
    def __init__(self, feature_dim=512, sat_grid_size=16):
        super().__init__()
        
        # A. UAV视角流
        self.uav_encoder = UAVEncoder(backbone='resnet18', feature_dim=feature_dim)
        
        # B. 卫星视角流
        self.sat_encoder = SatelliteEncoder(feature_dim=feature_dim)
        
        # C. 创新点: 跨视角注意力融合
        self.cross_fusion = CrossViewCrossAttention(
            uav_dim=feature_dim,
            sat_dim=feature_dim,
            num_heads=8
        )
        
        # D. 匹配头
        self.matching_head = MatchingHead(
            feature_dim=feature_dim,
            sat_grid_size=sat_grid_size
        )
        
        print(f"✅ 初始化高级模型:")
        print(f"  UAV编码器: ResNet18 → 可变形卷积 → Token投影")
        print(f"  卫星编码器: 多尺度CNN → 特征融合")
        print(f"  融合模块: 跨视角交叉注意力 (CVCA)")
        print(f"  匹配头: 粗匹配(16x16热力图) + 精匹配")
        
    def forward(self, uav_img, sat_img):
        """
        输入:
            uav_img: [B, 3, 256, 256]
            sat_img: [B, 3, 512, 512]
        输出: 包含预测结果的字典
        """
        # 1. 编码
        uav_tokens, uav_features = self.uav_encoder(uav_img)  # [B, T, D]
        sat_features = self.sat_encoder(sat_img)  # [B, D, H, W]
        
        # 2. 融合
        fused_features, attention_weights = self.cross_fusion(uav_tokens, sat_features)
        
        # 3. 匹配和定位
        outputs = self.matching_head(fused_features)
        outputs['attention_weights'] = attention_weights
        
        return outputs