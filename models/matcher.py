# models/matcher.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingHead(nn.Module):
    """匹配头 - 粗匹配 + 精匹配"""
    def __init__(self, feature_dim=512, sat_grid_size=16):
        super().__init__()
        
        self.sat_grid_size = sat_grid_size
        
        # 1. 粗匹配模块
        self.coarse_locator = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, sat_grid_size * sat_grid_size)  # 输出16x16热力图
        )
        
        # 2. 位置编码
        self.position_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # 3. 精匹配模块
        self.fine_locator = nn.Sequential(
            nn.Linear(feature_dim + 128, 512),  # 特征 + 位置编码
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 输出经纬度
        )
        
    def forward(self, fused_features):
        """
        输入: [B, D] 融合特征
        输出: 粗热力图和精细坐标
        """
        B = fused_features.shape[0]
        
        # 1. 粗匹配: 生成热力图
        coarse_logits = self.coarse_locator(fused_features)
        coarse_map = coarse_logits.view(B, self.sat_grid_size, self.sat_grid_size)
        
        # 2. 获取粗略位置
        flat_coarse = coarse_map.view(B, -1)
        coarse_probs = F.softmax(flat_coarse, dim=1)
        
        # 使用加权平均得到连续位置
        grid_coords = torch.zeros(B, 2, device=fused_features.device)
        for i in range(self.sat_grid_size):
            for j in range(self.sat_grid_size):
                idx = i * self.sat_grid_size + j
                weight = coarse_probs[:, idx].unsqueeze(1)
                grid_coords[:, 0] += weight.squeeze() * j  # x坐标
                grid_coords[:, 1] += weight.squeeze() * i  # y坐标
        
        # 归一化到[0, 1]
        grid_coords[:, 0] = grid_coords[:, 0] / (self.sat_grid_size - 1)
        grid_coords[:, 1] = grid_coords[:, 1] / (self.sat_grid_size - 1)
        
        # 3. 位置编码
        pos_encoding = self.position_encoder(grid_coords)
        
        # 4. 精匹配: 结合位置编码
        fine_input = torch.cat([fused_features, pos_encoding], dim=1)
        fine_coords = self.fine_locator(fine_input)
        
        return {
            'coarse_map': coarse_map,
            'coarse_probs': coarse_probs,
            'grid_coords': grid_coords,
            'fine_coords': fine_coords,
            'pred_lat': fine_coords[:, 0:1],
            'pred_lon': fine_coords[:, 1:2]
        }