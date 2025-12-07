# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .geo_utils import haversine_distance

class MultiTaskLoss(nn.Module):
    """多任务损失函数（步骤3要求）"""
    
    def __init__(self, lambda_match=1.0, lambda_reg=0.1, lambda_haversine=0.01):
        super().__init__()
        self.lambda_match = lambda_match
        self.lambda_reg = lambda_reg
        self.lambda_haversine = lambda_haversine
        
        # 匹配损失（对比损失）
        self.contrastive_loss = nn.CosineEmbeddingLoss()
        
        # 回归损失
        self.reg_loss = nn.SmoothL1Loss()
    
    def forward(self, outputs, labels):
        """
        Args:
            outputs: 模型输出字典
            labels: 包含以下键的字典
                - latlon: 真实经纬度 [B, 2]
                - grid_labels: 网格标签 [B, 2]
        """
        # 从模型输出中提取
        pred_latlon = outputs['fine_coords']
        coarse_heatmap = outputs['coarse_heatmap']
        fused_tokens = outputs['fused_tokens']
        
        # 从标签中提取
        latlon_labels = labels['latlon']
        grid_labels = labels['grid_labels']
        
        # 1. 匹配损失（使用模型内置的匹配头损失）
        match_losses = outputs.get('match_losses', {
            'total': torch.tensor(0.0),
            'coarse': torch.tensor(0.0),
            'fine': torch.tensor(0.0)
        })
        
        # 2. 回归损失
        reg_loss = self.reg_loss(pred_latlon, latlon_labels)
        
        # 3. 哈弗辛距离损失
        distances = haversine_distance(
            pred_latlon[:, 0], pred_latlon[:, 1],
            latlon_labels[:, 0], latlon_labels[:, 1]
        )
        haversine_loss = distances.mean()
        
        # 4. 总损失
        total_loss = (
            self.lambda_match * match_losses['total'] +
            self.lambda_reg * reg_loss +
            self.lambda_haversine * haversine_loss
        )
        
        return {
            'total': total_loss,
            'match': match_losses['total'],
            'regression': reg_loss,
            'haversine': haversine_loss,
            'distances': distances
        }