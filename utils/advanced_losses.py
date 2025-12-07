# utils/advanced_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HaversineLoss(nn.Module):
    """Haversine距离损失 - 用于监控，不用于反向传播"""
    def __init__(self, R=6371000.0):
        super().__init__()
        self.R = R
    
    def forward(self, pred_coords, true_coords):
        """
        计算Haversine距离（米）
        输入:
            pred_coords: [B, 2] (纬度, 经度)
            true_coords: [B, 2] (纬度, 经度)
        输出: 平均距离
        """
        # 将坐标转换为弧度
        pred_lat = torch.deg2rad(pred_coords[:, 0])
        pred_lon = torch.deg2rad(pred_coords[:, 1])
        true_lat = torch.deg2rad(true_coords[:, 0])
        true_lon = torch.deg2rad(true_coords[:, 1])
        
        # Haversine公式
        dlat = true_lat - pred_lat
        dlon = true_lon - pred_lon
        
        a = torch.sin(dlat/2)**2 + torch.cos(pred_lat) * torch.cos(true_lat) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        distance = self.R * c
        
        return distance.mean()

class AdvancedGeolocationLoss(nn.Module):
    """完整的损失函数 - 符合面试要求"""
    
    def __init__(self, lambda_coarse=0.5, lambda_fine=1.0, temperature=0.07):
        super().__init__()
        self.lambda_coarse = lambda_coarse
        self.lambda_fine = lambda_fine
        self.temperature = temperature
        
        # 1. 粗匹配损失
        self.coarse_loss = nn.CrossEntropyLoss()
        
        # 2. 精细定位损失
        self.fine_loss = nn.SmoothL1Loss(beta=1.0)
        
        # 3. Haversine距离监控
        self.haversine = HaversineLoss()
        
        # 4. 对比损失（可选）
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
    
    def forward(self, outputs, labels, uav_features=None, sat_features=None):
        losses = {}
        
        B = outputs['fine_coords'].shape[0] if 'fine_coords' in outputs else 1
        
        # 1. 粗匹配损失（如果有标签）
        if 'coarse_map' in outputs and 'grid_labels' in labels:
            coarse_logits = outputs['coarse_map'].view(B, -1)
            coarse_loss = self.coarse_loss(coarse_logits, labels['grid_labels'])
            losses['coarse'] = coarse_loss * self.lambda_coarse
        
        # 2. 精细定位损失
        if 'fine_coords' in outputs and 'latlon' in labels:
            fine_loss = self.fine_loss(outputs['fine_coords'], labels['latlon'])
            losses['fine'] = fine_loss * self.lambda_fine
        
        # 3. Haversine距离（用于监控）
        if 'fine_coords' in outputs and 'latlon' in labels:
            with torch.no_grad():
                haversine_dist = self.haversine(outputs['fine_coords'], labels['latlon'])
                losses['haversine'] = haversine_dist
        
        # 4. 总损失
        total_loss = sum([v for k, v in losses.items() if k not in ['haversine']])
        losses['total'] = total_loss
        
        return losses


# 导出
__all__ = ['AdvancedGeolocationLoss', 'HaversineLoss']