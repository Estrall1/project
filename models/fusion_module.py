# models/fusion_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossViewCrossAttention(nn.Module):
    """创新点2: 跨视角注意力融合 CVCA"""
    def __init__(self, uav_dim=512, sat_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = sat_dim // num_heads
        
        # 投影层
        self.uav_q_proj = nn.Linear(uav_dim, sat_dim)
        self.sat_k_proj = nn.Linear(sat_dim, sat_dim)
        self.sat_v_proj = nn.Linear(sat_dim, sat_dim)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=sat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(sat_dim)
        self.norm2 = nn.LayerNorm(sat_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(sat_dim, sat_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sat_dim * 4, sat_dim)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(sat_dim, sat_dim)
        
    def forward(self, uav_tokens, sat_feature_map):
        """
        uav_tokens: [B, T, D] UAV tokens
        sat_feature_map: [B, D, H, W] 卫星特征图
        输出: 融合特征 [B, D]
        """
        B, D, H, W = sat_feature_map.shape
        
        # 重塑卫星特征图为序列
        sat_tokens = sat_feature_map.view(B, D, H*W).permute(0, 2, 1)  # [B, HW, D]
        
        # 投影
        uav_query = self.uav_q_proj(uav_tokens)  # [B, T, D]
        sat_key = self.sat_k_proj(sat_tokens)    # [B, HW, D]
        sat_value = self.sat_v_proj(sat_tokens)  # [B, HW, D]
        
        # 自注意力残差连接
        attended, attention_weights = self.attention(
            uav_query, sat_key, sat_value
        )
        attended = self.norm1(attended + uav_query)
        
        # 前馈网络残差连接
        ffn_out = self.ffn(attended)
        attended = self.norm2(attended + ffn_out)
        
        # 聚合所有tokens
        aggregated = attended.mean(dim=1)  # [B, D]
        
        # 最终投影
        output = self.output_proj(aggregated)
        
        return output, attention_weights