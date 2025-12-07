# models/taskflow.py
import torch
import torch.nn as nn

def make_model(config=None):
    """
    创建模型的主函数
    根据配置返回相应的模型
    """
    if config is None:
        config = {}
    
    # 这里根据你的实际需求实现模型创建逻辑
    # 例如：
    from .uav_encoder import LocalSaliencyEncoder
    
    class TaskFlowModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = LocalSaliencyEncoder()
            # 添加其他需要的组件
            
        def forward(self, x):
            features = self.encoder(x)
            # 添加其他处理逻辑
            return features
    
    return TaskFlowModel()