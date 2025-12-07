# train_final_simple.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np

print("🚀 简化训练开始...")

# 1. 数据准备
with open('train_annotations.json', 'r') as f:
    train_data = json.load(f)[:100]  # 只用100个样本

# 计算归一化参数
lats = [item['lat'] for item in train_data]
lons = [item['lon'] for item in train_data]
lat_min, lat_max = min(lats), max(lats)
lon_min, lon_max = min(lons), max(lons)

def normalize_coord(coord, min_val, max_val):
    return (coord - min_val) / (max_val - min_val)

# 准备数据
norm_labels = []
for item in train_data:
    norm_lat = normalize_coord(item['lat'], lat_min, lat_max)
    norm_lon = normalize_coord(item['lon'], lon_min, lon_max)
    norm_labels.append([norm_lat, norm_lon])

norm_labels = torch.tensor(norm_labels, dtype=torch.float32)

# 2. 极简模型（确保能学习）
class FinalSimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 非常简单的模型
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()  # 确保输出在[0,1]范围！
        )
    
    def forward(self, uav_img, sat_img):
        uav_feat = self.conv(uav_img).view(uav_img.size(0), -1)
        sat_feat = self.conv(sat_img).view(sat_img.size(0), -1)
        combined = torch.cat([uav_feat, sat_feat], dim=1)
        return {'fine_coords': self.fc(combined)}

# 3. 训练
model = FinalSimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 生成虚拟图像数据
train_size = len(train_data)
uav_imgs = torch.randn(train_size, 3, 256, 256)
sat_imgs = torch.randn(train_size, 3, 512, 512)

print(f"训练数据: {train_size}个样本")
print(f"坐标范围: 纬度 [{lat_min:.3f}, {lat_max:.3f}]")
print(f"          经度 [{lon_min:.3f}, {lon_max:.3f}]")

# 训练循环
epochs = 20
for epoch in range(epochs):
    # 小批量训练
    indices = torch.randperm(train_size)
    epoch_loss = 0
    
    for i in range(0, train_size, 4):
        batch_indices = indices[i:i+4]
        
        batch_uav = uav_imgs[batch_indices]
        batch_sat = sat_imgs[batch_indices]
        batch_labels = norm_labels[batch_indices]
        
        outputs = model(batch_uav, batch_sat)
        loss = criterion(outputs['fine_coords'], batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / (train_size // 4)
    
    # 每5个epoch显示预测示例
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(uav_imgs[:3], sat_imgs[:3])
            print(f"\nEpoch {epoch+1}: 损失={avg_loss:.6f}")
            for j in range(3):
                pred = test_outputs['fine_coords'][j].numpy()
                true = norm_labels[j].numpy()
                print(f"  样本{j+1}: 预测({pred[0]:.3f},{pred[1]:.3f}) 真实({true[0]:.3f},{true[1]:.3f})")
        model.train()

# 4. 保存模型
os.makedirs('final_model', exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'lat_min': lat_min, 'lat_max': lat_max,
    'lon_min': lon_min, 'lon_max': lon_max,
    'train_size': train_size,
    'final_loss': avg_loss
}, 'final_model/simple_trained.pth')

print(f"\n✅ 训练完成! 模型保存到: final_model/simple_trained.pth")
print(f"   最终损失: {avg_loss:.6f}")

# 5. 测试
model.eval()
with torch.no_grad():
    test_outputs = model(uav_imgs[:5], sat_imgs[:5])
    
    print(f"\n🧪 最终测试:")
    for j in range(5):
        pred_norm = test_outputs['fine_coords'][j].numpy()
        
        # 反归一化
        pred_lat = pred_norm[0] * (lat_max - lat_min) + lat_min
        pred_lon = pred_norm[1] * (lon_max - lon_min) + lon_min
        
        true_lat = train_data[j]['lat']
        true_lon = train_data[j]['lon']
        
        print(f"  样本{j+1}:")
        print(f"    预测: ({pred_lat:.3f}, {pred_lon:.3f})")
        print(f"    真实: ({true_lat:.3f}, {true_lon:.3f})")
        print(f"    误差: {abs(pred_lat-true_lat):.3f}°, {abs(pred_lon-true_lon):.3f}°")