# train_improved.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("🚀 改进版训练开始...")

# 1. 加载更多数据
with open('train_annotations.json', 'r') as f:
    all_data = json.load(f)
    
print(f"总数据量: {len(all_data)} 个样本")

# 使用更多数据（例如800个）
train_data = all_data[:800]
print(f"使用 {len(train_data)} 个样本进行训练")

# 归一化参数
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

# 2. 划分训练集和验证集
indices = list(range(len(train_data)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

# 3. 改进的模型
class ImprovedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )
    
    def forward(self, uav_img, sat_img):
        uav_feat = self.conv(uav_img).view(uav_img.size(0), -1)
        sat_feat = self.conv(sat_img).view(sat_img.size(0), -1)
        combined = torch.cat([uav_feat, sat_feat], dim=1)
        return {'fine_coords': self.fc(combined)}

# 4. 训练准备
model = ImprovedModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# 生成虚拟图像数据
uav_imgs = torch.randn(len(train_data), 3, 256, 256)
sat_imgs = torch.randn(len(train_data), 3, 512, 512)

# 分离训练/验证数据
train_uav = uav_imgs[train_idx]
train_sat = sat_imgs[train_idx]
train_labels = norm_labels[train_idx]

val_uav = uav_imgs[val_idx]
val_sat = sat_imgs[val_idx]
val_labels = norm_labels[val_idx]

print(f"\n📊 数据统计:")
print(f"训练集: {len(train_idx)} 个样本")
print(f"验证集: {len(val_idx)} 个样本")
print(f"坐标范围: 纬度 [{lat_min:.3f}, {lat_max:.3f}]")
print(f"          经度 [{lon_min:.3f}, {lon_max:.3f}]")

# 5. 训练循环
epochs = 50
batch_size = 16
train_losses = []
val_losses = []

print(f"\n⏳ 开始训练 {epochs} 轮...")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    # 随机打乱训练数据
    indices = torch.randperm(len(train_idx))
    
    for i in range(0, len(train_idx), batch_size):
        batch_indices = indices[i:i+batch_size]
        
        batch_uav = train_uav[batch_indices]
        batch_sat = train_sat[batch_indices]
        batch_labels = train_labels[batch_indices]
        
        outputs = model(batch_uav, batch_sat)
        loss = criterion(outputs['fine_coords'], batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / (len(train_idx) // batch_size)
    train_losses.append(avg_train_loss)
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_uav, val_sat)
        val_loss = criterion(val_outputs['fine_coords'], val_labels)
        val_losses.append(val_loss.item())
    
    # 更新学习率
    scheduler.step()
    
    # 每5轮打印一次
    if (epoch + 1) % 5 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}: 训练损失={avg_train_loss:.6f}, "
              f"验证损失={val_loss.item():.6f}, LR={current_lr:.6f}")
        
        # 显示预测示例
        with torch.no_grad():
            test_outputs = model(uav_imgs[:3], sat_imgs[:3])
            for j in range(3):
                pred = test_outputs['fine_coords'][j].numpy()
                true = norm_labels[j].numpy()
                print(f"  样本{j+1}: 预测({pred[0]:.3f},{pred[1]:.3f}) 真实({true[0]:.3f},{true[1]:.3f})")

# 6. 保存模型
os.makedirs('improved_model', exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'lat_min': lat_min, 'lat_max': lat_max,
    'lon_min': lon_min, 'lon_max': lon_max,
    'train_size': len(train_data),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'final_train_loss': train_losses[-1],
    'final_val_loss': val_losses[-1]
}, 'improved_model/improved_trained.pth')

print(f"\n✅ 训练完成! 模型保存到: improved_model/improved_trained.pth")

# 7. 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(val_losses, 'r--', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('improved_model/loss_curves.png', dpi=150, bbox_inches='tight')
print("✅ 保存损失曲线图: improved_model/loss_curves.png")

# 8. 最终评估
model.eval()
with torch.no_grad():
    # 训练集评估
    train_outputs = model(train_uav[:50], train_sat[:50])
    train_pred = train_outputs['fine_coords'].numpy()
    train_true = train_labels[:50].numpy()
    
    # 验证集评估
    val_outputs = model(val_uav[:50], val_sat[:50])
    val_pred = val_outputs['fine_coords'].numpy()
    val_true = val_labels[:50].numpy()
    
    # 计算平均误差
    train_error = np.mean(np.abs(train_pred - train_true))
    val_error = np.mean(np.abs(val_pred - val_true))
    
    print(f"\n📊 最终评估结果:")
    print(f"训练集平均误差: {train_error:.6f}")
    print(f"验证集平均误差: {val_error:.6f}")
    print(f"损失收敛情况: 初始损失={train_losses[0]:.6f}, 最终损失={train_losses[-1]:.6f}")
    
    if train_losses[-1] < train_losses[0] * 0.5:
        print("✅ 损失明显收敛!")
    else:
        print("⚠️  损失收敛不够明显，可能需要更多训练或调整超参数")