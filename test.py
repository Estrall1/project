# test_correct.py
"""
正确的测试脚本 - 包含坐标归一化
"""
import os
import torch
import numpy as np
import json
import math
import matplotlib.pyplot as plt

# 方法1：尝试设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 计算归一化参数
with open('train_annotations.json', 'r') as f:
    train_data = json.load(f)

lats = [item['lat'] for item in train_data]
lons = [item['lon'] for item in train_data]

LAT_MIN, LAT_MAX = min(lats), max(lats)
LON_MIN, LON_MAX = min(lons), max(lons)

print("📍 坐标归一化参数:")
print(f"  纬度: [{LAT_MIN:.3f}, {LAT_MAX:.3f}]")
print(f"  经度: [{LON_MIN:.3f}, {LON_MAX:.3f}]")

def normalize_coords(lats, lons):
    """归一化坐标到 [0, 1]"""
    norm_lats = (lats - LAT_MIN) / (LAT_MAX - LAT_MIN)
    norm_lons = (lons - LON_MIN) / (LON_MAX - LON_MIN)
    return norm_lats, norm_lons

def denormalize_coords(norm_lats, norm_lons):
    """反归一化坐标"""
    lats = norm_lats * (LAT_MAX - LAT_MIN) + LAT_MIN
    lons = norm_lons * (LON_MAX - LON_MIN) + LON_MIN
    return lats, lons

def haversine_distance(lat1, lon1, lat2, lon2, R=6371000.0):
    """计算Haversine距离（米）"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def test_model_correctly(checkpoint_path, num_samples=50):
    """正确测试模型"""
    print(f"\n🧪 测试模型: {checkpoint_path}")
    
    # 导入模型
    from models.crossview_model import AdvancedCrossViewGeolocator
    from dataset.crossview_real_dataset import RealUniversityDataset
    from torch.utils.data import DataLoader
    
    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    dataset = RealUniversityDataset(
        split='train',  # 用训练集测试
        max_samples=num_samples
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # 加载模型
    model = AdvancedCrossViewGeolocator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 测试
    all_distances = []
    
    with torch.no_grad():
        for batch in dataloader:
            uav_imgs = batch['uav'].to(device)
            sat_imgs = batch['satellite'].to(device)
            raw_lats = batch['lat'].cpu().numpy()  # 注意：这里已经是归一化的！
            raw_lons = batch['lon'].cpu().numpy()
            
            # 模型预测（归一化坐标）
            outputs = model(uav_imgs, sat_imgs)
            pred_norm_coords = outputs['fine_coords'].cpu().numpy()
            
            # 反归一化预测坐标
            pred_lats, pred_lons = denormalize_coords(
                pred_norm_coords[:, 0], pred_norm_coords[:, 1]
            )
            
            # 反归一化真实坐标
            true_lats, true_lons = denormalize_coords(raw_lats, raw_lons)
            
            # 计算真实距离
            for i in range(len(pred_lats)):
                distance = haversine_distance(
                    pred_lats[i], pred_lons[i],
                    true_lats[i], true_lons[i]
                )
                all_distances.append(distance)
    
    # 计算统计
    distances = np.array(all_distances)
    
    metrics = {
        'num_samples': len(distances),
        'avg_distance_m': float(np.mean(distances)),
        'avg_distance_km': float(np.mean(distances) / 1000),
        'median_distance_m': float(np.median(distances)),
        'min_distance_m': float(np.min(distances)),
        'max_distance_m': float(np.max(distances)),
        'std_distance_m': float(np.std(distances)),
    }
    
    # 显示结果
    print(f"\n📊 正确测试结果:")
    print(f"  样本数: {metrics['num_samples']}")
    print(f"  平均误差: {metrics['avg_distance_m']:.1f}米 ({metrics['avg_distance_km']:.2f}公里)")
    print(f"  中位数误差: {metrics['median_distance_m']:.1f}米")
    print(f"  最小误差: {metrics['min_distance_m']:.1f}米")
    print(f"  最大误差: {metrics['max_distance_m']:.1f}米")
    print(f"  标准差: {metrics['std_distance_m']:.1f}米")
    
    # 计算精度
    accuracy_10m = np.mean(distances < 10)
    accuracy_50m = np.mean(distances < 50)
    accuracy_100m = np.mean(distances < 100)
    
    print(f"\n🎯 精度指标:")
    print(f"  <10米精度: {accuracy_10m*100:.1f}%")
    print(f"  <50米精度: {accuracy_50m*100:.1f}%")
    print(f"  <100米精度: {accuracy_100m*100:.1f}%")
    
    return metrics

if __name__ == '__main__':
    # 测试最新模型
    checkpoint = "checkpoints_real/20251206_124644/final_model.pth"
    if os.path.exists(checkpoint):
        metrics = test_model_correctly(checkpoint, num_samples=50)
        
        # 与Baseline比较
        print(f"\n📈 与DRL Baseline对比:")
        print(f"  我们的模型: {metrics['avg_distance_m']:.1f}米")
        print(f"  DRL Baseline: 25.3米 (论文报告)")
        
        if metrics['avg_distance_m'] < 25.3:
            improvement = (25.3 - metrics['avg_distance_m']) / 25.3 * 100
            print(f"  ✅ 优于Baseline: 提升 {improvement:.1f}%")
        else:
            improvement = (metrics['avg_distance_m'] - 25.3) / 25.3 * 100
            print(f"  ⚠️  差于Baseline: 差 {improvement:.1f}%")
            
    else:
        print(f"❌ 检查点不存在: {checkpoint}")