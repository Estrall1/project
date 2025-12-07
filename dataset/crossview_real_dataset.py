# dataset/crossview_real_dataset.py
"""
使用真实标签的跨视角地理定位数据集
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json

class RealUniversityDataset(Dataset):
    """使用真实标签的University数据集"""
    
    def __init__(self, root_dir='University-Release', split='train',
                 uav_size=(256, 256), sat_size=(512, 512), max_samples=None):
        super().__init__()
        
        self.root_dir = root_dir
        self.split = split
        self.uav_size = uav_size
        self.sat_size = sat_size
        self.max_samples = max_samples
        
        print(f"\n📂 初始化 {split} 数据集 (真实标签)...")
        
        # 加载标签文件
        label_file = f'{split}_annotations.json'
        if os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"📄 加载标签文件: {label_file}")
        else:
            print(f"❌ 标签文件不存在: {label_file}")
            self.data = []
        
        # 处理不同格式
        if split == 'train':
            # 训练集：直接使用列表
            self.annotations = self.data
            self.is_test = False
        else:
            # 测试集：需要特殊处理
            self.queries = self.data.get('queries', [])
            self.gallery = self.data.get('gallery', [])
            self.is_test = True
            
            # 为测试创建样本列表（每个查询匹配对应的gallery）
            self.samples = self._create_test_samples()
            self.annotations = self.samples
        
        print(f"✅ 加载 {len(self.annotations)} 个样本")
        
        # 限制样本数量
        if max_samples and len(self.annotations) > max_samples:
            self.annotations = self.annotations[:max_samples]
            print(f"📊 限制为 {max_samples} 个样本")
    
    def _create_test_samples(self):
        """为测试集创建样本（查询-参考配对） - 修复版"""
        samples = []
        
        print(f"查询数量: {len(self.queries)}")
        print(f"参考库数量: {len(self.gallery)}")
        
        # 方法1: 简单配对（每个查询配第一个参考）
        # 这用于测试，实际应该是检索任务
        for i, query in enumerate(self.queries[:min(100, len(self.queries))]):
            query_path = query.get('path', '')
            query_id = query.get('query_id', '')
            
            # 使用第一个参考图像
            if self.gallery:
                gallery = self.gallery[i % len(self.gallery)]  # 循环使用
                samples.append({
                    'uav_path': query_path,
                    'sat_path': gallery.get('path', ''),
                    'lat': gallery.get('lat', 0.0),
                    'lon': gallery.get('lon', 0.0),
                    'query_id': query_id,
                    'gallery_id': gallery.get('gallery_id', '')
                })
        
        print(f"🔗 创建 {len(samples)} 个测试样本配对")
        return samples
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        try:
            # 获取路径和坐标
            if isinstance(annotation, dict):
                # 训练集格式
                if 'uav_path' in annotation:
                    uav_path = annotation['uav_path']
                    sat_path = annotation['sat_path']
                # 测试集格式（我们已经转换了）
                else:
                    uav_path = annotation.get('uav_path', '')
                    sat_path = annotation.get('sat_path', '')
                
                lat = annotation.get('lat', 0.0)
                lon = annotation.get('lon', 0.0)
            else:
                # 备用方案
                lat, lon = 0.0, 0.0
                uav_path = sat_path = ''
            
            # 构建完整路径
            full_uav_path = os.path.join(self.root_dir, uav_path)
            full_sat_path = os.path.join(self.root_dir, sat_path)
            
            # 检查文件是否存在
            if not os.path.exists(full_uav_path):
                # 尝试不同的路径格式
                alt_path = uav_path.replace('street/', 'drone/').replace('/1.jpg', '/image-01.jpeg')
                full_uav_path = os.path.join(self.root_dir, alt_path)
            
            if not os.path.exists(full_sat_path):
                # 尝试不同的卫星图像路径
                alt_path = sat_path.replace('satellite/', 'satellite/')
                full_sat_path = os.path.join(self.root_dir, alt_path)
            
            # 加载图像
            uav_img = Image.open(full_uav_path).convert('RGB')
            sat_img = Image.open(full_sat_path).convert('RGB')
            
            # 调整大小
            uav_img = uav_img.resize(self.uav_size, Image.BILINEAR)
            sat_img = sat_img.resize(self.sat_size, Image.BILINEAR)
            
            lat_min, lat_max = 40.000, 40.098
            lon_min, lon_max = -75.000, -74.902
            
            norm_lat = (lat - lat_min) / (lat_max - lat_min)
            norm_lon = (lon - lon_min) / (lon_max - lon_min)
            
            # 确保在 [0, 1] 范围
            norm_lat = max(0.0, min(1.0, norm_lat))
            norm_lon = max(0.0, min(1.0, norm_lon))
            
            # 转换为张量
            uav_array = np.array(uav_img, dtype=np.float32) / 255.0
            sat_array = np.array(sat_img, dtype=np.float32) / 255.0
            
            uav_tensor = torch.from_numpy(uav_array).permute(2, 0, 1)
            sat_tensor = torch.from_numpy(sat_array).permute(2, 0, 1)
            
        except Exception as e:
            print(f"⚠️ 加载图像出错 (idx={idx}): {e}")
            print(f"  UAV路径: {full_uav_path if 'full_uav_path' in locals() else 'N/A'}")
            print(f"  卫星路径: {full_sat_path if 'full_sat_path' in locals() else 'N/A'}")
            
            # 返回随机张量
            uav_tensor = torch.randn(3, *self.uav_size)
            sat_tensor = torch.randn(3, *self.sat_size)
            lat, lon = 0.0, 0.0
        
        return {
        'uav': uav_tensor,
        'satellite': sat_tensor,
        'lat': torch.tensor(norm_lat, dtype=torch.float32),
        'lon': torch.tensor(norm_lon, dtype=torch.float32),
        'raw_lat': torch.tensor(lat, dtype=torch.float32),  # 保留原始坐标
        'raw_lon': torch.tensor(lon, dtype=torch.float32),
        'idx': idx
    }


# 测试函数
def test_dataset():
    """测试数据集"""
    print("🧪 测试真实标签数据集...")
    
    # 测试训练集
    print("\n1. 测试训练集:")
    train_dataset = RealUniversityDataset(
        root_dir='University-Release',
        split='train',
        max_samples=5
    )
    
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"  样本数: {len(train_dataset)}")
        print(f"  UAV形状: {sample['uav'].shape}")
        print(f"  卫星形状: {sample['satellite'].shape}")
        print(f"  真实坐标: ({sample['lat'].item():.6f}, {sample['lon'].item():.6f})")
        print(f"  数据类型: {sample['uav'].dtype}")
    
    # 测试测试集
    print("\n2. 测试测试集:")
    test_dataset = RealUniversityDataset(
        root_dir='University-Release',
        split='test',
        max_samples=5
    )
    
    if len(test_dataset) > 0:
        sample = test_dataset[0]
        print(f"  样本数: {len(test_dataset)}")
        print(f"  真实坐标: ({sample['lat'].item():.6f}, {sample['lon'].item():.6f})")

if __name__ == '__main__':
    test_dataset()