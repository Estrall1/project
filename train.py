# train_real_fixed.py
"""
修复版训练代码 - 确保能加载完整数据集
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
import json
import sys

# 添加项目根目录到路径
sys.path.append('.')

# 导入
try:
    from dataset.crossview_real_dataset import RealUniversityDataset
    from models.crossview_model import AdvancedCrossViewGeolocator
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 设置matplotlib字体 - 解决中文显示问题
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用英文字体避免乱码
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def haversine_distance(pred_coords, true_coords, R=6371000.0):
    """计算Haversine距离"""
    pred_lat = torch.deg2rad(pred_coords[:, 0])
    pred_lon = torch.deg2rad(pred_coords[:, 1])
    true_lat = torch.deg2rad(true_coords[:, 0])
    true_lon = torch.deg2rad(true_coords[:, 1])
    
    dlat = true_lat - pred_lat
    dlon = true_lon - pred_lon
    
    a = torch.sin(dlat/2)**2 + torch.cos(pred_lat) * torch.cos(true_lat) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    distance = R * c
    
    return distance

class RealGeolocationLoss(nn.Module):
    """真实地理定位损失"""
    def __init__(self, lambda_coord=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()
        self.lambda_coord = lambda_coord
        
    def forward(self, outputs, labels):
        losses = {}
        
        if 'fine_coords' in outputs and 'latlon' in labels:
            # MSE损失
            mse_loss = self.mse(outputs['fine_coords'], labels['latlon'])
            
            # Huber损失（更鲁棒）
            huber_loss = self.huber(outputs['fine_coords'], labels['latlon'])
            
            # Haversine距离（用于监控）
            with torch.no_grad():
                haversine_dist = haversine_distance(outputs['fine_coords'], labels['latlon'])
                losses['haversine'] = haversine_dist.mean()
            
            # 组合损失
            total_loss = self.lambda_coord * (mse_loss + huber_loss)
            losses['total'] = total_loss
            losses['mse'] = mse_loss
            losses['huber'] = huber_loss
        
        return losses

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    haversine_distances = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # 获取数据 - 添加调试信息
            uav_imgs = batch['uav'].to(device)
            satellite_imgs = batch['satellite'].to(device)
            lat_labels = batch['lat'].to(device)
            lon_labels = batch['lon'].to(device)
            
            # 检查数据形状
            B = uav_imgs.shape[0]
            if B == 0:
                logger.warning(f"批次 {batch_idx} 为空，跳过")
                continue
            
            # 准备标签
            latlon_labels = torch.cat([lat_labels.view(B, 1), lon_labels.view(B, 1)], dim=1)
            
            # 检查标签范围（应该是0-1的归一化值）
            if torch.any(latlon_labels < 0) or torch.any(latlon_labels > 1):
                logger.warning(f"批次 {batch_idx}: 标签超出[0,1]范围")
            
            # 前向传播
            outputs = model(uav_imgs, satellite_imgs)
            
            # 检查输出
            if 'fine_coords' not in outputs:
                logger.error(f"模型没有输出'fine_coords'")
                continue
            
            # 计算损失
            labels = {'latlon': latlon_labels}
            losses = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 记录损失
            total_loss += losses['total'].item()
            
            # 记录Haversine距离
            if 'haversine' in losses:
                haversine_distances.append(losses['haversine'].item())
            
            # 更新进度条
            avg_dist = np.mean(haversine_distances[-10:]) if haversine_distances else 0
            progress_bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'dist': f"{avg_dist:.1f}m"
            })
            
        except Exception as e:
            logger.error(f"训练批次 {batch_idx} 出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    num_batches = len(dataloader)
    metrics = {
        'total_loss': total_loss / num_batches if num_batches > 0 else 0,
        'avg_distance': np.mean(haversine_distances) if haversine_distances else 0,
    }
    
    return metrics

def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    haversine_distances = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        for batch_idx, batch in enumerate(progress_bar):
            try:
                uav_imgs = batch['uav'].to(device)
                satellite_imgs = batch['satellite'].to(device)
                lat_labels = batch['lat'].to(device)
                lon_labels = batch['lon'].to(device)
                
                B = uav_imgs.shape[0]
                if B == 0:
                    continue
                
                latlon_labels = torch.cat([lat_labels.view(B, 1), lon_labels.view(B, 1)], dim=1)
                
                outputs = model(uav_imgs, satellite_imgs)
                labels = {'latlon': latlon_labels}
                losses = criterion(outputs, labels)
                
                total_loss += losses['total'].item()
                
                if 'haversine' in losses:
                    haversine_distances.append(losses['haversine'].item())
                
                avg_dist = np.mean(haversine_distances[-10:]) if haversine_distances else 0
                progress_bar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'dist': f"{avg_dist:.1f}m"
                })
                
            except Exception as e:
                logger.error(f"验证批次 {batch_idx} 出错: {e}")
                continue
    
    num_batches = len(dataloader)
    if num_batches == 0:
        return {
            'total_loss': 0,
            'avg_distance': 0,
            'median_distance': 0,
            'min_distance': 0,
            'max_distance': 0,
        }
    
    metrics = {
        'total_loss': total_loss / num_batches,
        'avg_distance': np.mean(haversine_distances) if haversine_distances else 0,
        'median_distance': np.median(haversine_distances) if haversine_distances else 0,
        'min_distance': np.min(haversine_distances) if haversine_distances else 0,
        'max_distance': np.max(haversine_distances) if haversine_distances else 0,
    }
    
    return metrics

def check_data_samples(data_root='University-Release'):
    """检查数据集样本数量"""
    try:
        # 检查标签文件
        train_json = os.path.join(data_root, 'train', 'train_annotations.json')
        test_json = os.path.join(data_root, 'test', 'test_annotations.json')
        
        if os.path.exists(train_json):
            with open(train_json, 'r') as f:
                train_data = json.load(f)
            logger.info(f"训练标签文件: {len(train_data)} 个样本")
        
        if os.path.exists(test_json):
            with open(test_json, 'r') as f:
                test_data = json.load(f)
            logger.info(f"测试标签文件: {len(test_data)} 个样本")
        
        # 检查图像文件
        uav_train_dir = os.path.join(data_root, 'train', 'uav')
        sat_train_dir = os.path.join(data_root, 'train', 'satellite')
        
        if os.path.exists(uav_train_dir):
            uav_files = [f for f in os.listdir(uav_train_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            logger.info(f"UAV训练图像: {len(uav_files)} 个文件")
        
        if os.path.exists(sat_train_dir):
            sat_files = [f for f in os.listdir(sat_train_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            logger.info(f"卫星训练图像: {len(sat_files)} 个文件")
            
    except Exception as e:
        logger.warning(f"检查数据集时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train with Real Geolocation Labels - Fixed Version')
    parser.add_argument('--data_root', type=str, default='University-Release', 
                       help='数据集根目录，包含train/test子目录')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备')
    parser.add_argument('--train_samples', type=int, default=-1, 
                       help='训练样本数，-1表示使用全部')
    parser.add_argument('--val_samples', type=int, default=100, 
                       help='验证样本数，-1表示使用全部')
    parser.add_argument('--num_workers', type=int, default=2, 
                       help='数据加载线程数，0表示不使用多线程')
    parser.add_argument('--save_every', type=int, default=5, 
                       help='每多少epoch保存一次模型')
    
    args = parser.parse_args()
    
    # 检查数据集
    logger.info(f"数据集根目录: {args.data_root}")
    if not os.path.exists(args.data_root):
        logger.error(f"数据集目录不存在: {args.data_root}")
        logger.info("请确保数据集已下载并解压到正确位置")
        logger.info("数据集应该包含 train/ 和 test/ 目录")
        return
    
    check_data_samples(args.data_root)
    
    # 设置设备
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'使用GPU: {torch.cuda.get_device_name(args.gpu)}')
    else:
        device = torch.device('cpu')
        logger.info('使用CPU')
    
    # 创建数据集 - 添加更多调试信息
    logger.info("加载真实标签数据集...")
    try:
        train_dataset = RealUniversityDataset(
            root_dir=args.data_root,
            split='train',
            uav_size=(256, 256),
            sat_size=(512, 512),
            max_samples=args.train_samples if args.train_samples > 0 else None
        )
        
        val_dataset = RealUniversityDataset(
            root_dir=args.data_root,
            split='test',
            uav_size=(256, 256),
            sat_size=(512, 512),
            max_samples=args.val_samples if args.val_samples > 0 else None
        )
        
        logger.info(f"✅ 数据集加载成功:")
        logger.info(f"   训练样本: {len(train_dataset)}")
        logger.info(f"   验证样本: {len(val_dataset)}")
        
        if len(train_dataset) > 0:
            # 显示第一个样本的信息
            sample = train_dataset[0]
            logger.info(f"   数据形状: UAV={sample['uav'].shape}, Satellite={sample['satellite'].shape}")
            logger.info(f"   坐标示例: lat={sample['lat'].item():.6f}, lon={sample['lon'].item():.6f}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=min(2, args.num_workers),  # 限制工作线程数
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=min(2, args.num_workers),
            pin_memory=True if torch.cuda.is_available() else False
        )
        
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试使用备用数据集
        logger.info("尝试使用备用数据集路径...")
        alt_paths = [
            '.',
            './data',
            '../University-Release',
            '../../University-Release'
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(os.path.join(alt_path, 'train_annotations.json')):
                logger.info(f"找到备用路径: {alt_path}")
                args.data_root = alt_path
                break
        
        # 重新尝试
        try:
            train_dataset = RealUniversityDataset(
                root_dir=args.data_root,
                split='train',
                uav_size=(256, 256),
                sat_size=(512, 512),
                max_samples=args.train_samples if args.train_samples > 0 else None
            )
            val_dataset = RealUniversityDataset(
                root_dir=args.data_root,
                split='test',
                uav_size=(256, 256),
                sat_size=(512, 512),
                max_samples=args.val_samples if args.val_samples > 0 else None
            )
            logger.info("✅ 备用路径加载成功")
        except:
            logger.error("所有路径都失败，请检查数据集")
            return
    
    # 创建模型
    logger.info("初始化模型...")
    try:
        model = AdvancedCrossViewGeolocator().to(device)
        logger.info(f"✅ 模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        logger.error(f"创建模型失败: {e}")
        # 使用简化模型作为后备
        logger.info("使用简化模型作为后备...")
        from models.simple_model import SimpleCrossViewModel
        model = SimpleCrossViewModel().to(device)
    
    # 损失函数和优化器
    criterion = RealGeolocationLoss(lambda_coord=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('checkpoints_real', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"检查点保存到: {save_dir}")
    
    # 保存配置
    config_file = os.path.join(save_dir, 'config.txt')
    with open(config_file, 'w') as f:
        f.write(f"训练时间: {timestamp}\n")
        f.write(f"数据根目录: {args.data_root}\n")
        f.write(f"训练样本: {len(train_dataset)}\n")
        f.write(f"验证样本: {len(val_dataset)}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"设备: {device}\n")
    
    # 训练循环
    best_val_distance = float('inf')
    train_losses = []
    val_losses = []
    val_distances = []
    
    logger.info("开始训练...")
    
    for epoch in range(args.epochs):
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch+1}/{args.epochs}')
        logger.info(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 训练
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_metrics['total_loss'])
        
        logger.info(f'训练损失: {train_metrics["total_loss"]:.6f}')
        logger.info(f'平均距离: {train_metrics["avg_distance"]:.1f}米')
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_metrics['total_loss'])
        val_distances.append([
            val_metrics['avg_distance'],
            val_metrics['median_distance'],
            val_metrics['min_distance'],
            val_metrics['max_distance']
        ])
        
        logger.info(f'验证损失: {val_metrics["total_loss"]:.6f}')
        logger.info(f'平均距离: {val_metrics["avg_distance"]:.1f}米')
        logger.info(f'中位数距离: {val_metrics["median_distance"]:.1f}米')
        logger.info(f'范围: [{val_metrics["min_distance"]:.1f}, {val_metrics["max_distance"]:.1f}]米')
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存模型
        if (epoch + 1) % args.save_every == 0:
            model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['total_loss'],
                'val_loss': val_metrics['total_loss'],
                'val_distance': val_metrics['avg_distance'],
            }, model_path)
            logger.info(f'💾 保存中间模型: {model_path}')
        
        # 保存最佳模型
        if val_metrics['avg_distance'] < best_val_distance:
            best_val_distance = val_metrics['avg_distance']
            model_path = os.path.join(save_dir, f'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['total_loss'],
                'val_loss': val_metrics['total_loss'],
                'val_distance': val_metrics['avg_distance'],
                'best_val_distance': best_val_distance,
            }, model_path)
            logger.info(f'✅ 保存最佳模型: {model_path}')
    
    # 保存最终模型
    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_distances': val_distances,
        'best_val_distance': best_val_distance,
        'args': vars(args)
    }, final_path)
    logger.info(f'💾 保存最终模型: {final_path}')
    
    # 绘制结果 - 使用英文标签避免乱码
    try:
        results_dir = 'results_real'
        os.makedirs(results_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        epochs_range = range(1, len(train_losses) + 1)
        ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss', marker='o', markersize=4)
        ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss', marker='s', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Progress\nBest Validation Distance: {best_val_distance:.1f}m')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 距离误差
        ax2.plot(epochs_range, [d[0] for d in val_distances], 'g-', label='Avg Distance', marker='^', markersize=4)
        ax2.plot(epochs_range, [d[1] for d in val_distances], 'm-', label='Median Distance', marker='d', markersize=4)
        ax2.axhline(y=25.3, color='r', linestyle='--', alpha=0.7, label='Baseline (25.3m)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Distance Error (m)')
        ax2.set_title('Validation Distance Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'training_results_{timestamp}.png'), 
                   dpi=150, bbox_inches='tight')
        
        # 单独保存一个简单的损失曲线
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f'📊 结果图表保存到: {results_dir}/')
        
    except Exception as e:
        logger.warning(f"绘制图表失败: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    
    logger.info(f'\n{"="*60}')
    logger.info('🎉 训练完成!')
    logger.info(f'训练轮数: {args.epochs}')
    logger.info(f'最佳验证距离: {best_val_distance:.1f}米')
    logger.info(f'与Baseline对比: {best_val_distance:.1f}m vs 25.3m')
    logger.info(f'模型保存到: {save_dir}/')
    
    # 显示关键指标
    if len(val_distances) > 0:
        final_avg = val_distances[-1][0]
        final_median = val_distances[-1][1]
        logger.info(f'最终验证距离: 平均={final_avg:.1f}m, 中位数={final_median:.1f}m')

if __name__ == '__main__':
    main()