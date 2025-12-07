# utils/vis.py
"""
可视化工具
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_matching_pair(uav_img, sat_img, pred_coords=None, true_coords=None):
    """可视化单个匹配对"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # UAV图像
    if isinstance(uav_img, torch.Tensor):
        uav_img = uav_img.cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(uav_img)
    axes[0].set_title('UAV Image')
    axes[0].axis('off')
    
    # 卫星图像
    if isinstance(sat_img, torch.Tensor):
        sat_img = sat_img.cpu().numpy().transpose(1, 2, 0)
    axes[1].imshow(sat_img)
    axes[1].set_title('Satellite Image')
    axes[1].axis('off')
    
    # 如果提供了坐标，在卫星图像上标记
    if pred_coords is not None or true_coords is not None:
        axes[2].imshow(sat_img)
        
        if pred_coords is not None:
            # 假设pred_coords是归一化坐标
            h, w = sat_img.shape[:2]
            pred_x = int(pred_coords[0] * w)
            pred_y = int(pred_coords[1] * h)
            axes[2].scatter(pred_x, pred_y, c='red', s=200, marker='X', 
                           linewidths=3, label='Predicted')
        
        if true_coords is not None:
            true_x = int(true_coords[0] * w)
            true_y = int(true_coords[1] * h)
            axes[2].scatter(true_x, true_y, c='green', s=200, marker='o', 
                           linewidths=3, label='Ground Truth')
        
        axes[2].set_title('Location Comparison')
        axes[2].legend()
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def plot_attention_heatmap(attention_weights, uav_img=None, sat_img=None):
    """可视化注意力热力图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # UAV图像
    if uav_img is not None:
        if isinstance(uav_img, torch.Tensor):
            uav_img = uav_img.cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(uav_img)
        axes[0].set_title('UAV Image')
        axes[0].axis('off')
    
    # 卫星图像
    if sat_img is not None:
        if isinstance(sat_img, torch.Tensor):
            sat_img = sat_img.cpu().numpy().transpose(1, 2, 0)
        axes[1].imshow(sat_img)
        axes[1].set_title('Satellite Image')
        axes[1].axis('off')
    
    # 注意力热力图
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # 平均所有头的注意力
    if len(attention_weights.shape) == 3:  # [batch, num_tokens, num_sat_patches]
        attn_mean = attention_weights.mean(axis=(0, 1))
        attn_map = attn_mean.reshape(int(np.sqrt(len(attn_mean))), -1)
    else:
        attn_map = attention_weights
    
    im = axes[2].imshow(attn_map, cmap='hot', interpolation='nearest')
    axes[2].set_title('Attention Heatmap')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    return fig

def plot_coarse_matching(coarse_map, sat_img=None):
    """可视化粗匹配热力图"""
    fig, axes = plt.subplots(1, 2 if sat_img is not None else 1, figsize=(10, 5))
    
    if sat_img is not None:
        if isinstance(sat_img, torch.Tensor):
            sat_img = sat_img.cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(sat_img)
        axes[0].set_title('Satellite Image')
        axes[0].axis('off')
        
        ax = axes[1]
    else:
        ax = axes
    
    if isinstance(coarse_map, torch.Tensor):
        coarse_map = coarse_map.cpu().numpy()
    
    im = ax.imshow(coarse_map, cmap='viridis')
    ax.set_title('Coarse Matching Heatmap (16x16)')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # 标记最大值位置
    max_pos = np.unravel_index(np.argmax(coarse_map), coarse_map.shape)
    ax.scatter(max_pos[1], max_pos[0], c='red', s=100, marker='x', linewidths=2)
    
    plt.tight_layout()
    return fig
def plot_training_history(train_losses, val_losses, val_distances, save_path=None):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 距离误差
    if val_distances and len(val_distances[0]) >= 2:
        avg_dist = [d[0] for d in val_distances]
        median_dist = [d[1] for d in val_distances]
        
        axes[1].plot(epochs, avg_dist, 'g-', label='Avg Distance', linewidth=2)
        axes[1].plot(epochs, median_dist, 'orange', label='Median Distance', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Distance (m)', fontsize=12)
        axes[1].set_title('Geolocation Error', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    # 误差范围
    if val_distances and len(val_distances[0]) >= 4:
        min_dist = [d[2] for d in val_distances]
        max_dist = [d[3] for d in val_distances]
        
        axes[2].fill_between(epochs, min_dist, max_dist, alpha=0.3, color='blue', label='Error Range')
        if 'avg_dist' in locals():
            axes[2].plot(epochs, avg_dist, 'b-', label='Avg Distance', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Distance (m)', fontsize=12)
        axes[2].set_title('Error Range', fontsize=14)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    else:
        return fig