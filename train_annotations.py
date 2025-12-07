# fix_train_annotations.py
import json
import os
from pathlib import Path

def fix_train_annotations():
    base_path = "University-Release/train"
    
    # 获取所有卫星图文件夹
    sat_folders = sorted([f for f in os.listdir(os.path.join(base_path, "satellite")) 
                         if os.path.isdir(os.path.join(base_path, "satellite", f))])
    
    annotations = []
    
    for folder in sat_folders:
        # 卫星图路径
        sat_folder_path = os.path.join(base_path, "satellite", folder)
        sat_images = [f for f in os.listdir(sat_folder_path) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not sat_images:
            continue
        
        sat_path = f"train/satellite/{folder}/{sat_images[0]}"
        
        # 使用街景图像作为UAV视角
        street_folder_path = os.path.join(base_path, "street", folder)
        if os.path.exists(street_folder_path):
            street_images = [f for f in os.listdir(street_folder_path) 
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if street_images:
                # 使用第一张街景图
                street_path = f"train/street/{folder}/{street_images[0]}"
                
                # 创建虚拟坐标
                folder_num = int(folder) if folder.isdigit() else hash(folder) % 1000
                lat = 40.0 + (folder_num % 100) / 1000.0
                lon = -75.0 + (folder_num % 100) / 1000.0
                
                annotations.append({
                    "id": folder,
                    "uav_path": street_path,  # 注意：这里使用街景路径
                    "sat_path": sat_path,
                    "lat": lat,
                    "lon": lon,
                    "note": "使用街景图像作为UAV视角近似"
                })
    
    with open("train_annotations_fixed.json", "w") as f:
        json.dump(annotations, f, indent=2)
    
    print(f"创建了 {len(annotations)} 个训练样本（使用街景作为UAV视角）")
    return annotations

if __name__ == "__main__":
    fix_train_annotations()