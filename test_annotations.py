import os
import json
def create_test_annotations():
    base_path = "University-Release/test"
    
    # 查询图像（无人机）
    query_dir = os.path.join(base_path, "query_drone")
    query_folders = sorted([f for f in os.listdir(query_dir) 
                           if os.path.isdir(os.path.join(query_dir, f))])
    
    queries = []
    for folder in query_folders:
        query_folder_path = os.path.join(query_dir, folder)
        query_images = [f for f in os.listdir(query_folder_path) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for img in query_images:
            queries.append({
                "query_id": f"{folder}_{img.split('.')[0]}",
                "path": f"test/query_drone/{folder}/{img}"
            })
    
    # 数据库图像（卫星）
    gallery_dir = os.path.join(base_path, "gallery_satellite")
    gallery_folders = sorted([f for f in os.listdir(gallery_dir) 
                            if os.path.isdir(os.path.join(gallery_dir, f))])
    
    gallery = []
    for folder in gallery_folders:
        gallery_folder_path = os.path.join(gallery_dir, folder)
        gallery_images = [f for f in os.listdir(gallery_folder_path) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for img in gallery_images:
            gallery.append({
                "gallery_id": f"{folder}_{img.split('.')[0]}",
                "path": f"test/gallery_satellite/{folder}/{img}",
                "lat": 40.0 + int(folder) % 100 / 1000.0,  # 虚拟坐标
                "lon": -75.0 + int(folder) % 100 / 1000.0
            })
    
    test_data = {
        "queries": queries,
        "gallery": gallery
    }
    
    with open("test_annotations.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"创建了 {len(queries)} 个查询和 {len(gallery)} 个数据库图像")

create_test_annotations()