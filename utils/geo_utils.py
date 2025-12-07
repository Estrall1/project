# utils/geo_utils.py
"""
地理工具函数
"""
import math
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2, R=6371000.0):
    """计算两个经纬度点之间的Haversine距离（米）"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def latlon_to_mercator(lat, lon):
    """将经纬度转换为墨卡托投影坐标"""
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return x, y

def mercator_to_latlon(x, y):
    """将墨卡托投影坐标转换为经纬度"""
    lon = x * 180 / 20037508.34
    y = y * 180 / 20037508.34
    lat = 180 / math.pi * (2 * math.atan(math.exp(y * math.pi / 180)) - math.pi / 2)
    return lat, lon

def calculate_distance_matrix(pred_coords, true_coords):
    """计算预测和真实坐标之间的距离矩阵"""
    n = len(pred_coords)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = haversine_distance(
                pred_coords[i][0], pred_coords[i][1],
                true_coords[j][0], true_coords[j][1]
            )
    return distances