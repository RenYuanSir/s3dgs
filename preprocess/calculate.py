import numpy as np
import plyfile
import torch


def calculate_robust_scene_scale(ply_path, quantile=0.9):
    """
    通过过滤离群点，计算仅针对前景物体的 scene_scale
    """
    # 1. 读取 COLMAP 稀疏点云
    plydata = plyfile.PlyData.read(ply_path)
    v = plydata['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1)

    # 2. 计算几何中心 (使用中位数抗干扰)
    center = np.median(xyz, axis=0)

    # 3. 计算所有点到中心的距离
    dists = np.linalg.norm(xyz - center, axis=1)

    # 4. 关键：过滤掉背景点 (假设番茄是前景，占据中心)
    # 取 90% 分位数的距离作为“有效半径”
    # 这会忽略掉远处 10% 的背景墙面/天花板噪点
    robust_radius = np.quantile(dists, quantile)

    # 5. gsplat 推荐 scene_scale 为“场景范围”，通常取直径或半径的某个倍数
    # 为了让细节更敏感，我们稍微收缩这个范围
    scene_scale = float(robust_radius)

    print(f"Point Cloud Stats:")
    print(f"  - Total points: {len(xyz)}")
    print(f"  - Median Center: {center}")
    print(f"  - Max Distance (Outliers): {np.max(dists):.4f}")
    print(f"  - Robust Radius (90%): {robust_radius:.4f}")
    print(f"Recommended scene_scale: {scene_scale:.4f}")

    return scene_scale


# 使用你的路径
pcd_path = r"D:\PythonProject\PythonProject\data\video_data\colmap_data\video2_output_ply\sparse\0\points3D.ply"
calculate_robust_scene_scale(pcd_path)