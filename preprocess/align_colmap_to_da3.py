import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R


def read_colmap_cameras(cam_path):
    """
    解析 cameras.txt
    返回: dict {camera_id: intrinsic_matrix (3x3)}
    """
    cam_dict = {}
    with open(cam_path, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            # params 可能会根据模型不同而变化，这里假设是 SIMPLE_RADIAL, PINHOLE 或 RADIAL
            # COLMAP standard:
            # SIMPLE_RADIAL: f, cx, cy, k
            # PINHOLE: fx, fy, cx, cy
            params = [float(p) for p in parts[4:]]

            K = np.eye(3)
            if model == "SIMPLE_PINHOLE" or model == "SIMPLE_RADIAL":
                f, cx, cy = params[0], params[1], params[2]
                K[0, 0] = f
                K[1, 1] = f
                K[0, 2] = cx
                K[1, 2] = cy
            elif model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                K[0, 0] = fx
                K[1, 1] = fy
                K[0, 2] = cx
                K[1, 2] = cy
            else:
                # 针对 OPENCV 等复杂模型，通常只取前4个参数作为近似线性内参
                # fx, fy, cx, cy, ...
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                K[0, 0] = fx
                K[1, 1] = fy
                K[0, 2] = cx
                K[1, 2] = cy

            cam_dict[cam_id] = K
    return cam_dict


def read_colmap_images(img_path):
    """
    解析 images.txt
    返回: dict {image_name: (camera_id, R (3x3), t (3x1))}
    注意：COLMAP 存储的是 World-to-Camera (w2c)
    """
    img_dict = {}
    with open(img_path, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#") or line == "":
                i += 1
                continue

            # Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            parts = line.split()
            img_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id = int(parts[8])
            name = parts[9]

            # Quaternion to Rotation Matrix
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()  # Scipy 使用 xyzw 顺序
            t = np.array([tx, ty, tz]).reshape(3, 1)

            img_dict[name] = (cam_id, rot, t)

            # Skip Line 2 (Points2D)
            i += 2
    return img_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_dir", required=True, help="包含 images.txt, cameras.txt 的目录")
    parser.add_argument("--image_dir", required=True, help="图片文件夹，用于确定排序顺序")
    parser.add_argument("--output_path", default="poses.npz", help="输出 npz 文件路径")
    args = parser.parse_args()

    cameras_file = os.path.join(args.colmap_dir, "cameras.txt")
    images_file = os.path.join(args.colmap_dir, "images.txt")

    print("1. 读取 COLMAP 数据...")
    cam_map = read_colmap_cameras(cameras_file)
    img_map = read_colmap_images(images_file)

    print("2. 匹配图片顺序...")
    # 关键步骤：DA3 读取图片通常是 os.listdir 排序或 sorted()
    # 我们必须模拟这个顺序，或者你明确指定一个列表
    sorted_image_names = sorted([
        f for f in os.listdir(args.image_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    intrinsics_list = []
    extrinsics_list = []
    valid_names = []

    for name in sorted_image_names:
        if name not in img_map:
            print(f"[Warning] 图片 {name} 在 COLMAP 中未找到位姿，将被跳过 (这会导致索引错位，请确保图片集一致!)")
            continue

        cam_id, rot, t = img_map[name]
        K = cam_map[cam_id]

        # 构造 Extrinsics (3x4)
        # API.md 要求: World-to-Camera transformation
        # COLMAP 已经是 w2c: P_cam = R * P_world + t
        # 所以直接拼接 [R | t]
        ext_4x4 = np.eye(4)
        ext_4x4[:3, :3] = rot
        ext_4x4[:3, 3:4] = t
        extrinsics_list.append(ext_4x4)

        intrinsics_list.append(K)
        valid_names.append(name)

    # 转换为 Numpy 数组
    # Shape: (N, 3, 3)
    intrinsics_np = np.stack(intrinsics_list).astype(np.float32)
    # Shape: (N, 3, 4)
    extrinsics_np = np.stack(extrinsics_list).astype(np.float32)

    print(f"3. 生成数据: {len(valid_names)} 帧")
    print(f"   Intrinsics shape: {intrinsics_np.shape}")
    print(f"   Extrinsics shape: {extrinsics_np.shape}")

    np.savez(args.output_path,
             intrinsics=intrinsics_np,
             extrinsics=extrinsics_np,
             image_names=valid_names)  # 把文件名也存进去，方便核对

    print(f"✅ 完成！已保存至 {args.output_path}")
    print("   请确保将 image_names 与传给 DA3 的 image list 顺序进行二次核对。")


if __name__ == "__main__":
    main()