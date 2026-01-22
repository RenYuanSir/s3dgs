import os
import json
import numpy as np
import cv2
from tqdm import tqdm

# --- 语义映射配置 ---
# Channel 0: U (Up Node) + 主茎上段
# Channel 1: N (Node)   + 节点中心 (枢纽)
# Channel 2: D (Down Node) + 主茎下段
# Channel 3: P (Peduncle) + 果梗/侧枝
CLASS_MAP = {
    'U': 0,
    'N': 1,
    'D': 2,
    'P': 3
}
NUM_CLASSES = 4

# --- 拓扑连接规则 ---
# (起点类别, 终点类别, 绘制在哪个通道, 粗细参考边, 粗细比例)
# 'w' = bbox_width, 'h' = bbox_height
SKELETON_RULES = [
    # U -> N 连线，画在 U 通道，基于框宽度，占 40%
    ('U', 'N', 0, 'w', 0.40),

    # N -> D 连线，画在 D 通道，基于框宽度，占 40%
    # 注意：方向是无所谓的，画线连通即可
    ('N', 'D', 2, 'w', 0.40),

    # N -> P 连线，画在 P 通道，基于框高度，占 15%
    ('N', 'P', 3, 'h', 0.15)
]


def generate_adaptive_heatmap(height, width, keypoints, bbox_w, bbox_h, sigma=20):
    """
    生成带有骨架连接的、尺度自适应的语义热力图
    修复了 OpenCV 内存布局不兼容的问题 (Channel First -> Draw -> Transpose)
    """
    # 1. 修改初始化：使用 (Channel, Height, Width) 格式
    # 这样 heatmap[channel_idx] 提取出的就是连续内存，cv2.line 不会报错
    heatmap = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)

    # 建立关键点坐标查询表 (过滤低置信度)
    kp_dict = {}
    for kp in keypoints:
        if kp['confidence'] > 0.5:
            kp_dict[kp['class']] = (int(kp['x']), int(kp['y']))

    # 2. 绘制骨架 (Skeleton)
    for start_cls, end_cls, channel_idx, ref_dim, ratio in SKELETON_RULES:
        if start_cls in kp_dict and end_cls in kp_dict:
            pt1 = kp_dict[start_cls]
            pt2 = kp_dict[end_cls]

            # 计算动态粗细
            if ref_dim == 'w':
                thickness = int(bbox_w * ratio)
            else:
                thickness = int(bbox_h * ratio)

            thickness = max(5, thickness)

            # [FIX] 直接在对应的 channel 上画线，现在它是连续的内存
            cv2.line(heatmap[channel_idx], pt1, pt2, color=1.0, thickness=thickness)

    # 3. 叠加关键点高斯球 (Hotspots)
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    for cls_name, (x, y) in kp_dict.items():
        if cls_name in CLASS_MAP:
            idx = CLASS_MAP[cls_name]

            dist_sq = (xx - x) ** 2 + (yy - y) ** 2
            gaussian = np.exp(-dist_sq / (2 * sigma ** 2))

            # [FIX] 使用 channel first 索引
            heatmap[idx] = np.maximum(heatmap[idx], gaussian)

    # 4. 全局高斯模糊 (Softening)
    ksize = int(sigma * 2) | 1

    for k in range(NUM_CLASSES):
        # [FIX] 使用 channel first 索引
        if np.max(heatmap[k]) > 0:
            heatmap[k] = cv2.GaussianBlur(
                heatmap[k],
                (ksize, ksize),
                0
            )

            max_val = np.max(heatmap[k])
            if max_val > 0.1:
                heatmap[k] /= max_val

    # 5. [FIX] 最后转置回 (H, W, C) 格式以匹配后续训练管线
    return heatmap.transpose(1, 2, 0)


def process_data(image_dir, json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])

    # 记录置信度用于 Loss Gating
    confidence_record = {}

    print(f"处理中... (共 {len(img_files)} 帧)")

    for img_file in tqdm(img_files):
        basename = os.path.splitext(img_file)[0]
        json_file = f"{basename}.json"
        json_path = os.path.join(json_dir, json_file)

        if not os.path.exists(json_path):
            continue

        # 读取图片尺寸
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        H, W = img.shape[:2]

        # 读取 JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        try:
            preds = data[0]['predictions']['predictions']
            if not preds: continue

            # 取置信度最高的检测框
            best_pred = max(preds, key=lambda x: x['confidence'])
            kpts = best_pred['keypoints']

            # 获取 BBox 尺寸 (用于计算茎粗)
            bbox_w = best_pred.get('width', 100)
            bbox_h = best_pred.get('height', 100)

        except Exception as e:
            print(f"Skipping {json_file}: {e}")
            continue

        # 生成增强热力图
        hm = generate_adaptive_heatmap(H, W, kpts, bbox_w, bbox_h, sigma=20)

        # 保存 .npy
        save_path = os.path.join(output_dir, basename + '.npy')
        np.save(save_path, hm)

        # 记录置信度 (取关键点的最大置信度或平均值，这里简化为关键点存在即置信)
        conf_vec = [0.0] * NUM_CLASSES
        for kp in kpts:
            if kp['class'] in CLASS_MAP:
                idx = CLASS_MAP[kp['class']]
                conf_vec[idx] = kp['confidence']
        confidence_record[basename] = conf_vec

    # 保存置信度文件
    conf_save_path = os.path.join(os.path.dirname(output_dir), 'confidence_video2_stem.json')
    with open(conf_save_path, 'w') as f:
        json.dump(confidence_record, f, indent=2)

    print(f"完成！置信度文件已保存至: {conf_save_path}")


# --- 配置路径 ---
# 请根据您的实际环境修改
IMAGE_DIR = r'D:\PythonProject\PythonProject\data\video_data\frames\video2_frame'
JSON_DIR = r'D:\PythonProject\PythonProject\data\video_data\detect_results\video2_frame_results'
OUTPUT_DIR = r'D:\PythonProject\PythonProject\data\heatmaps\heatmap_video2_stem'

if __name__ == "__main__":
    process_data(IMAGE_DIR, JSON_DIR, OUTPUT_DIR)