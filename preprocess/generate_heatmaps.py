import os
import json
import numpy as np
import cv2
from tqdm import tqdm

# --- 科研级配置参数 ---
# 1. 关键点语义映射：必须固定顺序，训练时 Channel 0 永远代表 U
CLASS_MAP = {
    'U': 0,  # Up node (主茎上部)
    'N': 1,  # Node (节点中心)
    'D': 2,  # Down node (主茎下部)
    'P': 3  # Peduncle (果梗弯曲点)
}
NUM_CLASSES = 4
SIGMA = 20  # 高斯核半径，针对 1920x1088 分辨率，20px 是比较稳健的值 (约 1%-2%)


def generate_gaussian_heatmap(height, width, keypoints, sigma):
    """
    生成多通道高斯热力图
    Output shape: [Height, Width, NUM_CLASSES]
    """
    heatmap = np.zeros((height, width, NUM_CLASSES), dtype=np.float32)

    # 以此为中心生成网格，优化速度
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    for kp in keypoints:
        cls_name = kp['class']
        if cls_name not in CLASS_MAP:
            continue

        idx = CLASS_MAP[cls_name]
        x, y = kp['x'], kp['y']
        confidence = kp['confidence']

        # 严谨检查：只有置信度高且在图内的点才生成热力图
        if confidence > 0.5 and 0 <= x < width and 0 <= y < height:
            # 计算距离平方
            dist_sq = (xx - x) ** 2 + (yy - y) ** 2
            # 生成高斯分布
            heatmap[:, :, idx] = np.exp(-dist_sq / (2 * sigma ** 2))

    return heatmap


def process_data(image_dir, json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件列表
    img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])

    confidence_record = {}

    print(f"开始处理 {len(img_files)} 帧数据...")

    for img_file in tqdm(img_files):
        # 假设 json 文件名与图片一致 (只是后缀不同)
        basename = os.path.splitext(img_file)[0]
        # 注意：这里需要根据你实际的文件名匹配规则修改
        # 如果图片是 LS_...jpg，JSON 是 frame_...json，你需要手动建立映射列表
        json_file = f"{basename}.json"
        json_path = os.path.join(json_dir, json_file)

        if not os.path.exists(json_path):
            print(f"[Warning] 缺少 JSON 文件: {json_path}，跳过该帧")
            continue

        # 1. 读取图像获取真实分辨率 (双重验证)
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        # 2. 读取 JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 解析你的特定 JSON 结构
        # 结构: List -> Item -> "predictions" -> "predictions" List -> Item -> "keypoints"
        try:
            # 假设每张图只有一个主要目标 (DeLeaving)
            # 如果有多个，需要根据 Rank-Aware 逻辑筛选，这里取置信度最高的
            preds = data[0]['predictions']['predictions']
            if not preds:
                kpts = []
            else:
                # 取 detection confidence 最高的那个检测框
                best_pred = max(preds, key=lambda x: x['confidence'])
                kpts = best_pred['keypoints']
        except (IndexError, KeyError) as e:
            print(f"[Error] JSON 解析失败 {json_file}: {e}")
            kpts = []

        # 3. 生成热力图
        hm = generate_gaussian_heatmap(H, W, kpts, sigma=SIGMA)

        # 4. 保存为 .npy (Float32 精度)
        save_path = os.path.join(output_dir, basename + '.npy')
        np.save(save_path, hm)

        # 5. 记录置信度 (用于 Loss Gating)
        # 初始化为 0
        conf_vec = [0.0] * NUM_CLASSES
        for kp in kpts:
            if kp['class'] in CLASS_MAP:
                idx = CLASS_MAP[kp['class']]
                conf_vec[idx] = kp['confidence']

        confidence_record[basename] = conf_vec

    # 保存置信度字典
    with open('gt_heatmaps/confidence_video2.json', 'w') as f:
        json.dump(confidence_record, f, indent=2)

    print("数据预处理完成！")


# --- 运行配置 ---
# 请修改为你的实际路径
IMAGE_DIR = r'D:\PythonProject\PythonProject\video_data\frames\video2_frame'
JSON_DIR = r'D:\PythonProject\PythonProject\video_data\detect_results\video2_frame_results'
OUTPUT_DIR = './gt_heatmaps/heatmap_video2'

if __name__ == "__main__":
    process_data(IMAGE_DIR, JSON_DIR, OUTPUT_DIR)