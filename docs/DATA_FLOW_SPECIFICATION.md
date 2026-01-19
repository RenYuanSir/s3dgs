# S-3DGS 数据流详细说明书

## 1. 完整数据流动图

```
┌─────────────┐
│  Raw Video  │ (.mp4)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Frame Extraction (extract_frames) │
│   Input: video.mp4                  │
│   Output: frame_*.jpg (10 fps)      │
└──────┬──────────────────────────────┘
       │
       ├─────────────────────────────────────┐
       │                                     │
       ▼                                     ▼
┌──────────────────┐              ┌──────────────────────┐
│ COLMAP Recon     │              │ Roboflow YOLO        │
│ (External Tool)  │              │ (yolo_inference)     │
│                  │              │                      │
│ Input: frames/   │              │ Input: frames/       │
│ Output:          │              │ Output:              │
│ - cameras.bin    │              │ - *.json (4 kpts)    │
│ - images.bin     │              │   + confidence       │
│ - points3D.ply   │              │                      │
└──────┬───────────┘              └──────┬───────────────┘
       │                                 │
       │                                 ▼
       │                    ┌────────────────────────┐
       │                    │ Skeleton Heatmap Gen   │
       │                    │ (generate_heatmaps_    │
       │                    │  stem)                 │
       │                    │                        │
       │                    │ Input:                 │
       │                    │ - *.json (kpts)        │
       │                    │ - frame dimensions    │
       │                    │ Output:                │
       │                    │ - *.npy [H,W,4]        │
       │                    └──────┬─────────────────┘
       │                           │
       ▼                           ▼
┌────────────────────────────────────────────┐
│         TomatoDataset (dataset.py)         │
│                                            │
│ Loads:                                     │
│ - cameras.bin → camera intrinsics (K)      │
│ - images.bin → World-to-Camera poses       │
│ - *.jpg → RGB images [0,1]                 │
│ - *.npy → semantic heatmaps [0,1]          │
│ - *.json → per-class confidence            │
│                                            │
│ Caches: ~2GB RAM for 100 frames            │
└──────┬─────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────┐
│     SemanticGaussianModel (model.py)       │
│                                            │
│ Initialized from points3D.ply:             │
│ - N Gaussians (N ≈ 16,000)                │
│ - Each has 63 parameters:                 │
│   • 3 position (xyz)                       │
│   • 3 scale (log-space)                    │
│   • 4 rotation (quaternion)                │
│   • 1 opacity (logit-space)                │
│   • 48 appearance (SH DC+Rest)             │
│   • 4 semantic (logits)                    │
└──────┬─────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────┐
│      Training Loop (train.py)              │
│                                            │
│ For each iteration:                        │
│ 1. Sample batch (1 frame)                  │
│ 2. Dual-pass render:                       │
│    Pass 1: RGB via SH                      │
│    Pass 2: Semantic via softmax            │
│ 3. Compute losses:                         │
│    L1(RGB) + λ_sem × MSE(Sem)              │
│ 4. Backward pass                           │
│ 5. Densify (DefaultStrategy):              │
│    - Clone high-grad small Gaussians       │
│    - Split high-grad large Gaussians       │
│    - Prune low-opacity Gaussians           │
│ 6. Optimizer step                          │
└──────┬─────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────┐
│           Trained Model                    │
│                                            │
│ Capabilities:                              │
│ - Novel view synthesis (RGB)               │
│ - Semantic segmentation (U,N,D,P)          │
│ - 3D geometry reconstruction               │
│                                            │
│ Output:                                    │
│ - Checkpoints (.pt)                        │
│ - Rendered images                          │
│ - Semantic maps                            │
└────────────────────────────────────────────┘
```

---

## 2. 数据格式详细说明

### 2.1 输入：视频文件

**格式**: MP4 (H.264编码)
**分辨率**: 1920×1088
**帧率**: ~30 fps

### 2.2 中间产物1：RGB帧

**文件**: `frame_000.333.jpg`
**格式**: JPEG, 8-bit per channel
**分辨率**: 1920×1088 (原始) 或 1440×816 (下采样×0.75)
**数值范围**: [0, 255] uint8

**Dataset处理**:
```python
image = Image.open(path).convert("RGB")  # [H, W, 3] uint8
image = np.array(image) / 255.0          # [H, W, 3] float32 [0,1]
```

### 2.3 中间产物2：关键点检测JSON

**文件**: `frame_000.333.json`

**结构**:
```json
[
  {
    "predictions": {
      "predictions": [
        {
          "confidence": 0.95,
          "x": 960, "y": 544, "width": 200, "height": 400,
          "keypoints": [
            {"class": "U", "x": 960, "y": 344, "confidence": 0.9},
            {"class": "N", "x": 960, "y": 544, "confidence": 0.8},
            {"class": "D", "x": 960, "y": 744, "confidence": 0.85},
            {"class": "P", "x": 1060, "y": 544, "confidence": 0.75}
          ]
        }
      ]
    }
  }
]
```

**字段说明**:
- `confidence`: 检测框置信度 [0,1]
- `x, y`: 检测框中心坐标
- `width, height`: 检测框尺寸（用于计算骨架粗细）
- `keypoints`: 4个关键点
  - `class`: 语义类别 (U/N/D/P)
  - `x, y`: 像素坐标
  - `confidence`: 关键点置信度

### 2.4 中间产物3：骨架热图NPY

**文件**: `frame_000.333.npy`
**格式**: Float32 numpy array
**形状**: [H, W, 4] = [1088, 1920, 4]
**数值范围**: [0, 1] (归一化后)

**通道映射**:
- Channel 0: U (Up Node) - 红色
- Channel 1: N (Node) - 绿色
- Channel 2: D (Down Node) - 蓝色
- Channel 3: P (Peduncle) - 黄色

**生成算法**:
```python
# 1. 初始化 channel-first 格式 (C, H, W)
heatmap = np.zeros((4, H, W), dtype=np.float32)

# 2. 绘制骨架线
cv2.line(heatmap[0], pt_U, pt_N, thickness=int(0.4 * bbox_w))
cv2.line(heatmap[2], pt_N, pt_D, thickness=int(0.4 * bbox_w))
cv2.line(heatmap[3], pt_N, pt_P, thickness=int(0.15 * bbox_h))

# 3. 叠加关键点高斯
for kp in keypoints:
    gaussian = exp(-((xx - kp.x)**2 + (yy - kp.y)**2) / (2 * σ**2))
    heatmap[channel_idx] = maximum(heatmap[channel_idx], gaussian)

# 4. 高斯模糊
for c in range(4):
    heatmap[c] = GaussianBlur(heatmap[c], ksize=(41,41))
    heatmap[c] /= max(heatmap[c])  # 归一化

# 5. 转置回 (H, W, C)
heatmap = heatmap.transpose(1, 2, 0)
```

### 2.5 中间产物4：COLMAP二进制文件

#### cameras.bin

**格式**: 紧凑二进制格式

**结构**:
```
uint64: num_cameras
对于每个相机:
  uint32: camera_id
  uint32: model (1=PINHOLE)
  uint64: width
  uint64: height
  double[4]: params (fx, fy, cx, cy)
```

**解析结果**:
```python
cameras = {
    camera_id: {
        'model': 1,              # PINHOLE
        'width': 1920,
        'height': 1088,
        'params': [fx, fy, cx, cy]  # 内参
    }
}
```

**内参矩阵K**:
```
K = [[fx, 0,  cx],
     [0,  fy, cy],
     [0,  0,  1 ]]
```

#### images.bin

**格式**: 紧凑二进制格式

**结构**:
```
uint64: num_images
对于每个图像:
  uint32: image_id
  double[4]: q (qw, qx, qy, qz)  # 四元数 [w,x,y,z]
  double[3]: t (tx, ty, tz)       # 平移向量
  uint32: camera_id
  char[]: image_name (null-terminated)
  uint64: num_2d_points
  point3d_id[0]...point3d_id[N-1]
```

**解析结果**:
```python
images = {
    image_id: {
        'id': 24,
        'name': "frame_000.333.jpg",
        'camera_id': 5,
        'q': [qw, qx, qy, qz],  # 世界到相机旋转
        'tvec': [tx, ty, tz]    # 世界到相机平移
    }
}
```

**World-to-Camera矩阵**:
```
R = quaternion_to_matrix(qw, qx, qy, qz)
t = tvec.reshape(3, 1)

W2C = [[R00, R01, R02, tx],
       [R10, R11, R12, ty],
       [R20, R21, R22, tz],
       [0,   0,   0,   1 ]]
```

#### points3D.ply

**格式**: ASCII PLY

**结构**:
```
ply
format ascii 1.0
element vertex 16311
property float x
property float y
property float z
property uint8 red
property uint8 green
property uint8 blue
end_header
x1 y1 z1 r1 g1 b1
x2 y2 z2 r2 g2 b2
...
```

**加载结果**:
```python
xyz = [[x1, y1, z1],
       [x2, y2, z2],
       ...]  # [N, 3] float32

rgb = [[r1/255, g1/255, b1/255],
       [r2/255, g2/255, b2/255],
       ...]  # [N, 3] float32
```

### 2.6 中间产物5：置信度文件

**文件**: `confidence_video2.json`

**结构**:
```json
{
  "frame_000.333": {
    "U": 0.9,
    "N": 0.8,
    "D": 0.85,
    "P": 0.75
  },
  "frame_001.234": {
    ...
  }
}
```

**用途**:
- 训练时对语义损失进行门控
- 只使用 `confidence > 0.5` 的类别计算损失

---

## 3. Dataset类数据流详解

### 3.1 初始化阶段 (`__init__`)

```python
def __init__(self, colmap_dir, images_dir, heatmap_dir, confidence_path):
    # 1. 解析COLMAP相机参数
    cameras = parse_cameras_bin(colmap_dir + "/cameras.bin")
    # cameras: {camera_id: {model, width, height, params}}

    # 2. 解析COLMAP图像位姿
    images_data = parse_images_bin(colmap_dir + "/images.bin")
    # images_data: [{id, name, camera_id, q, tvec}]

    # 3. 加载置信度
    with open(confidence_path) as f:
        confidence_dict = json.load(f)
    # confidence_dict: {frame_name: {U: 0.9, N: 0.8, ...}}

    # 4. 预加载所有数据到内存
    self.cached_data = []
    for img_data in images_data:
        # 读取RGB图像
        image = Image.open(images_dir + "/" + img_data['name'])
        image = np.array(image) / 255.0  # [H, W, 3]

        # 读取热图
        basename = img_data['name'].replace('.jpg', '')
        heatmap = np.load(heatmap_dir + "/" + basename + ".npy")
        # heatmap: [H, W, 4]

        # 获取置信度
        confidence = confidence_dict.get(basename, [0.25]*4)

        # 构建viewmat
        R = quaternion_to_matrix(img_data['q'])
        t = img_data['tvec'].reshape(3, 1)
        viewmat = np.vstack([np.hstack([R, t]), [[0,0,0,1]]])
        # viewmat: [4, 4]

        # 获取内参
        camera = cameras[img_data['camera_id']]
        fx, fy, cx, cy = camera['params']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # K: [3, 3]

        # 缓存
        self.cached_data.append({
            'image': torch.from_numpy(image).float(),
            'heatmap': torch.from_numpy(heatmap).float(),
            'confidence': torch.tensor(confidence).float(),
            'viewmat': torch.from_numpy(viewmat).float(),
            'K': torch.from_numpy(K).float(),
            'height': camera['height'],
            'width': camera['width'],
            'image_id': img_data['id']
        })
```

**内存占用**:
- RGB图像: 1088×1920×3×4 bytes = ~25 MB/frame
- 热图: 1088×1920×4×4 bytes = ~33 MB/frame
- 总计（100帧）: ~25 MB × 100 + ~33 MB × 100 ≈ 5.8 GB
- 实际: 2-3 GB（PyTorch tensor共享内存）

### 3.2 访问阶段 (`__getitem__`)

```python
def __getitem__(self, idx):
    data = self.cached_data[idx]

    return {
        'image': data['image'],        # [H, W, 3] in [0,1]
        'heatmap': data['heatmap'],    # [H, W, 4] in [0,1]
        'confidence': data['confidence'],  # [4]
        'viewmat': data['viewmat'],    # [4, 4] World-to-Camera
        'K': data['K'],               # [3, 3] Intrinsics
        'height': data['height'],     # int
        'width': data['width'],       # int
        'image_id': data['image_id']  # int
    }
```

**时间复杂度**: O(1) - 直接从RAM读取

---

## 4. 双通道渲染数据流

### 4.1 Pass 1: RGB渲染流程

```
Input:
  means: [N, 3]          # 高斯中心
  scales: [N, 3]         # 对数尺度
  quats: [N, 4]          # 四元数
  opacities: [N]         # 透明度（已sigmoid）
  sh_features: [N, K, 3]  # 球谐系数 K=(deg+1)²
  viewmat: [4, 4]        # 世界到相机矩阵
  K: [3, 3]              # 内参矩阵

Steps:
  1. 投影: 3D → 2D
     - means_camera = transform(means, viewmat)  # [N, 3]
     - means2d = project(means_camera, K)         # [N, 2]

  2. 光栅化
     - 计算每个高斯的2D椭圆 (conic, radius)
     - 按深度排序
     - Alpha blending

  3. 球谐着色
     - 根据 view directions 计算 SH
     - 输出每个像素的 RGB

Output:
  rgb: [H, W, 3]         # 渲染的RGB图像
  alpha: [H, W]          # 累积透明度
  meta: {
    'means2d': [N, 2],   # 2D投影坐标（保留梯度）
    'radii': [N, 2],     # 2D椭圆半径
    'conics': [N, 3],    # 2D椭圆参数
    ...
  }
```

### 4.2 Pass 2: 语义渲染流程

```
Input:
  means, scales, quats, opacities: (同Pass 1，不重新计算)
  semantic_logits: [N, 4]     # 语义logits
  viewmat, K: (同Pass 1)

Steps:
  1. 激活
     sem_probs = softmax(semantic_logits, dim=1)  # [N, 4]
     # 归一化: sum(prob) = 1 for each Gaussian

  2. 通道复用 (4通道 → 6通道)
     sem_padded = zeros(N, 6)
     sem_padded[:, 0:3] = sem_probs[:, 0:3]  # U, N, D → R, G, B
     sem_padded[:, 3] = sem_probs[:, 3]       # P → R

  3. 分批渲染
     # Batch 1: U, N, D (通道0-2)
     sem_render1 = rasterize(
       colors=sem_padded[:, 0:3][None],  # [1, N, 3]
       geometry=(同Pass 1)
     )

     # Batch 2: P (通道3 + 零填充)
     sem_render2 = rasterize(
       colors=sem_padded[:, 3:6][None],  # [1, N, 3]
       geometry=(同Pass 1)
     )

  4. 合并
     sem_map = cat([sem_render1, sem_render2[..., 0:1]], dim=-1)
     # [H, W, 4]

Output:
  semantic: [H, W, 4]     # 每个像素的语义概率分布
```

### 4.3 几何重用

**关键优化**: Pass 2 不重新计算几何投影

```python
# Pass 1 渲染
rgb, alpha, meta = rasterization(..., absgrad=True)

# 保留中间结果
meta['means2d'].retain_grad()  # 保留2D投影梯度

# Pass 2 渲染 - 使用相同的几何参数
sem_render1, _, _ = rasterization(
    ...,
    absgrad=True,  # 累积绝对梯度
    # 注意: meta['means2d'] 已经有梯度了
)
```

---

## 5. 训练循环数据流

### 5.1 单次迭代完整流程

```
┌─────────────────────────────────────────────┐
│  1. 数据加载 (DataLoader)                   │
│      batch = dataloader.next()             │
│      • 1帧图像                              │
│      • 对应的热图                           │
│      • 相机参数                             │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  2. 双通道渲染 (render_dual_pass)            │
│      • Pass 1: RGB 渲染                     │
│      • Pass 2: 语义渲染                     │
│      • 保留几何中间结果                      │
└──────────┬──────────────────────────────────┘
           │
           ├──────────────┬──────────────┐
           ▼              ▼              ▼
      pred_rgb      pred_sem     meta['means2d']
      [H,W,3]       [H,W,4]        [N,2]
           │              │              │
           ▼              ▼              │
┌─────────────────────────────────────────────┐
│  3. 损失计算                                │
│      loss_rgb = L1(pred_rgb, gt_rgb)       │
│      loss_sem = MSE(pred_sem, gt_heatmap)   │
│                  × foreground_weight         │
│                  × confidence_mask          │
│      total_loss = loss_rgb + λ × loss_sem  │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  4. 策略预反向传播                           │
│      strategy.step_pre_backward(           │
│        params, optimizers, state, iter      │
│      )                                      │
│      • 准备密集化                           │
│      • 更新状态                             │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  5. 反向传播                                │
│      total_loss.backward()                  │
│      • 计算所有参数梯度                      │
│      • meta['means2d'].grad 被累积          │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  6. 策略后反向传播                           │
│      strategy.step_post_backward(          │
│        params, optimizers, state, iter,     │
│        info, packed=True                   │
│      )                                      │
│      • 根据2D梯度决定密集化                 │
│        - grad > 5e-5 且 scale 小 → clone   │
│        - grad > 5e-5 且 scale 大 → split   │
│      • 剪枝低透明度高斯                     │
│      • 更新 optimizer 内部状态              │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  7. 优化器步进                              │
│      for opt in optimizers.values():       │
│          opt.step()                         │
│          opt.zero_grad()                    │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  8. 日志与检查点                            │
│      if iter % 100 == 0:                   │
│          print(loss, num_gaussians)        │
│      if iter % 1000 == 0:                  │
│          save_checkpoint()                 │
└─────────────────────────────────────────────┘
```

### 5.2 梯度流动示意图

```
Loss
  │
  ├─→ loss_rgb ──→ pred_rgb ──→ sh_features ──→ sh0, shN
  │                                              │
  └─→ loss_sem ──→ pred_sem ──→ sem_logits ──→ semantic

                       ↑
                       │
                Both depend on geometry
                       │
                       ├─→ means (xyz)
                       ├─→ scales
                       ├─→ quats
                       └─→ opacities

                  (Densification targets)
```

---

## 6. 密集化数据流

### 6.1 DefaultStrategy 工作流程

```
┌─────────────────────────────────────────────┐
│  Input:                                     │
│    means2d.grad: [N] abs gradient norms    │
│    scales: [N, 3]                          │
│    opacities: [N]                          │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Filter High-Gradient Gaussians            │
│    mask_grad = means2d.grad > 5e-5         │
└──────────┬──────────────────────────────────┘
           │
           ├─────────────────────┬─────────────────────┐
           ▼                     ▼                     ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   Small Scale    │   │   Large Scale    │   │   Low Opacity    │
│   clone_mask     │   │   split_mask     │   │   prune_mask     │
└──────┬───────────┘   └──────┬───────────┘   └──────┬───────────┘
       │                     │                     │
       ▼                     ▼                     ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   Clone Action   │   │   Split Action    │   │   Prune Action   │
│                  │   │                  │   │                  │
│  复制参数到新高斯  │   │  分裂成N个小高斯  │   │  删除低透明度高斯 │
│                  │   │                  │   │                  │
│  新位置 = 原位置   │   │  新位置 = 原位置  │   │  保留: ~mask     │
│  新尺度 = 原尺度   │   │    + 随机扰动    │   │                  │
│  新旋转 = 原旋转   │   │  新尺度 = 原尺度/ │   │                  │
│  新SH = 原SH       │   │    1.6          │   │                  │
│  新语义 = 原语义   │   │  新旋转 = 随机   │   │                  │
│                  │   │  新语义 = 原语义 │   │                  │
└──────────────────┘   └──────────────────┘   └──────────────────┘
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Updated Model Parameters │
              │  N → N + ΔN_clone + ΔN_split│
              │  N → N - ΔN_prune           │
              └────────────────────────────┘
```

### 6.2 密集化示例

**初始状态**:
- N = 16,000 个高斯
- 梯度分布:
  - 95%: grad < 1e-7 (背景)
  - 4%: 1e-7 < grad < 5e-5 (边缘)
  - 1%: grad > 5e-5 (茎区域)

**密集化操作** (iter 1000):
1. **Clone**: 400 个小尺度高斯 (4% × 10000 × scale_small)
   - 新增: 400 个
2. **Split**: 100 个大尺度高斯 (1% × 10000 × scale_large)
   - 分裂为: 200 个 (每个分2个)
   - 新增: 100 个
3. **Prune**: 80 个低透明度高斯 (opacity < 0.005)
   - 删除: 80 个

**净增长**: 400 + 100 - 80 = **420 个高斯**
**新总数**: 16,420 个

---

## 7. 输出数据流

### 7.1 训练输出

**目录结构**:
```
output/video2_skeleton_supervision/
├── ckpts/
│   ├── ckpt_1000.pt
│   ├── ckpt_2000.pt
│   └── ...
├── renders/
│   ├── iter_1000/
│   │   ├── frame_000.png
│   │   └── ...
│   └── iter_2000/
└── logs/
    └── train_log.txt
```

**Checkpoint 格式**:
```python
{
    'iteration': 1000,
    'model': {
        'params': {
            'means': Tensor[N, 3],
            'scales': Tensor[N, 3],
            ...
        }
    },
    'optimizers': {
        'means': {'state': ...},
        'scales': {'state': ...},
        ...
    }
}
```

### 7.2 推理输出

**渲染图像**: `frame_000.png`
- RGB: [H, W, 3] uint8
- 语义: [H, W, 4] float32 (概率分布)
- Alpha: [H, W] float32 (透明度)

**语义分割**: 可视化
- 类别映射: 0→红色, 1→绿色, 2→蓝色, 3→黄色
- Argmax: `pred_class = argmax(semantic, axis=-1)`

---

**文档版本**: 1.0
**最后更新**: 2025-01-16
