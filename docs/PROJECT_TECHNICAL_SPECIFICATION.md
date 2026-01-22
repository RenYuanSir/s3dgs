# Semantic 3D Gaussian Splatting (S-3DGS) for Tomato Plant Phenotype Analysis

**Project Type**: Semantic-Guided 3D Scene Reconstruction
**Core Technology**: 3D Gaussian Splatting (3DGS) with Semantic Supervision
**Library**: gsplat v1.5.3
**Application Domain**: Agricultural Phenotyping (Tomato Plant Architecture)

---

## 1. System Overview

### 1.1 Objective

Reconstruct 3D geometry and semantic structure of tomato plants from multi-view RGB images, enabling:
- High-fidelity novel view synthesis
- Fine-grained stem structure reconstruction (3-5mm diameter)
- Semantic segmentation of plant organs (Up Node, Node, Down Node, Peduncle)

### 1.2 Technical Innovation

**Semantic 3DGS**: Augments standard 3DGS with semantic supervision through dual-pass rendering:
- Pass 1: Standard RGB reconstruction using Spherical Harmonics (SH)
- Pass 2: Semantic probability field rendering using skeleton-based heatmaps

**Skeleton Heatmap Supervision**: Continuous line-based supervision replacing isolated point supervision to eliminate negative learning in stem regions.

---

## 2. Data Pipeline

### 2.1 Data Flow Diagram

```
Raw Video → Extract Frames → Keypoint Detection → Skeleton Heatmaps → COLMAP Reconstruction → Training
     ↓              ↓                ↓                   ↓                    ↓             ↓
  .mp4         .jpg frames      Roboflow YOLO        .npy + .json        .bin/.ply     3DGS Model
```

### 2.2 Preprocessing Stage

#### 2.2.1 Frame Extraction (`preprocess/extract_frames.py`)

**Input**: Video file (.mp4)
**Output**: RGB image sequence (.jpg)

**Parameters**:
- Interval: 0.1 seconds (10 fps)
- Naming: `frame_{timestamp:07.3f}.jpg`

**Implementation**:
```python
extract_frames(video_path, output_folder, interval=0.1)
```

#### 2.2.2 Keypoint Detection (`preprocess/yolo_inference.py`)

**Input**: RGB image frames
**Output**: Detection JSON with 4 keypoints per plant

**Detection Framework**: Roboflow Inference SDK
- **Workspace**: smartagrizust
- **Workflow ID**: custom-workflow-2
- **API Endpoint**: http://localhost:9001

**Keypoint Schema**:
```json
{
  "predictions": {
    "predictions": [{
      "confidence": 0.95,
      "keypoints": [
        {"class": "U", "x": 100, "y": 200, "confidence": 0.9},
        {"class": "N", "x": 150, "y": 250, "confidence": 0.8},
        {"class": "D", "x": 120, "y": 300, "confidence": 0.85},
        {"class": "P", "x": 180, "y": 280, "confidence": 0.75}
      ],
      "width": 200,
      "height": 400
    }]
  }
}
```

**Semantic Classes**:
- **U (Up Node)**: Upper stem section
- **N (Node)**: Central node/hub
- **D (Down Node)**: Lower stem section
- **P (Peduncle)**: Fruit stem/branch

#### 2.2.3 Skeleton Heatmap Generation (`preprocess/generate_heatmaps_stem.py`)

**Input**: Detection JSON + Image dimensions
**Output**: Multi-channel heatmap [H, W, 4] (.npy)

**Key Innovation**: Topology-aware skeleton connectivity

**Skeleton Rules**:
```python
SKELETON_RULES = [
    ('U', 'N', 0, 'w', 0.40),  # U→N: channel 0, 40% bbox width
    ('N', 'D', 2, 'w', 0.40),  # N→D: channel 2, 40% bbox width
    ('N', 'P', 3, 'h', 0.15)   # N→P: channel 3, 15% bbox height
]
```

**Algorithm**:
1. **Channel-First Initialization**: `heatmap[C, H, W]` for OpenCV compatibility
2. **Skeleton Drawing**: `cv2.line()` with adaptive thickness
3. **Keypoint Enhancement**: Gaussian overlay (σ=20 pixels)
4. **Gaussian Blur**: Global smoothing (kernel size = 2σ)
5. **Normalization**: Per-channel max normalization
6. **Transpose**: Output [H, W, C] format

**Memory Layout Fix**:
- **Problem**: `heatmap[:,:,c]` creates non-contiguous views
- **Solution**: Use `[C, H, W]` layout, direct `heatmap[c]` access, final transpose

#### 2.2.4 COLMAP Structure from Motion

**Input**: RGB image frames
**Output**: Camera poses + 3D point cloud

**Files Generated**:
```
sparse/0/
├── cameras.bin    # Intrinsics (fx, fy, cx, cy)
├── images.bin     # Exinsics (quaternion + translation)
└── points3D.ply   # Initial point cloud
```

**Binary Format Parsers** (`s3dgs/dataset.py`):
- `parse_cameras_bin()`: Decodes camera intrinsics
- `parse_images_bin()`: Decodes World-to-Camera poses

**Coordinate Convention**:
- X_cam = R * X_world + T
- OpenCV convention: Right-Down-Forward

### 2.3 Dataset Structure

```
data/
├── video_data/
│   ├── video/                          # Source videos
│   ├── frames/videoN_frame/            # RGB frames (.jpg)
│   ├── colmap_data/videoN_output_ply/
│   │   └── sparse/0/
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       └── points3D.ply
│   └── detect_results/videoN_frame_results/  # Keypoint JSONs
└── heatmaps/
    ├── heatmap_videoN_stem/            # Skeleton heatmaps (.npy)
    └── confidence_videoN.json          # Per-class confidence
```

---

## 3. Model Architecture

### 3.1 SemanticGaussianModel (`s3dgs/model.py`)

**Inheritance**: `torch.nn.Module`
**Parameter Storage**: `nn.ParameterDict` (gsplat v1.5.3 compatibility)

#### 3.1.1 Parameter Definition

| Parameter Key | Shape | Description | Activation |
|--------------|-------|-------------|------------|
| `means` | [N, 3] | Gaussian centers | Identity |
| `scales` | [N, 3] | Log-space scales | `exp()` |
| `quats` | [N, 4] | Rotation quaternions [w,x,y,z] | `normalize()` |
| `opacities` | [N, 1] | Logit-space opacity | `sigmoid()` |
| `sh0` | [N, 1, 3] | SH DC component | Identity |
| `shN` | [N, K-1, 3] | SH higher-order | Identity |
| `semantic` | [N, 4] | Semantic logits | `softmax()` |

**Total Parameters per Gaussian**:
- Geometry: 3 + 3 + 4 + 1 = 11
- Appearance: 3 × (1 + 15) = 48 (degree 3 SH)
- Semantics: 4
- **Total**: 63 values

#### 3.1.2 Initialization Strategy

**From COLMAP Point Cloud** (`create_from_pcd()`):

1. **Position**: Direct from PLY XYZ
2. **Scale**: KNN distance (k=4) mean → log-space
3. **Rotation**: Unit quaternions [1, 0, 0, 0]
4. **Opacity**: `inverse_sigmoid(0.1) ≈ -2.197`
5. **Color**: RGB → SH DC via `(RGB - 0.5) / 0.28209`
6. **Semantics**: `N(0, 0.01)` random noise (break symmetry)

**Code**:
```python
# Scale computation
kdtree = KDTree(xyz)
distances, _ = kdtree.query(xyz, k=4)
mean_distances = np.mean(distances[:, 1:], axis=1)
scales = np.tile(np.log(mean_distances)[:, np.newaxis], (1, 3))

# Semantic initialization
semantic_init = torch.randn(num_points, num_classes) * 0.01
```

#### 3.1.3 Property Accessors

```python
@property
def get_xyz(self):
    return self.params['means']

@property
def get_scaling(self):
    return torch.exp(self.params['scales'])

@property
def get_opacity(self):
    return torch.sigmoid(self.params['opacities'])

@property
def get_semantic(self):
    return torch.softmax(self.params['semantic'], dim=1)
```

### 3.2 Dual-Pass Rendering (`s3dgs/train.py`)

#### 3.2.1 Pass 1: RGB Rendering

**Input**: SH features [N, K, 3] where K = (sh_degree + 1)²
**Output**: Rendered RGB [H, W, 3]

**gsplat API**:
```python
rgb, alpha, meta = rasterization(
    means=means,          # [N, 3]
    quats=quats,          # [N, 4]
    scales=scales,        # [N, 3] (pre-exp)
    opacities=opacities,  # [N] (post-sigmoid)
    colors=colors_sh,     # [1, N, K, 3]
    viewmats=viewmat[None],  # [1, 4, 4]
    Ks=K[None],           # [1, 3, 3]
    width=width,
    height=height,
    sh_degree=sh_degree,
    packed=True,
    absgrad=True  # AbsGS: accumulate absolute gradients
)
```

**Geometry Forwarding**: `meta['means2d']` retained for Pass 2

#### 3.2.2 Pass 2: Semantic Rendering

**Challenge**: gsplat rasterizer expects 3-channel colors, but semantic has 4 classes.

**Solution**: Channel Multiplexing
1. Pad [N, 4] → [N, 6]
2. Split into two batches:
   - Batch 1: [U, N, D] as RGB channels
   - Batch 2: [P, 0, 0] as RGB channels
3. Render separately with **same geometry**
4. Concatenate results

**Implementation**:
```python
sem_probs = model.get_semantic  # [N, 4]

# Pad to 6 channels
sem_padded = torch.zeros(N, 6)
sem_padded[:, :3] = sem_probs[:, :3]  # U, N, D
sem_padded[:, 3] = sem_probs[:, 3]    # P

# Render batches
sem_render1 = rasterization(..., colors=sem_padded[:, :3][None])
sem_render2 = rasterization(..., colors=sem_padded[:, 3:6][None])

# Concatenate
sem_map = torch.cat([sem_render1, sem_render2[..., 0:1]], dim=-1)
```

### 3.3 Loss Functions

#### 3.3.1 RGB Reconstruction Loss

```python
L1_loss = |pred_rgb - gt_rgb|.mean()
```

#### 3.3.2 Semantic Loss with Confidence Gating

**Formula**:
```
loss_sem = MSE(pred_sem, gt_heatmap) × (1 + fg_weight × fg_mask) × valid_mask
```

**Components**:
1. **MSE Loss**: `(pred - gt)²`
2. **Foreground Weighting**: 20× weight for pixels where `gt > 0.1`
3. **Confidence Gating**: Mask out classes with `confidence < 0.5`

**Implementation**:
```python
def semantic_loss_with_gating(pred_sem, gt_heatmap, confidence, fg_weight=20.0):
    loss = (pred_sem - gt_heatmap) ** 2
    fg_mask = (gt_heatmap > 0.1).float()
    loss = loss * (1.0 + fg_weight * fg_mask)
    valid_mask = (confidence > 0.5).float().view(1, 1, K)
    return (loss * valid_mask).mean()
```

#### 3.3.3 Scheduled Semantic Learning

**Warm-up Strategy**: "Geometry First"
- Iterations 0-4000: `lambda_sem = 0` (RGB only)
- Iterations 4000+: `lambda_sem = 0.05` (Enable semantics)

**Rationale**: Prevents semantic loss from pruning stem geometry during early training.

### 3.4 Densification Strategy

**Framework**: `gsplat.strategy.DefaultStrategy` (v1.5.3)

**Configuration**:
```python
strategy = DefaultStrategy(
    refine_every=100,        # Densify every 100 iterations
    grow_grad2d=5e-5,        # 2D gradient threshold for split/clone
    grow_scale3d=0.005,      # 3D scale threshold for split
    prune_opa=0.005,         # Opacity threshold for pruning
    prune_scale3d=3.0,       # Max scale before pruning
    refine_start_iter=500,   # Start densification at iter 500
    refine_stop_iter=15000,  # Stop densification at iter 15000
    reset_every=3000,        # Reset optimizer every 3000 iters
    absgrad=True,            # Use absolute gradients (AbsGS)
    verbose=True
)
```

**Densification Logic** (handled by DefaultStrategy):
- **Clone**: High gradient + small scale → duplicate
- **Split**: High gradient + large scale → divide into N smaller Gaussians
- **Prune**: Low opacity → remove

**AbsGS (Absolute Gradients)**:
- Accumulates absolute gradient magnitudes instead of averages
- Provides stronger signal for sparse regions (stems)
- Enabled via `absgrad=True` in rasterization

---

## 4. Training Pipeline

### 4.1 Dataset Loader (`s3dgs/dataset.py`)

**Class**: `TomatoDataset`
**Base**: `torch.utils.data.Dataset`

**Initialization**:
```python
dataset = TomatoDataset(
    colmap_dir="path/to/sparse/0",
    images_dir="path/to/frames",
    heatmap_dir="path/to/heatmaps",
    confidence_path="path/to/confidence.json"
)
```

**Data Caching**: All frames loaded into RAM during `__init__` (~100 frames, ~2GB)

**`__getitem__` Output**:
```python
{
    'image': Tensor[H, W, 3],      # RGB in [0, 1]
    'heatmap': Tensor[H, W, 4],    # Semantic in [0, 1]
    'confidence': Tensor[4],       # Per-class confidence
    'viewmat': Tensor[4, 4],       # World-to-Camera matrix
    'K': Tensor[3, 3],             # Intrinsics
    'height': int,
    'width': int,
    'image_id': int
}
```

**Filename Matching**:
- COLMAP `image_name`: `frame_000.333.jpg`
- Heatmap: `frame_000.333.npy`
- JSON: `frame_000.333.json`

### 4.2 Training Loop (`s3dgs/train.py`)

#### 4.2.1 Initialization

```python
# Model
model = SemanticGaussianModel(sh_degree=3, num_classes=4)
model.create_from_pcd("points3D.ply", spatial_lr_scale=1.0)
model = model.to(device)

# Dataset
dataset = TomatoDataset(...)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Optimizers (one per parameter group)
optimizers = {
    'means': Adam(lr=0.00016),
    'scales': Adam(lr=0.005),
    'quats': Adam(lr=0.001),
    'opacities': Adam(lr=0.05),
    'sh0': Adam(lr=0.0025),
    'shN': Adam(lr=0.000125),
    'semantic': Adam(lr=0.01)
}

# Strategy
strategy = DefaultStrategy(...)
strategy_state = strategy.initialize_state(scene_scale=5.0)
```

#### 4.2.2 Iteration Loop

```python
for iteration in range(num_iterations):
    # 1. Load batch
    data = next(dataloader)

    # 2. Dual-pass render
    renders = render_dual_pass(model, viewmat, K, width, height)
    pred_rgb = renders['rgb']
    pred_sem = renders['semantic']

    # 3. Compute loss
    loss_rgb = l1_loss(pred_rgb, gt_image)
    loss_sem = semantic_loss_with_gating(pred_sem, gt_heatmap, confidence)

    # 4. Schedule lambda_sem
    if iteration < warmup_iterations:
        lambda_sem = 0.0
    else:
        lambda_sem = 0.05

    total_loss = loss_rgb + lambda_sem * loss_sem

    # 5. Pre-backward (prepare for densification)
    strategy.step_pre_backward(params, optimizers, state, iteration, info)

    # 6. Backward
    total_loss.backward()

    # 7. Post-backward (densification)
    strategy.step_post_backward(params, optimizers, state, iteration, info, packed=True)

    # 8. Optimizer step
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad()

    # 9. Logging & checkpointing
    if iteration % log_every == 0:
        print(f"Iter {iteration}: loss={total_loss.item():.4f}, GS={model.num_gaussians()}")
```

### 4.3 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Training** | | |
| `num_iterations` | 30,000 | Total training iterations |
| `warmup_iterations` | 4,000 | Geometry-only warm-up |
| `lambda_sem` | 0.05 | Semantic loss weight |
| `lr_scale` | 1.0 | Global LR multiplier |
| `spatial_lr_scale` | 1.0 | Position LR multiplier |
| **Densification** | | |
| `grow_grad2d` | 5e-5 | 2D gradient threshold |
| `grow_scale3d` | 0.005 | 3D scale threshold |
| `refine_every` | 100 | Densification interval |
| `refine_start_iter` | 500 | Start densification |
| `refine_stop_iter` | 15,000 | Stop densification |
| **Loss Weights** | | |
| `fg_weight` | 20.0 | Foreground pixel multiplier |
| **Resolution** | | |
| `resolution_scale` | 0.75 | Image downsample factor |

### 4.4 Output Format

**Checkpoints**: Saved every 1000 iterations
```python
{
    'iteration': int,
    'model': model.state_dict(),
    'optimizers': {k: v.state_dict() for k, v in optimizers.items()}
}
```

**Logging**:
- Loss values (L1, Semantic, Total)
- Gaussian count
- GPU memory
- Timing metrics

---

## 5. Technical Challenges & Solutions

### 5.1 Memory Layout Incompatibility (Step 15)

**Problem**:
```
cv2.error: Layout of the output array img is incompatible with cv::Mat
```

**Root Cause**: `heatmap[:,:,c]` creates non-contiguous memory views

**Solution**:
1. Initialize as `[C, H, W]` (Channel-First)
2. Direct access `heatmap[c]` (contiguous)
3. Final transpose to `[H, W, C]`

### 5.2 Insufficient Densification (Step 12)

**Problem**: Only 5% Gaussian growth (16k → 17k)

**Root Cause**: Gradient threshold `1e-6` too high for actual gradients (~1e-8)

**Solution**:
- Lower `grow_grad2d` from `1e-6` → `5e-5`
- Increase `lambda_sem` from `0.05` → `0.2`
- Enable `absgrad=True` (AbsGS)

### 5.3 Negative Learning in Stems (Step 15-17)

**Problem**: Point supervision treats stems as background

**Root Cause**: Isolated Gaussian blobs (point supervision) → semantic loss prunes stem regions

**Solution**:
- Replace point heatmaps with **skeleton heatmaps**
- Continuous lines connecting U-N-D-P keypoints
- Adaptive thickness (main stem: 40% bbox, peduncle: 15% bbox)

### 5.4 Performance Bottleneck (Step 12)

**Problem**: 6 hours for 16k iterations (~1 FPS)

**Root Cause**: CPU-GPU sync every iteration via `.item()` calls

**Solution**:
- Move gradient checks inside `if iteration % log_every == 0`
- Only synchronize when logging

---

## 6. File Structure

```
PythonProject/
├── preprocess/
│   ├── extract_frames.py          # Video → frames
│   ├── yolo_inference.py          # Keypoint detection
│   ├── generate_heatmaps_stem.py  # Skeleton heatmaps
│   └── convert_bin_to_ply.py      # COLMAP bin → PLY
├── s3dgs/
│   ├── __init__.py
│   ├── model.py                   # SemanticGaussianModel
│   ├── dataset.py                 # TomatoDataset + parsers
│   └── train.py                   # Training loop
├── data/
│   ├── video_data/
│   │   ├── frames/                # RGB images
│   │   ├── colmap_data/           # COLMAP outputs
│   │   └── detect_results/        # Keypoint JSONs
│   └── heatmaps/
│       ├── heatmap_videoN_stem/   # Skeleton heatmaps
│       └── confidence_videoN.json
├── output/
│   └── video2_skeleton_supervision/  # Training results
├── plan.md                         # Development history
└── vis_heatmap_overlay.py          # Visualization tool
```

---

## 7. Evaluation Metrics

### 7.1 Reconstruction Quality

**Metrics**:
- L1 Loss (RGB reconstruction)
- SSIM (structural similarity)
- PSNR (peak signal-to-noise ratio)
- LPIPS (perceptual similarity)

### 7.2 Semantic Segmentation

**Metrics**:
- mIoU (mean Intersection-over-Union)
- Per-class IoU (U, N, D, P)
- Pixel accuracy

### 7.3 Densification Effectiveness

**Metrics**:
- Gaussian count progression
- Densification rate (splits + clones per 100 iters)
- Prune rate

---

## 8. Dependencies

### 8.1 Core Libraries

```bash
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# 3D Gaussian Splatting
gsplat==1.5.3

# Computer Vision
opencv-python>=4.8.0
pillow>=10.0.0

# Data Processing
numpy>=1.24.0
scipy>=1.11.0
plyfile>=0.8.0

# Utilities
tqdm>=4.66.0
pyyaml>=6.0
```

### 8.2 Optional Dependencies

```bash
# Visualization
matplotlib>=3.8.0

# Tensorboard
tensorboard>=2.14.0

# Metrics
torchmetrics>=1.0.0
```

---

## 9. Known Limitations

### 9.1 Current Bottlenecks

1. **Memory Requirement**: ~100 frames × 1920×1088 RGB → ~2GB RAM
2. **Training Speed**: ~50 FPS with `resolution_scale=0.75`
3. **Densification**: Limited by gradient signal in sparse stem regions

### 9.2 Future Work

1. **Streaming DataLoader**: For larger datasets (>500 frames)
2. **Multi-GPU Training**: Distributed data parallel
3. **Real-time Rendering**: Optimize rasterization for live preview
4. **Adaptive Densification**: Region-specific threshold adjustment

---

## 10. References

### 10.1 Core Papers

- **3D Gaussian Splatting**: Kerbl et al., SIGGRAPH 2023
- **AbsGS**: Adaptive Gradient Accumulation for densification
- **Semantic 3DGS**: This work (extension)

### 10.2 Codebases

- **gsplat v1.5.3**: https://github.com/nerfstudio-project/gsplat
- **COLMAP**: https://github.com/colmap/colmap

---

**Document Version**: 1.0
**Last Updated**: 2025-01-16
**Author**: Project Technical Documentation
**Status**: Production Ready
