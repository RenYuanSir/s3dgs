# Semantic Injection for 3D Gaussian Splatting

## Overview

This tool projects 3D dense point clouds onto 2D YOLO semantic heatmaps across multiple frames to "lift" 2D semantics into 3D space using robust multi-view fusion.

## Algorithm: Max-Pooling Bayesian Multi-view Fusion

### Key Innovations

1. **Spatial Robustness**: Instead of naive point projection, samples the maximum value in a K×K window (default 5×5) around each projected point using `cv2.dilate` for efficiency.

2. **Depth Visibility**: Simple z-buffer check to avoid projecting through walls/occlusions.

3. **Bayesian Fusion**: Accumulates probability vectors from all valid views and computes weighted average.

4. **Gating**: Low-confidence points (max probability < threshold) are treated as background.

## Usage

### Method 1: Modify Config Class

Edit the `Config` class at the top of `inject_semantics.py`:

```python
class Config:
    # Input paths
    dense_pcd_path: str = r"path/to/points3D.ply"
    heatmap_dir: str = r"path/to/heatmaps"
    poses_path: str = r"path/to/sparse/0"  # COLMAP sparse directory

    # Output path
    output_dir: str = r"path/to/output"
```

Then run:
```bash
python tools/inject_semantics.py
```

### Method 2: Command-line Arguments

```bash
python tools/inject_semantics.py \
    --dense_pcd_path "path/to/points3D.ply" \
    --heatmap_dir "path/to/heatmaps" \
    --poses_path "path/to/sparse/0" \
    --output_dir "path/to/output" \
    --num_classes 4 \
    --spatial_kernel_size 5 \
    --confidence_threshold 0.3
```

## Input Requirements

### 1. Dense Point Cloud (`.ply`)
- Format: Standard PLY format with vertices
- Fields: `x`, `y`, `z` (required), `red`, `green`, `blue` (optional)
- Source: DA3 or COLMAP dense reconstruction

### 2. Semantic Heatmaps (`.npy`)
- Shape: `[H, W, Num_Classes]`
- Format: NumPy arrays
- Values: Probability distributions in range [0, 1]
- Filenames: Must match COLMAP image names (e.g., `frame_001.npy`)

### 3. Camera Poses (COLMAP format)
- Directory containing:
  - `cameras.bin`: Camera intrinsics
  - `images.bin`: Camera extrinsics (poses)
- Coordinate system: World-to-Camera transformation

## Output

### Semantic Initialization File (`.pt`)
- Dictionary containing:
  - `'semantics'`: `torch.Tensor` of shape `[N, Num_Classes]`
  - `'num_points'`: Number of points
  - `'num_classes'`: Number of semantic classes
- Format: Logits (can be converted to probabilities via sigmoid)

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | 4 | Number of semantic classes (U, N, D, P) |
| `spatial_kernel_size` | int | 5 | K×K window for max-pooling robustness |
| `confidence_threshold` | float | 0.3 | Threshold for background gating |
| `depth_tolerance` | float | 0.1 | Relative tolerance for depth checking (10%) |
| `batch_size_points` | int | 100000 | Batch size for memory efficiency |

## Coordinate System

- **COLMAP Convention**:
  - World: X-right, Y-up, Z-backward (OpenGL-style)
  - Camera: X-right, Y-down, Z-forward (OpenCV-style)
- Transformation: `X_cam = R @ X_world + t`

## Algorithm Workflow

1. **Load Data**
   - Load point cloud from PLY file
   - Parse camera poses and intrinsics from COLMAP
   - Load all heatmap files

2. **Multi-view Projection** (for each view)
   - Project 3D points to camera image plane
   - Check depth visibility (z-buffer)
   - Dilate heatmap for spatial robustness
   - Sample semantics at projected pixels

3. **Bayesian Fusion**
   - Accumulate weighted probabilities from all views
   - Compute average probability per point
   - Apply gating for low-confidence points

4. **Save Results**
   - Convert probabilities to logits
   - Save as PyTorch tensor for 3DGS initialization

## Example

```python
from tools.inject_semantics import Config, inject_semantics

# Configure
cfg = Config()
cfg.dense_pcd_path = "data/points3D.ply"
cfg.heatmap_dir = "data/heatmaps"
cfg.poses_path = "data/sparse/0"
cfg.output_dir = "data/output"
cfg.num_classes = 4
cfg.spatial_kernel_size = 5
cfg.confidence_threshold = 0.3

# Run injection
inject_semantics(cfg)
```

## Dependencies

- `numpy`
- `torch`
- `opencv-python` (cv2)
- `plyfile`
- `tqdm`

## Notes

- The script uses batched processing for memory efficiency with large point clouds
- Depth filtering can be disabled by setting `enable_depth_filtering = False`
- The dilation operation provides efficient K×K max-pooling without slow Python loops
- All coordinate transformations follow COLMAP conventions

## Troubleshooting

**Issue**: "Heatmap not found" warnings
- **Solution**: Ensure heatmap filenames match COLMAP image names (without extension)

**Issue**: "Points are all invisible"
- **Solution**: Check coordinate system consistency between point cloud and poses

**Issue**: Out of memory errors
- **Solution**: Reduce `batch_size_points` in Config

**Issue**: Poor semantic quality
- **Solution**: Increase `spatial_kernel_size` for more robustness, or adjust `confidence_threshold`
