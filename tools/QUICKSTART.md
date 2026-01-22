# Quick Start Guide - Semantic Injection Tool

## 30-Second Setup

### 1. Install Dependencies
```bash
pip install numpy torch opencv-python plyfile tqdm
```

### 2. Prepare Your Data

Ensure you have:
- ✅ Dense point cloud (`.ply` file from DA3/COLMAP)
- ✅ Semantic heatmaps (`.npy` files, one per frame)
- ✅ COLMAP sparse data (`cameras.bin` and `images.bin`)

### 3. Run Injection

**Option A: Command Line (Fastest)**
```bash
python tools/inject_semantics.py \
    --dense_pcd_path "path/to/points3D.ply" \
    --heatmap_dir "path/to/heatmaps" \
    --poses_path "path/to/sparse/0" \
    --output_dir "path/to/output"
```

**Option B: Python Script**
```python
from tools.inject_semantics import Config, inject_semantics

cfg = Config()
cfg.dense_pcd_path = "path/to/points3D.ply"
cfg.heatmap_dir = "path/to/heatmaps"
cfg.poses_path = "path/to/sparse/0"
cfg.output_dir = "path/to/output"

inject_semantics(cfg)
```

### 4. Use Output

The output file `points3D_semantics.pt` contains:
```python
{
    'semantics': torch.Tensor,  # Shape: [N, 4] for U, N, D, P classes
    'num_points': int,
    'num_classes': int
}
```

Load in your 3DGS training:
```python
semantic_data = torch.load('path/to/points3D_semantics.pt')
gaussians.semantics.data = semantic_data['semantics']
```

## Troubleshooting

**Problem**: "Heatmap not found" warnings
- **Solution**: Ensure heatmap filenames match COLMAP image names (e.g., `frame_001.npy`)

**Problem**: All points marked as invisible
- **Solution**: Check coordinate system consistency between point cloud and poses

**Problem**: Out of memory
- **Solution**: Reduce `batch_size_points` in Config or process smaller point clouds

## Customization

### Adjust Spatial Robustness
```bash
# More robust to jitter (use if 2D detection is noisy)
python tools/inject_semantics.py ... --spatial_kernel_size 7

# Less robust (use if 2D detection is accurate)
python tools/inject_semantics.py ... --spatial_kernel_size 3
```

### Adjust Confidence Threshold
```bash
# More conservative (more points labeled as background)
python tools/inject_semantics.py ... --confidence_threshold 0.5

# Less conservative (more points keep their labels)
python tools/inject_semantics.py ... --confidence_threshold 0.2
```

## Testing

Test with synthetic data:
```bash
python tools/test_inject_semantics.py
```

## Full Documentation

See `tools/README.md` for detailed documentation.

## Example Workflow

```bash
# 1. Generate heatmaps from YOLO detections
python preprocess/generate_heatmaps.py

# 2. Run COLMAP sparse reconstruction (if not done)
colmap mapper ...

# 3. Inject semantics into point cloud
python tools/inject_semantics.py \
    --dense_pcd_path "data/points3D.ply" \
    --heatmap_dir "data/heatmaps" \
    --poses_path "data/sparse/0" \
    --output_dir "data/output"

# 4. Train 3DGS with semantic initialization
python s3dgs/train.py \
    --semantics_path "data/output/points3D_semantics.pt"
```

## Expected Output

```
============================================================
Semantic Injection Pipeline
============================================================
Loading point cloud from: data/points3D.ply
Loaded 1234567 points
Loading camera data from: data/sparse/0
Loaded 1 cameras, 50 images
Found 50 heatmap files

Processing views...
Injecting semantics: 100%|████████████| 50/50 [01:23<00:00,  0.60it/s]

Fusing semantics from 50 views...

Saved semantic logits to: data/output/points3D_semantics.pt
Shape: torch.Size([1234567, 4])
============================================================

Semantic Statistics:
  U: mean=0.0234, max=0.9876
  N: mean=0.0456, max=0.9234
  D: mean=0.0198, max=0.9123
  P: mean=0.0123, max=0.8765

Done!
```

## Support

For issues or questions:
1. Check `tools/README.md` for detailed documentation
2. Review `tools/IMPLEMENTATION_SUMMARY.md` for algorithm details
3. Run `tools/test_inject_semantics.py` to verify installation
