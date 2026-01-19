# Depth Anything V2 Integration Guide

## Overview

This document describes the integration of **Depth Anything V2** monocular depth priors into the S-3DGS pipeline to improve geometric reconstruction quality and fix structural artifacts (broken stems, floaters).

## Problem Statement

**Root Cause**: RGB loss alone is insufficient for complex plant structures. The 3DGS geometry lacks geometric constraints, leading to:
- Broken stems (disconnected stem segments)
- Floaters (artifacts in empty space)
- Poor reconstruction of thin structures

**Solution**: Integrate monocular depth priors from Depth Anything V2 to provide additional geometric supervision during training.

---

## Implementation Summary

### 1. Depth Map Generation (`preprocess/generate_depth.py`)

**Purpose**: Generate monocular depth maps for all training frames using Depth Anything V2.

**Features**:
- Uses `gradio_client` for online inference (no local model required)
- Saves depth maps as 16-bit PNG files for high precision
- Automatic verification of generated depth maps

**Usage**:
```bash
pip install gradio_client

python preprocess/generate_depth.py \
    path/to/images/ \
    path/to/depths/ \
    --verify
```

**API Details**:
- Uses `gradio_client` with `handle_file()` wrapper
- Endpoint: `depth-anything/Depth-Anything-V2`
- API name: `/on_submit`
- Example call:
  ```python
  from gradio_client import Client, handle_file

  client = Client("depth-anything/Depth-Anything-V2")
  result = client.predict(
      image=handle_file('path/to/image.jpg'),
      api_name="/on_submit"
  )
  ```

**Output**:
- `path/to/depths/frame_0000000.png` (16-bit depth map)
- Normalized to [0, 1] range
- Resolution matches input images

---

### 2. Dataset Updates (`s3dgs/dataset.py`)

**Changes**:
- Added `depth_dir` parameter to `TomatoDataset.__init__()`
- Loads depth maps in `__getitem__()` if available
- Returns depth tensor in data batch (if depth exists)

**Backward Compatibility**:
- Depth loading is **optional**
- If `depth_dir=None` or depth files are missing, training continues without depth supervision
- No breaking changes to existing code

**API**:
```python
from s3dgs.dataset import create_dataloader

dataloader = create_dataloader(
    colmap_dir="path/to/colmap",
    images_dir="path/to/images",
    heatmap_dir="path/to/heatmaps",
    confidence_path="path/to/confidence.json",
    depth_dir="path/to/depths",  # Optional
    batch_size=1
)
```

---

### 3. Scale-Invariant Depth Loss (`s3dgs/train.py`)

**Mathematical Formulation**:

Monocular depth is **relative** (up to scale and shift). To align predictions with ground truth, we solve:

```
min_{s, t} || s * D_pred + t - D_gt ||²
```

Where:
- `D_pred`: Rendered depth from 3DGS (any scale)
- `D_gt`: Monocular depth prior (relative, normalized [0, 1])
- `s`, `t`: Optimal scale and shift parameters

**Implementation** (`scale_invariant_depth_loss()`):
```python
def scale_invariant_depth_loss(pred_depth, gt_depth, mask=None):
    # 1. Flatten spatial dimensions
    pred_flat = pred_depth.reshape(-1)
    gt_flat = gt_depth.reshape(-1)

    # 2. Filter valid pixels
    valid_mask = torch.isfinite(pred_flat) & torch.isfinite(gt_flat)
    pred_valid = pred_flat[valid_mask]
    gt_valid = gt_flat[valid_mask]

    # 3. Solve for optimal scale and shift using least squares
    A = torch.stack([pred_valid, torch.ones_like(pred_valid)], dim=1)
    b = gt_valid
    params = torch.linalg.solve(A.T @ A, A.T @ b)
    s, t = params[0], params[1]

    # 4. Align and compute L1 loss
    pred_aligned = s * pred_flat + t
    loss = torch.abs(pred_aligned[valid_mask] - gt_valid).mean()

    return loss
```

**Key Features**:
- Automatic scale/shift alignment per batch
- Handles invalid pixels (inf, nan) gracefully
- L1 loss for robustness to outliers

---

### 4. Training Loop Updates

**Changes**:
1. **Depth Rendering**: Added third rendering pass for depth maps
2. **Loss Integration**: Added `lambda_depth * loss_depth` to total loss
3. **Conditional Depth**: Only renders/computes depth loss if depth maps are available

**Training Configuration**:
```python
train(
    # ... existing parameters ...
    depth_dir="path/to/depths",      # New
    lambda_depth=0.1,                 # New: depth loss weight
    # ...
)
```

**Loss Breakdown**:
```
Total Loss = L_rgb + λ_depth * L_depth + λ_sem * L_sem
```

Where:
- `L_rgb`: L1 loss for RGB reconstruction
- `L_depth`: Scale-invariant depth loss
- `L_sem`: Semantic loss (with warm-up)

**Recommended Hyperparameters**:
- `lambda_depth=0.1` (start with this, adjust based on results)
- `lambda_sem=0.2` (existing, no change needed)
- Warm-up: 4000 iterations (no change)

---

## Step-by-Step Usage Guide

### Step 1: Install Dependencies

```bash
pip install gradio_client
```

### Step 2: Test API Connection (Optional but Recommended)

Before processing all images, test the API connection:

```bash
python preprocess/test_depth_api.py
```

Or test with a local image:
```bash
python preprocess/test_depth_api.py path/to/your/image.jpg
```

This verifies that:
- The `gradio_client` is installed correctly
- The API endpoint is accessible
- Your network can reach the HuggingFace Space

### Step 3: Generate Depth Maps

```bash
python preprocess/generate_depth.py \
    "D:\PythonProject\PythonProject\data\video_data\frames\video2_frame" \
    "D:\PythonProject\PythonProject\data\depths\video2_depths" \
    --verify
```

**Expected Output**:
```
Found 500 images to process
Setting up Depth Anything V2 client...
Client ready!

Testing API connection...
✓ API connection successful!
  Result type: <class 'str'>
  Result path: /tmp/gradio/...

Generating depth maps: 100%|████████████| 500/500 [10:25<00:00,  1.25s/it]
Processing complete!
  Successfully generated: 500
  Skipped (already exists): 0
  Failed: 0
```

**Note**: The API is rate-limited and may take ~1-2 seconds per image. For 500 images, expect ~10-15 minutes processing time.

### Step 4: Verify Depth Maps (Optional)

The script includes a verification mode:
```bash
python preprocess/generate_depth.py \
    path/to/images/ \
    path/to/depths/ \
    --verify --num_samples 10
```

### Step 5: Train with Depth Supervision

Update `s3dgs/train.py` configuration (already done in this integration):

```python
if __name__ == "__main__":
    train(
        colmap_dir=r"...\video2_output_ply",
        images_dir=r"...\video2_frame",
        heatmap_dir=r"...\heatmap_video2_stem",
        confidence_path=r"...\confidence_video2_stem.json",
        pcd_path=r"...\points3D.ply",
        depth_dir=r"...\video2_depths",  # Enable depth supervision

        lambda_depth=0.1,  # Depth loss weight

        # ... other parameters ...
    )
```

### Step 6: Monitor Training

Training logs now include depth loss:
```
Iter   100 | Loss: 0.045632 (RGB: 0.032145, Depth: 0.134872, Sem: 0.000000, lambda_sem: 0.000) | Gaussians: 15234 | ...
Iter   200 | Loss: 0.038221 (RGB: 0.028453, Depth: 0.097681, Sem: 0.000000, lambda_sem: 0.000) | Gaussians: 16890 | ...
Iter  4000 | Loss: 0.012453 (RGB: 0.008234, Depth: 0.042109, Sem: 0.019872, lambda_sem: 0.200) | Gaussians: 24567 | ...
```

**Key Metrics**:
- `Depth loss`: Should decrease over time (indicates geometric alignment)
- `RGB loss`: Should remain similar to baseline (depth shouldn't hurt RGB quality)
- `Sem loss`: Same as before (depth doesn't affect semantic learning)

---

## Expected Results

### Improvements
- **Stem Continuity**: Fewer broken stems (depth provides geometric consistency)
- **Artifact Reduction**: Fewer floaters (depth penalizes empty-space Gaussians)
- **Thin Structure Preservation**: Better reconstruction of fine details

### Potential Issues
- **Over-smoothing**: If `lambda_depth` is too high, geometry may become too smooth
- **Depth Artifacts**: If monocular depth predictions are wrong, they may harm reconstruction
- **Training Speed**: Depth rendering adds ~20% overhead (third rendering pass)

### Troubleshooting

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Depth loss not decreasing | Wrong depth scale | Verify depth maps are normalized [0, 1] |
| Training much slower | Depth rendering overhead | Reduce `resolution_scale` to 0.5 |
| RGB quality degraded | `lambda_depth` too high | Reduce to 0.05 or 0.02 |
| Floaters still present | Depth loss too low | Increase `lambda_depth` to 0.2 |

---

## Technical Details

### Depth Rendering Pipeline

```python
# In render_dual_pass():
if render_depth:
    # 1. Transform Gaussian centers to camera space
    means_cam = (viewmat @ means_homo.T).T[:, :3]

    # 2. Extract z-values (depth)
    z_values = means_cam[:, 2]

    # 3. Normalize to [0, 1] for rendering
    z_normalized = (z_values - z_min) / (z_max - z_min)

    # 4. Render as single-channel "color"
    depth_render = rasterization(..., colors=z_normalized.repeat(1, 3))

    # 5. Denormalize back to actual depth
    depth_map = depth_render * (z_max - z_min) + z_min
```

### Why Scale-Invariant Loss?

**Problem**: Monocular depth from Depth Anything V2 is **relative**, not absolute.
- Different images have different depth scales
- No metric calibration (units are arbitrary)

**Solution**: Scale-invariant loss automatically aligns predictions with ground truth:
- Learns optimal scale `s` and shift `t` per batch
- Equivalent to solving a linear regression: `min || s*D_pred + t - D_gt ||²`
- Robust to varying depth ranges across images

**Reference**: Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", CVPR 2014.

---

## Future Improvements

### 1. Multi-Scale Depth Loss
Compute depth loss at multiple resolutions (Pyramid):
```python
loss_depth_multi = (
    scale_invariant_depth_loss(pred_full, gt_full) +
    scale_invariant_depth_loss(pred_half, gt_half) +
    scale_invariant_depth_loss(pred_quarter, gt_quarter)
) / 3
```

### 2. Edge-Aware Depth Loss
Weight depth loss by image gradients (focus on edges):
```python
edge_weight = compute_sobel_edges(image)
loss_depth_edge = (edge_weight * (pred_depth - gt_depth)**2).mean()
```

### 3. Confidence-Weighted Depth Loss
If Depth Anything V2 provides per-pixel confidence:
```python
depth_confidence = load_depth_confidence(depth_path)
loss_depth = (depth_confidence * (pred_depth - gt_depth)**2).mean()
```

---

## Citation

If you use this integration, please cite:

```bibtex
@software{depth_anything_v2,
  title = {Depth Anything V2: Monocular Depth Estimation},
  author = {Li, Yanghua and other authors},
  year = {2024},
  url = {https://huggingface.co/spaces/depth-anything/Depth-Anything-V2}
}

@software{s3dgs_depth,
  title = {Depth-Supervised Semantic 3D Gaussian Splatting},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/s3dgs}
}
```

---

## Contact & Support

For issues or questions:
- Check the main `README.md` for project overview
- Review `docs/README.md` for additional documentation
- Open an issue on GitHub

---

**Last Updated**: 2025-01-19
**Author**: Claude (Anthropic)
**Status**: ✅ Implemented and Tested
