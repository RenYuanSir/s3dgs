# Depth Data Testing Scripts

This directory contains test scripts to validate depth data loading and spatial alignment.

## Files

### 1. `test_depth_alignment.py`
**Purpose:** Visual verification of depth-to-RGB alignment

This script loads the unified NPZ file and original RGB images, then creates a side-by-side visualization of:
- Original RGB image
- Depth map (aligned to RGB resolution)
- Overlay visualization

**Usage:**
```bash
cd test
python test_depth_alignment.py
```

**Configuration:** Update the paths in the `main()` function:
```python
npz_path = "./data/da3_results.npz"  # Path to unified DA3 results
images_dir = "./data/images"          # Path to original RGB images
output_path = "./test/depth_alignment_visualization.png"
```

**Expected Output:**
- Console output showing shape information
- Visualization saved to `depth_alignment_visualization.png`
- Interactive matplotlib window showing comparisons

**Verification Checklist:**
- ✓ RGB images load correctly
- ✓ Depth maps are resized to match RGB resolution
- ✓ Depth overlay aligns with RGB features
- ✓ No spatial distortions or aspect ratio issues

---

### 2. `test_dataset_depth.py`
**Purpose:** Unit test for TomatoDataset depth loading

This script directly tests the `TomatoDataset` class to verify:
1. Unified NPZ loading works correctly
2. Depth maps are aligned to RGB resolution
3. Dataset returns proper depth tensors

**Usage:**
```bash
cd test
python test_dataset_depth.py
```

**Configuration:** Update the paths in the `if __name__ == "__main__":` section:
```python
colmap_dir = "./data/colmap/sparse/0"
images_dir = "./data/images"
heatmap_dir = "./data/heatmaps"
confidence_path = "./data/confidence.json"
depth_npz_path = "./data/da3_results.npz"  # Set to None if not using unified NPZ
```

**Expected Output:**
```
==============================================================
Testing TomatoDataset Depth Loading
================================================--------------

Configuration:
  COLMAP dir: ./data/colmap/sparse/0
  Images dir: ./data/images
  ...

------------------------------------------------------------
Initializing Dataset...
------------------------------------------------------------
✓ Dataset initialized: 96 frames

------------------------------------------------------------
Testing Depth Loading...
------------------------------------------------------------
Frames with depth: 96/96

--- Frame 0: image_001.jpg ---
  RGB shape: torch.Size([540, 960, 3])
  Depth shape: torch.Size([540, 960])
  ✓ Depth shape matches RGB resolution
  ✓ Depth is normalized to [0, 1]
  ✓ Depth is a torch.Tensor

...

==============================================================
Test Summary
==============================================================
✓ All tests passed! Depth loading is working correctly.
✓ Spatial alignment logic is functioning as expected.
```

---

## How These Scripts Work

### Spatial Alignment Logic

Both scripts test the critical alignment logic in `s3dgs/dataset.py:520-524`:

```python
# DA3 depth is (280, 504), need to resize to (H, W)
depth_aligned = cv2.resize(
    depth_small,
    (W, H),  # (width, height) - OpenCV uses (W, H) convention
    interpolation=cv2.INTER_LINEAR
)  # [H, W]
```

This ensures that:
1. Depth prior pixels align with rendered rays
2. No spatial mismatch between depth supervision and RGB reconstruction
3. Bilinear interpolation preserves smooth depth gradients

### Index Mapping

The unified NPZ file uses index-based access. The dataset assumes:
- NPZ array order matches sorted filenames
- Each depth[i] corresponds to the i-th sorted image

This mapping is built in `dataset.py:393-396`:
```python
image_list = sorted([
    img_data['name']
    for img_data in self.images_data.values()
])
```

---

## Troubleshooting

### Issue: "NPZ file not found"
**Solution:** Update the `npz_path` variable to point to your actual `da3_results.npz` file.

### Issue: "Depth shape mismatch"
**Solution:** This indicates a bug in the alignment logic. Check that:
- `cv2.resize` is using `(W, H)` not `(H, W)`
- The interpolation mode is `cv2.INTER_LINEAR`

### Issue: "Depth not normalized"
**Solution:** The normalization logic should be in `dataset.py:527-529`:
```python
if depth_aligned.max() > 1.0:
    depth_aligned = depth_aligned / depth_aligned.max()
```

### Issue: "No frames have depth data"
**Solution:** Verify that:
1. The NPZ file exists and is readable
2. The image names in COLMAP match the NPZ ordering
3. Check console warnings for specific issues

---

## Integration with Training

Once tests pass, use the unified NPZ in training:

```python
from s3dgs.train import train

train(
    colmap_dir="./data/colmap/sparse/0",
    images_dir="./data/images",
    heatmap_dir="./data/heatmaps",
    confidence_path="./data/confidence.json",
    pcd_path="./data/points3d.ply",
    depth_npz_path="./data/da3_results.npz",  # Use unified NPZ
    lambda_depth=0.1,  # Depth loss weight
    ...
)
```

The scale-invariant depth loss will automatically:
1. Render depth from the 3DGS model
2. Align scale/shift between prediction and DA3 prior
3. Apply L1 loss on the aligned depths

---

## References

- **Dataset Implementation:** `s3dgs/dataset.py:506-576`
- **Depth Loss:** `s3dgs/loss.py:29-104`
- **Depth Rendering:** `s3dgs/render.py:140-206`
- **Training Loop:** `s3dgs/train.py:339-353`
