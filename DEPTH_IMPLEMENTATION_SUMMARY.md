# Depth Data Loading and Scale-Invariant Depth Loss Implementation

## Summary

This implementation adds support for loading Depth Anything V3 (DA3) priors from a unified NPZ file with proper spatial alignment and scale-invariant depth loss for training.

## Key Changes

### 1. Dataset Updates (`s3dgs/dataset.py`)

#### New Parameter: `depth_npz_path`
- Added `depth_npz_path` parameter to `TomatoDataset.__init__()` (line 318)
- Added to `create_dataloader()` function (line 636)

#### Unified NPZ Loading (lines 361-372)
```python
# Load unified NPZ file if provided (Depth Anything V3 format)
self.unified_depth_data = None
if depth_npz_path is not None:
    if os.path.exists(depth_npz_path):
        print(f"Loading unified depth NPZ from: {depth_npz_path}")
        self.unified_depth_data = np.load(depth_npz_path)
        print(f"  NPZ keys: {list(self.unified_depth_data.files)}")
        print(f"  depth shape: {self.unified_depth_data['depth'].shape}")
```

#### Index-Based Mapping (lines 391-429)
- Builds sorted image list for index-based NPZ access
- Assumes NPZ order matches sorted filenames
- Stores `depth_source` and `depth_index` in cached data

#### **CRITICAL: Spatial Alignment (lines 512-530)**
```python
if depth_source == 'unified_npz' and self.unified_depth_data is not None:
    # Load from unified NPZ file with spatial alignment
    # DA3 depth is (280, 504), need to resize to (H, W)
    depth_index = data['depth_index']
    depth_small = self.unified_depth_data['depth'][depth_index]  # [280, 504]

    # SPATIAL ALIGNMENT: Resize depth to match RGB resolution
    # Use INTER_LINEAR for smooth upsampling (bilinear interpolation)
    depth_aligned = cv2.resize(
        depth_small,
        (W, H),  # (width, height) - OpenCV uses (W, H) convention
        interpolation=cv2.INTER_LINEAR
    )  # [H, W]

    # Normalize to [0, 1] if not already
    if depth_aligned.max() > 1.0:
        depth_aligned = depth_aligned / depth_aligned.max()

    depth_tensor = torch.from_numpy(depth_aligned.astype(np.float32))  # [H, W]
```

**Why This Matters:**
- DA3 outputs depth at `(280, 504)` resolution
- Training images are typically much larger (e.g., `540x960` or `1080x1920`)
- Without alignment, depth supervision would be spatially mismatched with rendered rays
- Bilinear interpolation preserves smooth depth gradients while upsampling

### 2. Training Updates (`s3dgs/train.py`)

#### New Parameter (line 62)
```python
depth_npz_path: str = None,  # New: path to unified DA3 NPZ file
```

#### Dataset Initialization (lines 146, 156)
```python
dataset = TomatoDataset(
    ...
    depth_npz_path=depth_npz_path,  # Pass unified NPZ path
    ...
)
```

#### Depth Availability Check (line 164)
```python
has_depth = (depth_dir is not None or depth_npz_path is not None) and \
            any(data.get('has_depth', False) for data in dataset.cached_data)
```

### 3. Existing Depth Infrastructure (Already Complete)

#### Depth Rendering (`s3dgs/render.py:140-206`)
- Renders depth map using z-coordinate in camera space
- Returns `[H, W]` tensor with actual depth values
- Integrated with `render_dual_pass()` via `render_depth` flag

#### Scale-Invariant Depth Loss (`s3dgs/loss.py:29-104`)
- Solves for optimal scale (s) and shift (t) using least squares
- Formula: `s * pred_depth + t ≈ gt_depth`
- Handles monocular depth ambiguity (scale and shift invariant)
- Robust to invalid pixels (NaN, inf)

#### Training Integration (`s3dgs/train.py:339-353`)
```python
# Depth loss (scale-invariant, if available)
loss_depth = torch.tensor(0.0, device=device)
if pred_depth is not None and gt_depth is not None:
    # Ensure GT depth matches rendered resolution
    if gt_depth.shape != pred_depth.shape:
        gt_depth_resized = torch.nn.functional.interpolate(
            gt_depth.unsqueeze(0).unsqueeze(0),
            size=pred_depth.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze()
    else:
        gt_depth_resized = gt_depth

    loss_depth = scale_invariant_depth_loss(pred_depth, gt_depth_resized)
```

## Data Format

### Unified NPZ Structure (`da3_results.npz`)
```
da3_results.npz
├── depth: (96, 280, 504), float32  # Depth maps for all frames
└── image: (96, 280, 504, 3), uint8  # DA3 input images (optional)
```

**Assumptions:**
- NPZ array order matches sorted filenames from dataset
- Each `depth[i]` corresponds to the i-th sorted image
- Depth values are metric or relative (scale-invariant loss handles both)

## Usage

### Training with Unified NPZ

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

### Testing the Implementation

```bash
# Visual verification of depth alignment
cd test
python test_depth_alignment.py

# Unit test for dataset loading
python test_dataset_depth.py
```

## Scale-Invariant Depth Loss Details

### Why Scale-Invariant?

Monocular depth estimation (DA3) produces **relative depth** that is:
- **Scale-ambiguous:** Can be multiplied by any constant
- **Shift-ambiguous:** Can have an arbitrary offset
- **Metric-agnostic:** Not in absolute units (meters)

The scale-invariant loss solves this by aligning prediction and GT before computing loss:

### Algorithm (from `s3dgs/loss.py:29-104`)

1. **Extract valid pixels:** Filter out NaN, inf
2. **Solve for alignment:** Find optimal s, t that minimize `|| s * pred + t - gt ||²`
3. **Apply alignment:** `pred_aligned = s * pred + t`
4. **Compute loss:** `L = |pred_aligned - gt|` (L1 loss)

### Mathematical Formulation

Given:
- `pred_depth`: Rendered depth from 3DGS (any scale)
- `gt_depth`: DA3 prior (relative depth)

We solve the least squares problem:
```
min || s * pred + t - gt ||²
s,t
```

This has a closed-form solution:
```
A = [pred, 1]  # Design matrix [N, 2]
b = gt         # Target [N]

[s, t]ᵀ = (AᵀA)⁻¹ Aᵀb  # Linear least squares
```

Then compute L1 loss on aligned depth:
```
L = |s * pred + t - gt|.mean()
```

## Verification

The implementation provides two test scripts:

### 1. Visual Alignment Test (`test/test_depth_alignment.py`)
- Loads NPZ and RGB images
- Resizes depth to match RGB (mirrors dataset logic)
- Creates side-by-side visualization
- **Pass criteria:** Depth overlay aligns with RGB features

### 2. Dataset Unit Test (`test/test_dataset_depth.py`)
- Tests `TomatoDataset` directly
- Verifies depth loading and alignment
- Checks tensor properties (shape, dtype, range)
- **Pass criteria:** All assertions pass, depth shape matches RGB

## Architecture Decisions

### 1. Unified NPZ vs Individual Files
**Decision:** Support both, with unified NPZ taking precedence

**Rationale:**
- Unified NPZ is more efficient (single file read vs N files)
- Easier to generate from DA3 pipeline
- Individual files still supported for flexibility

### 2. Spatial Alignment Method
**Decision:** Use `cv2.resize` with `INTER_LINEAR` (bilinear)

**Rationale:**
- Bilinear preserves smooth gradients (important for depth)
- Faster than bicubic
- No edge artifacts compared with nearest neighbor
- Matches PyTorch's `F.interpolate(mode='bilinear')`

### 3. Index-Based Mapping
**Decision:** Assume NPZ order matches sorted filenames

**Rationale:**
- Simpler than storing filename-to-index mapping
- DA3 generation scripts can easily ensure ordering
- Fallback to individual files if mapping fails

### 4. Depth Normalization
**Decision:** Normalize to [0, 1] if max > 1.0

**Rationale:**
- DA3 outputs can be metric or relative
- Normalization ensures stable training
- Scale-invariant loss handles absolute scale anyway

## Performance Considerations

### Memory
- **NPZ Loading:** Entire NPZ loaded into memory once (fastest)
- **Individual Files:** Loaded on-demand (more memory efficient)

### Speed
- **Spatial Alignment:** `cv2.resize` is highly optimized
- **Index Lookup:** O(1) list indexing
- **Overhead:** Negligible compared to rendering

### Recommendations
- Use unified NPZ for datasets < 1000 frames
- Use individual files for very large datasets
- Cache aligned depth if training multiple runs

## Troubleshooting

### Issue: Depth shape mismatch
**Cause:** `cv2.resize` using wrong size tuple
**Fix:** Ensure `(W, H)` order (OpenCV convention)

### Issue: Index out of bounds
**Cause:** NPZ order doesn't match sorted filenames
**Fix:** Regenerate NPZ with sorted input images

### Issue: Depth loss is NaN
**Cause:** Invalid depth values or insufficient overlap
**Fix:** Check depth range, filter invalid pixels

### Issue: Depth loss doesn't decrease
**Cause:** Scale too different, need better initialization
**Fix:** Initialize Gaussians from DA3 point cloud (already done)

## References

- **Eigen et al.** "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", ICCV 2014
- **Depth Anything V3:** https://github.com/DepthAnything/Depth-Anything-V3
- **gsplat v1.5.3:** https://github.com/nerfstudio-project/gsplat

## Files Modified

1. `s3dgs/dataset.py`
   - Added `depth_npz_path` parameter
   - Implemented unified NPZ loading
   - Added spatial alignment logic

2. `s3dgs/train.py`
   - Added `depth_npz_path` parameter to `train()`
   - Updated dataset initialization

3. `test/test_depth_alignment.py` (NEW)
   - Visual verification script

4. `test/test_dataset_depth.py` (NEW)
   - Unit test script

5. `test/README.md` (NEW)
   - Testing documentation

## Next Steps

1. **Run Tests:** Execute test scripts to verify alignment
2. **Training:** Train with `lambda_depth=0.1` (adjust as needed)
3. **Monitor Loss:** Check depth loss decreases during training
4. **Ablation:** Compare training with/without depth supervision
5. **Visualization:** Render depth predictions to verify quality

---

**Implementation Status:** ✅ Complete
**Test Coverage:** ✅ Visual + Unit tests
**Documentation:** ✅ Comprehensive
