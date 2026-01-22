# Depth Loading Quick Reference

## Critical Implementation Details

### Spatial Alignment (CRITICAL)
**Location:** `s3dgs/dataset.py:520-524`

```python
# DA3 depth is (280, 504), RGB is (H, W)
depth_aligned = cv2.resize(
    depth_small,
    (W, H),  # NOTE: (W, H) not (H, W) - OpenCV convention
    interpolation=cv2.INTER_LINEAR
)
```

**Why:** Ensures depth pixels align with rendered rays during training.

---

## Usage

### Training
```python
from s3dgs.train import train

train(
    colmap_dir="./data/colmap/sparse/0",
    images_dir="./data/images",
    heatmap_dir="./data/heatmaps",
    confidence_path="./data/confidence.json",
    pcd_path="./data/points3d.ply",
    depth_npz_path="./data/da3_results.npz",  # ← Use this
    lambda_depth=0.1,
    num_iterations=7000
)
```

### Testing
```bash
cd test
python test_depth_alignment.py    # Visual test
python test_dataset_depth.py      # Unit test
```

---

## NPZ Format

```python
# da3_results.npz
{
    'depth': np.array(shape=(96, 280, 504), dtype=float32),
    'image': np.array(shape=(96, 280, 504, 3), dtype=uint8)
}

# Assumes: depth[i] corresponds to i-th sorted image
```

---

## Scale-Invariant Loss

**Algorithm:**
1. Solve: `min || s * pred + t - gt ||²`
2. Align: `pred_aligned = s * pred + t`
3. Loss: `L = |pred_aligned - gt|.mean()`

**Why:** Monocular depth is scale/shift ambiguous.

---

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `depth_npz_path` | `None` | Path to unified DA3 NPZ |
| `lambda_depth` | `0.1` | Depth loss weight |
| `render_depth` | `True` | Enable depth rendering |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Shape mismatch | Check `(W, H)` order in cv2.resize |
| Index error | Verify NPZ order matches sorted images |
| NaN loss | Filter invalid pixels (already done) |

---

## File Locations

| Component | File | Lines |
|-----------|------|-------|
| NPZ Loading | `dataset.py` | 361-372 |
| Index Mapping | `dataset.py` | 391-429 |
| Spatial Alignment | `dataset.py` | 520-524 |
| Depth Rendering | `render.py` | 140-206 |
| Depth Loss | `loss.py` | 29-104 |
| Training Loop | `train.py` | 339-353 |

---

## Common Mistakes

❌ **WRONG:** `cv2.resize(depth, (H, W))`
✅ **RIGHT:** `cv2.resize(depth, (W, H))`

❌ **WRONG:** Using depth directly without alignment
✅ **RIGHT:** Always align to RGB resolution

❌ **WRONG:** Assuming depth is metric
✅ **RIGHT:** Use scale-invariant loss

---

## Performance

| Metric | Value |
|--------|-------|
| NPZ Load Time | ~1-2 seconds |
| Resize Time | ~1-5ms per frame |
| Memory Overhead | ~50-100MB for 96 frames |

---

## Verification Checklist

- [ ] NPZ file exists and is readable
- [ ] Test scripts pass
- [ ] Depth shape matches RGB in dataset
- [ ] Depth loss decreases during training
- [ ] Visual inspection shows alignment

---

**Need help?** Check `test/README.md` or `DEPTH_IMPLEMENTATION_SUMMARY.md`
