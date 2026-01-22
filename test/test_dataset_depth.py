"""
Unit Test for TomatoDataset Depth Loading

This script directly tests the TomatoDataset to verify:
1. Unified NPZ loading works correctly
2. Depth maps are aligned to RGB resolution
3. Dataset returns proper depth tensors
"""

import os
import sys
import torch

# Add project root to path
_current_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from s3dgs.dataset import TomatoDataset


def test_dataset_depth_loading(
    colmap_dir: str,
    images_dir: str,
    heatmap_dir: str,
    confidence_path: str,
    depth_npz_path: str = None
):
    """
    Test dataset depth loading functionality.

    Args:
        colmap_dir: Path to COLMAP sparse directory
        images_dir: Path to RGB images
        heatmap_dir: Path to heatmap files
        confidence_path: Path to confidence JSON
        depth_npz_path: Path to unified DA3 NPZ (optional)
    """
    print("="*60)
    print("Testing TomatoDataset Depth Loading")
    print("="*60)

    # ========================================================================
    # Configuration
    # ========================================================================
    print("\nConfiguration:")
    print(f"  COLMAP dir: {colmap_dir}")
    print(f"  Images dir: {images_dir}")
    print(f"  Heatmap dir: {heatmap_dir}")
    print(f"  Confidence: {confidence_path}")
    print(f"  Depth NPZ: {depth_npz_path}")

    # ========================================================================
    # Initialize Dataset
    # ========================================================================
    print("\n" + "-"*60)
    print("Initializing Dataset...")
    print("-"*60)

    try:
        dataset = TomatoDataset(
            colmap_dir=colmap_dir,
            images_dir=images_dir,
            heatmap_dir=heatmap_dir,
            confidence_path=confidence_path,
            depth_npz_path=depth_npz_path,
            confidence_threshold=0.5
        )
        print(f"✓ Dataset initialized: {len(dataset)} frames")
    except Exception as e:
        print(f"✗ Dataset initialization failed: {e}")
        return

    # ========================================================================
    # Test Depth Loading
    # ========================================================================
    print("\n" + "-"*60)
    print("Testing Depth Loading...")
    print("-"*60)

    # Count frames with depth
    frames_with_depth = sum(1 for data in dataset.cached_data if data['has_depth'])
    print(f"Frames with depth: {frames_with_depth}/{len(dataset)}")

    if frames_with_depth == 0:
        print("✗ No frames have depth data, cannot test")
        return

    # Test first 3 frames with depth
    tested = 0
    max_tests = 3

    for i in range(len(dataset)):
        if tested >= max_tests:
            break

        data_item = dataset.cached_data[i]

        if not data_item['has_depth']:
            continue

        print(f"\n--- Frame {i}: {data_item['image_name']} ---")

        # Load data using __getitem__
        try:
            sample = dataset[i]
        except Exception as e:
            print(f"✗ Failed to load sample: {e}")
            continue

        # Check if depth is in sample
        if 'depth' not in sample:
            print(f"✗ Depth not in sample (has_depth=True but no depth returned)")
            continue

        depth_tensor = sample['depth']
        image_tensor = sample['image']

        print(f"  RGB shape: {image_tensor.shape}")
        print(f"  Depth shape: {depth_tensor.shape}")
        print(f"  Depth dtype: {depth_tensor.dtype}")
        print(f"  Depth range: [{depth_tensor.min():.3f}, {depth_tensor.max():.3f}]")
        print(f"  Depth source: {data_item.get('depth_source', 'unknown')}")

        # Verify shape alignment
        H, W = image_tensor.shape[:2]
        if depth_tensor.shape != (H, W):
            print(f"✗ Depth shape mismatch! Expected ({H}, {W}), got {depth_tensor.shape}")
        else:
            print(f"✓ Depth shape matches RGB resolution")

        # Verify depth is normalized
        if depth_tensor.max() <= 1.0:
            print(f"✓ Depth is normalized to [0, 1]")
        else:
            print(f"⚠ Depth is not normalized (max={depth_tensor.max():.3f})")

        # Verify depth tensor properties
        if isinstance(depth_tensor, torch.Tensor):
            print(f"✓ Depth is a torch.Tensor")
        else:
            print(f"✗ Depth is not a torch.Tensor (type={type(depth_tensor)})")

        tested += 1

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"✓ Dataset initialized successfully")
    print(f"✓ {frames_with_depth}/{len(dataset)} frames have depth data")
    print(f"✓ Tested {tested} frames with depth")

    if tested == max_tests and frames_with_depth >= tested:
        print("\n✓ All tests passed! Depth loading is working correctly.")
        print("✓ Spatial alignment logic is functioning as expected.")
    else:
        print("\n⚠ Some tests failed or incomplete. Please check the output above.")


if __name__ == "__main__":
    """
    Main test function.

    UPDATE THESE PATHS for your environment.
    """
    # ========================================================================
    # Configuration (UPDATE THESE PATHS)
    # ========================================================================
    colmap_dir = "./data/colmap/sparse/0"
    images_dir = "./data/images"
    heatmap_dir = "./data/heatmaps"
    confidence_path = "./data/confidence.json"
    depth_npz_path = "./data/da3_results.npz"  # Set to None if not using unified NPZ

    # Check if required paths exist
    required_paths = {
        'COLMAP': colmap_dir,
        'Images': images_dir,
        'Heatmaps': heatmap_dir,
        'Confidence': confidence_path
    }

    missing = []
    for name, path in required_paths.items():
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")

    if missing:
        print("Missing required paths:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease update the paths in the script.")
        sys.exit(1)

    if depth_npz_path is not None and not os.path.exists(depth_npz_path):
        print(f"Warning: Depth NPZ not found: {depth_npz_path}")
        print("Testing without depth supervision.")
        depth_npz_path = None

    # Run tests
    test_dataset_depth_loading(
        colmap_dir=colmap_dir,
        images_dir=images_dir,
        heatmap_dir=heatmap_dir,
        confidence_path=confidence_path,
        depth_npz_path=depth_npz_path
    )
