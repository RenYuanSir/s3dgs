"""
Diagnostic script to understand DA3 depth values

This script helps determine:
1. What is the actual depth range?
2. Are smaller values closer or farther?
3. Should we invert the depth for training?
"""

import numpy as np
import cv2
import os


def diagnose_depth_values(npz_path: str, image_dir: str):
    """
    Analyze depth values to understand their semantic meaning.
    """
    print("="*60)
    print("DA3 Depth Diagnostic Tool")
    print("="*60)

    # Load NPZ
    print(f"\nLoading NPZ from: {npz_path}")
    data = np.load(npz_path)
    depth_all = data['depth']  # [N, H, W]
    image_all = data.get('image')  # [N, H, W, 3] (optional)

    print(f"Depth shape: {depth_all.shape}")
    print(f"Depth dtype: {depth_all.dtype}")

    # Analyze first frame
    depth_0 = depth_all[0]
    print(f"\n=== First Frame Analysis ===")
    print(f"Shape: {depth_0.shape}")
    print(f"Min: {depth_0.min():.6f}")
    print(f"Max: {depth_0.max():.6f}")
    print(f"Mean: {depth_0.mean():.6f}")
    print(f"Std: {depth_0.std():.6f}")

    # Sample some points
    print(f"\n=== Sample Depth Values ===")

    # Center of image (likely the main object)
    H, W = depth_0.shape
    center_h, center_w = H // 2, W // 2
    center_depth = depth_0[center_h, center_w]
    print(f"Center point depth: {center_depth:.6f}")

    # Corners (likely background)
    corner_depths = [
        depth_0[0, 0],           # Top-left
        depth_0[0, -1],          # Top-right
        depth_0[-1, 0],          # Bottom-left
        depth_0[-1, -1],         # Bottom-right
    ]
    print(f"Corner depths: {[f'{d:.6f}' for d in corner_depths]}")
    print(f"Corner average: {np.mean(corner_depths):.6f}")

    # Determine semantic
    print(f"\n=== Semantic Analysis ===")
    if center_depth < np.mean(corner_depths):
        print("✓ Center (object) has SMALLER depth values")
        print("✓ Corners (background) have LARGER depth values")
        print("\nConclusion: DA3 uses 'metric depth' semantics:")
        print("  - Smaller values = closer to camera")
        print("  - Larger values = farther from camera")
        print("\nRecommendation: Keep depth as-is for training.")
        print("  The scale-invariant loss will handle scale alignment.")
    else:
        print("✓ Center (object) has LARGER depth values")
        print("✓ Corners (background) have SMALLER depth values")
        print("\nConclusion: DA3 might use 'inverse depth' or different encoding")
        print("  - Larger values = closer to camera")
        print("  - Smaller values = farther from camera")
        print("\nRecommendation: Consider inverting depth before training.")

    # Check if we should invert
    print(f"\n=== Visualization Check ===")
    print("For intuitive visualization (Red=c close, Blue=far):")
    if center_depth < np.mean(corner_depths):
        print("  - Use: 1.0 - normalized_depth  (invert for display)")
    else:
        print("  - Use: normalized_depth  (no inversion needed)")

    # Overall statistics across all frames
    print(f"\n=== All Frames Statistics ===")
    print(f"Global min: {depth_all.min():.6f}")
    print(f"Global max: {depth_all.max():.6f}")
    print(f"Global mean: {depth_all.mean():.6f}")

    # Check if values are in metric range (meters)
    if depth_all.max() < 100.0:
        print("✓ Values appear to be in METRIC units (meters)")
    else:
        print("✓ Values appear to be NORMALIZED or in different units")

    print("\n" + "="*60)


if __name__ == "__main__":
    # UPDATE THESE PATHS
    npz_path = "./data/da3_results.npz"

    if not os.path.exists(npz_path):
        print(f"Error: NPZ file not found: {npz_path}")
        print("Please update the npz_path in the script.")
    else:
        diagnose_depth_values(npz_path, None)
