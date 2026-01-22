"""
Test Script for Depth Data Alignment Validation

This script loads the unified NPZ file and original RGB images,
then visualizes the first 3 frames with side-by-side comparison of:
- Original RGB image
- Depth map (aligned to RGB resolution)
- Overlay visualization

Purpose: Verify that the spatial alignment logic in dataset.py is correct.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import cv2

# Configure matplotlib for better display
rcParams['figure.figsize'] = [15, 10]
rcParams['figure.dpi'] = 100


def load_npz_data(npz_path: str) -> dict:
    """
    Load unified DA3 results NPZ file.

    Args:
        npz_path: Path to da3_results.npz

    Returns:
        Dictionary with 'depth' and 'image' arrays
    """
    print(f"Loading NPZ from: {npz_path}")
    data = np.load(npz_path)

    print(f"  NPZ keys: {list(data.files)}")
    print(f"  depth shape: {data['depth'].shape}")
    print(f"  image shape: {data['image'].shape}")

    return data


def load_original_images(images_dir: str, num_images: int = 3) -> list:
    """
    Load original RGB images from directory.

    Args:
        images_dir: Path to images directory
        num_images: Number of images to load

    Returns:
        List of (filename, image_array) tuples
    """
    print(f"\nLoading images from: {images_dir}")

    # Get sorted list of image files
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"  Found {len(image_files)} images")
    print(f"  Loading first {num_images} images...")

    loaded = []
    for i, filename in enumerate(image_files[:num_images]):
        path = os.path.join(images_dir, filename)
        img = cv2.imread(path)
        if img is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            loaded.append((filename, img_rgb))
            print(f"    [{i}] {filename}: shape {img_rgb.shape}")

    return loaded


def align_depth_to_rgb(depth_small: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Align depth map to RGB resolution using bilinear interpolation.
    This mirrors the logic in dataset.py:520-524

    Args:
        depth_small: Depth map [H_small, W_small]
        target_shape: Target (H, W) tuple

    Returns:
        Aligned depth map [H, W]
    """
    H, W = target_shape

    # SPATIAL ALIGNMENT: Resize depth to match RGB resolution
    # Use INTER_LINEAR for smooth upsampling (bilinear interpolation)
    depth_aligned = cv2.resize(
        depth_small,
        (W, H),  # (width, height) - OpenCV uses (W, H) convention
        interpolation=cv2.INTER_LINEAR
    )  # [H, W]

    return depth_aligned


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """
    Convert depth map to colormap for visualization.

    NOTE: Depth Anything V3 outputs metric depth where:
    - Smaller values = closer to camera
    - Larger values = farther from camera

    For visualization, we want:
    - Red/Warm colors = close (small depth values)
    - Blue/Cool colors = far (large depth values)

    Args:
        depth: Depth map [H, W]

    Returns:
        RGB colormap [H, W, 3]
    """
    # Normalize to [0, 1] - smaller is closer
    depth_min = depth.min()
    depth_max = depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

    # INVERT for visualization: we want close objects = RED, far = BLUE
    # COLORMAP_TURBO: Blue (0) -> Red (255)
    # So we need to invert: 1.0 - depth_norm
    depth_vis = 1.0 - depth_norm

    # Apply colormap
    depth_colormap = cv2.applyColorMap(
        (depth_vis * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO
    )

    # Convert BGR to RGB
    depth_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

    return depth_rgb


def create_overlay(rgb: np.ndarray, depth: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create overlay visualization of RGB and depth.

    Args:
        rgb: RGB image [H, W, 3]
        depth: Depth colormap [H, W, 3]
        alpha: Transparency weight for depth

    Returns:
        Overlay image [H, W, 3]
    """
    overlay = (1 - alpha) * rgb + alpha * depth
    return overlay.astype(np.uint8)


def visualize_comparison(
    images: list,
    depth_data: np.ndarray,
    save_path: str = None
):
    """
    Create side-by-side visualization for first 3 frames.

    Args:
        images: List of (filename, rgb_image) tuples
        depth_data: Depth array from NPZ [N, H, W]
        save_path: Optional path to save figure
    """
    num_frames = min(len(images), 3)

    fig, axes = plt.subplots(num_frames, 3, figsize=(15, 5 * num_frames))

    if num_frames == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_frames):
        filename, rgb = images[i]
        depth_small = depth_data[i]

        print(f"\n=== Frame {i}: {filename} ===")
        print(f"  RGB shape: {rgb.shape}")
        print(f"  Depth (small) shape: {depth_small.shape}")

        # Align depth to RGB resolution
        depth_aligned = align_depth_to_rgb(depth_small, rgb.shape[:2])
        print(f"  Depth (aligned) shape: {depth_aligned.shape}")
        print(f"  Depth range: [{depth_aligned.min():.3f}, {depth_aligned.max():.3f}]")

        # Convert depth to colormap
        depth_colormap = depth_to_colormap(depth_aligned)

        # Create overlay
        overlay = create_overlay(rgb, depth_colormap, alpha=0.4)

        # Plot RGB
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f'Frame {i}: {filename}\nRGB {rgb.shape}', fontsize=10)
        axes[i, 0].axis('off')

        # Plot Depth
        axes[i, 1].imshow(depth_colormap)
        axes[i, 1].set_title(f'Depth (Aligned)\n{depth_aligned.shape}', fontsize=10)
        axes[i, 1].axis('off')

        # Plot Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'Overlay (RGB + Depth)\nVerify pixel alignment', fontsize=10)
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")

    plt.show()


def main():
    """
    Main test function.
    """
    print("="*60)
    print("Depth Alignment Test Script")
    print("="*60)

    # ========================================================================
    # Configuration (UPDATE THESE PATHS)
    # ========================================================================
    npz_path = "./data/da3_results.npz"  # Path to unified DA3 results
    images_dir = "./data/images"          # Path to original RGB images
    output_path = "./test/depth_alignment_visualization.png"

    # Check if paths exist
    if not os.path.exists(npz_path):
        print(f"\n[ERROR] NPZ file not found: {npz_path}")
        print("Please update the npz_path variable in the script.")
        return

    if not os.path.exists(images_dir):
        print(f"\n[ERROR] Images directory not found: {images_dir}")
        print("Please update the images_dir variable in the script.")
        return

    # ========================================================================
    # Load Data
    # ========================================================================
    npz_data = load_npz_data(npz_path)

    # Extract depth and image arrays
    depth_all = npz_data['depth']  # [N, H, W]
    image_all = npz_data['image']  # [N, H, W, 3] (if available)

    print(f"\nDepth data range: [{depth_all.min():.3f}, {depth_all.max():.3f}]")

    # Load original RGB images
    original_images = load_original_images(images_dir, num_images=3)

    # ========================================================================
    # Visualize Comparison
    # ========================================================================
    print("\n" + "="*60)
    print("Creating Visualization...")
    print("="*60)

    visualize_comparison(
        images=original_images,
        depth_data=depth_all,
        save_path=output_path
    )

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print("\nVerification Checklist:")
    print("  [ ] RGB images load correctly")
    print("  [ ] Depth maps are resized to match RGB resolution")
    print("  [ ] Depth overlay aligns with RGB features")
    print("  [ ] No spatial distortions or aspect ratio issues")
    print("\nIf all checks pass, the dataset.py alignment logic is correct!")


if __name__ == "__main__":
    main()
