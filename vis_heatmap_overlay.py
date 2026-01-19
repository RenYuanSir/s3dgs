"""
Heatmap Overlay Visualization Tool
==================================

Visualizes skeleton-based heatmaps overlaid on original RGB images.
This helps verify that the skeleton lines are correctly connecting keypoints
and that the adaptive thickness is working as expected.

Semantic Channel Mapping:
- U (Up Node): Red
- N (Node): Green
- D (Down Node): Blue
- P (Peduncle): Yellow

Usage:
    python vis_heatmap_overlay.py
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Configuration
IMAGE_DIR = r"D:\PythonProject\PythonProject\data\video_data\frames\video2_frame"
HEATMAP_DIR = r"D:\PythonProject\PythonProject\data\heatmaps\heatmap_video2_stem"
OUTPUT_DIR = r"D:\PythonProject\PythonProject\debug\heatmap_vis"
NUM_SAMPLES = 5  # Visualize first N frames

# Channel to RGB color mapping
CHANNEL_COLORS = {
    0: [255, 0, 0],    # U -> Red
    1: [0, 255, 0],    # N -> Green
    2: [0, 0, 255],    # D -> Blue
    3: [255, 255, 0]   # P -> Yellow
}

CHANNEL_NAMES = {
    0: 'U (Up Node)',
    1: 'N (Node)',
    2: 'D (Down Node)',
    3: 'P (Peduncle)'
}


def create_colormap_from_color(rgb_color):
    """Create a single-channel colormap from an RGB color."""
    color = np.array(rgb_color) / 255.0

    cdict = {
        'red':   [(0.0, 0.0, 0.0), (1.0, color[0], color[0])],
        'green': [(0.0, 0.0, 0.0), (1.0, color[1], color[1])],
        'blue':  [(0.0, 0.0, 0.0), (1.0, color[2], color[2])]
    }

    return LinearSegmentedColormap('custom_cmap', cdict)


def visualize_heatmap_overlay(image, heatmap, save_path):
    """
    Create a visualization with:
    1. Original RGB image
    2. Combined heatmap overlay
    3. Individual channel heatmaps
    """

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Skeleton Heatmap Visualization', fontsize=16, fontweight='bold')

    # Plot 1: Original RGB Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original RGB Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Plot 2-5: Individual Channel Heatmaps
    # Mapping: channel_idx -> subplot position
    # channel 0: axes[0, 1], channel 1: axes[0, 2]
    # channel 2: axes[1, 0], channel 3: axes[1, 1]
    channel_positions = [
        (0, 1),  # Channel 0 (U)
        (0, 2),  # Channel 1 (N)
        (1, 0),  # Channel 2 (D)
        (1, 1),  # Channel 3 (P)
    ]

    for channel_idx in range(4):
        row, col = channel_positions[channel_idx]
        ax = axes[row, col]

        # Get heatmap channel
        channel_heatmap = heatmap[:, :, channel_idx]

        # Create custom colormap
        cmap = create_colormap_from_color(CHANNEL_COLORS[channel_idx])

        # Display heatmap
        im = ax.imshow(channel_heatmap, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f'Channel {channel_idx}: {CHANNEL_NAMES[channel_idx]}', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot 6: Combined Overlay (All Channels)
    # Normalize each channel and combine
    normalized_heatmap = np.zeros_like(heatmap)
    for c in range(4):
        channel = heatmap[:, :, c]
        max_val = np.max(channel)
        if max_val > 0:
            normalized_heatmap[:, :, c] = channel / max_val

    # Create RGB composite
    rgb_heatmap = np.zeros((heatmap.shape[0], heatmap.shape[1], 3))
    for c in range(4):
        color = np.array(CHANNEL_COLORS[c]) / 255.0
        for rgb_idx in range(3):
            rgb_heatmap[:, :, rgb_idx] += normalized_heatmap[:, :, c] * color[rgb_idx]

    # Clip to valid range
    rgb_heatmap = np.clip(rgb_heatmap, 0, 1)

    # Blend with original image (50% each)
    blended = 0.5 * image + 0.5 * rgb_heatmap

    axes[1, 0].imshow(blended)
    axes[1, 0].set_title('Overlay: RGB + All Channels (50%/50%)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [SAVED] {save_path}")


def count_lines_in_heatmap(heatmap):
    """Count how many pixels are activated (non-zero) in each channel."""
    counts = {}
    for c in range(4):
        channel = heatmap[:, :, c]
        # Count pixels with value > 0.1 (threshold for "active")
        active_pixels = np.sum(channel > 0.1)
        counts[CHANNEL_NAMES[c]] = active_pixels

    return counts


def main():
    """Main execution function."""

    print("=" * 60)
    print("Skeleton Heatmap Visualization")
    print("=" * 60)
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_DIR}")
    print()

    # Get list of frames
    frame_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])[:NUM_SAMPLES]

    if not frame_files:
        print(f"[ERROR] No frames found in {IMAGE_DIR}")
        return

    print(f"[INFO] Found {len(frame_files)} frames to visualize")
    print()

    # Process each frame
    for idx, frame_file in enumerate(frame_files):
        print(f"[{idx+1}/{len(frame_files)}] Processing: {frame_file}")

        # Load RGB image
        image_path = os.path.join(IMAGE_DIR, frame_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]

        # Load corresponding heatmap
        heatmap_file = frame_file.replace('.jpg', '.npy')
        heatmap_path = os.path.join(HEATMAP_DIR, heatmap_file)

        if not os.path.exists(heatmap_path):
            print(f"  [WARNING] Heatmap not found: {heatmap_file}")
            continue

        heatmap = np.load(heatmap_path)

        print(f"    Image shape: {image.shape}")
        print(f"    Heatmap shape: {heatmap.shape}")

        # Count active pixels in each channel
        counts = count_lines_in_heatmap(heatmap)
        print(f"    Active pixels:")
        for channel_name, count in counts.items():
            print(f"      - {channel_name}: {count} pixels")

        # Generate visualization
        basename = os.path.splitext(frame_file)[0]
        save_path = os.path.join(OUTPUT_DIR, f"{basename}_overlay.png")

        visualize_heatmap_overlay(image, heatmap, save_path)
        print()

    print("=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print()
    print(f"[INFO] Check the output directory: {OUTPUT_DIR}")
    print("[INFO] Look for:")
    print("  - Skeleton lines connecting U-N-D-P keypoints")
    print("  - Adaptive thickness (thicker for main stems, thinner for peduncles)")
    print("  - Smooth gradients from skeleton lines to background")
    print()


if __name__ == "__main__":
    main()
