"""
Test script for inject_semantics.py

This script demonstrates how to use the semantic injection tool
with synthetic data for testing purposes.
"""

import os
import numpy as np
import torch
import plyfile
from tools.inject_semantics import Config, inject_semantics


def create_synthetic_point_cloud(output_path: str, num_points: int = 10000):
    """
    Create a synthetic point cloud for testing.

    Args:
        output_path: Path to save the PLY file
        num_points: Number of points to generate
    """
    print(f"Creating synthetic point cloud with {num_points} points...")

    # Generate random points in a cube
    xyz = np.random.rand(num_points, 3).astype(np.float32) * 10 - 5  # [-5, 5]

    # Generate random colors
    rgb = np.random.randint(0, 256, (num_points, 3), dtype=np.uint8)

    # Create PLY structure
    vertex_data = [
        (xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb[i, 0], rgb[i, 1], rgb[i, 2])
        for i in range(num_points)
    ]

    vertex_elements = np.array(vertex_data, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])

    el = plyfile.PlyElement.describe(vertex_elements, 'vertex')
    ply_data = plyfile.PlyData([el], text=False)
    ply_data.write(output_path)

    print(f"Saved point cloud to: {output_path}")


def create_synthetic_heatmaps(output_dir: str, num_frames: int = 10,
                               img_height: int = 1080, img_width: int = 1920,
                               num_classes: int = 4):
    """
    Create synthetic semantic heatmaps for testing.

    Args:
        output_dir: Directory to save heatmap files
        num_frames: Number of frames to generate
        img_height: Image height
        img_width: Image width
        num_classes: Number of semantic classes
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating {num_frames} synthetic heatmaps...")

    for i in range(num_frames):
        # Create random probability distribution
        heatmap = np.random.rand(img_height, img_width, num_classes).astype(np.float32)

        # Normalize to sum to 1 per pixel (softmax-like)
        heatmap = heatmap / np.sum(heatmap, axis=2, keepdims=True)

        # Save heatmap
        filename = f"frame_{i:03d}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, heatmap)

    print(f"Saved heatmaps to: {output_dir}")


def create_synthetic_colmap_data(output_dir: str, num_frames: int = 10,
                                  img_height: int = 1080, img_width: int = 1920):
    """
    Create synthetic COLMAP data (cameras.bin and images.bin) for testing.

    Args:
        output_dir: Directory to save COLMAP files
        num_frames: Number of frames/images
        img_height: Image height
        img_width: Image width
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating synthetic COLMAP data for {num_frames} frames...")

    # Create cameras.bin
    cameras_bin_path = os.path.join(output_dir, "cameras.bin")

    with open(cameras_bin_path, "wb") as fid:
        # Write number of cameras
        fid.write(b'\x01\x00\x00\x00\x00\x00\x00\x00')  # 1 camera (uint64)

        # Write camera 1
        fid.write(b'\x01\x00\x00\x00')  # camera_id = 1 (uint32)
        fid.write(b'\x01\x00\x00\x00')  # model = PINHOLE (uint32)
        fid.write(img_width.to_bytes(8, 'little'))  # width (uint64)
        fid.write(img_height.to_bytes(8, 'little'))  # height (uint64)

        # PINHOLE parameters: fx, fy, cx, cy (doubles)
        fx = fy = 1000.0
        cx = img_width / 2.0
        cy = img_height / 2.0

        import struct
        fid.write(struct.pack('<dddd', fx, fy, cx, cy))

    print(f"Created cameras.bin")

    # Create images.bin
    images_bin_path = os.path.join(output_dir, "images.bin")

    with open(images_bin_path, "wb") as fid:
        # Write number of images
        fid.write(num_frames.to_bytes(8, 'little'))  # num_images (uint64)

        for i in range(num_frames):
            # Write image_id
            fid.write((i + 1).to_bytes(4, 'little'))  # image_id (uint32)

            # Write quaternion (identity rotation)
            fid.write(struct.pack('<dddd', 1.0, 0.0, 0.0, 0.0))  # qw, qx, qy, qz

            # Write translation (moving in a circle)
            angle = 2 * np.pi * i / num_frames
            radius = 5.0
            tx = radius * np.cos(angle)
            ty = 0.0
            tz = radius * np.sin(angle)
            fid.write(struct.pack('<ddd', tx, ty, tz))  # tx, ty, tz

            # Write camera_id
            fid.write(b'\x01\x00\x00\x00')  # camera_id = 1 (uint32)

            # Write image name (null-terminated string)
            image_name = f"frame_{i:03d}.jpg\0"
            fid.write(image_name.encode('utf-8'))

            # Write num_2d_points (0)
            fid.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')  # 0 points (uint64)

    print(f"Created images.bin")
    print(f"Saved COLMAP data to: {output_dir}")


def test_inject_semantics():
    """
    Test the semantic injection pipeline with synthetic data.
    """
    print("=" * 60)
    print("Testing Semantic Injection Pipeline")
    print("=" * 60)

    # Create temporary directory for test data
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)

    # Create synthetic data
    pcd_path = os.path.join(test_dir, "test_points3D.ply")
    heatmap_dir = os.path.join(test_dir, "heatmaps")
    poses_path = os.path.join(test_dir, "sparse")
    output_dir = os.path.join(test_dir, "output")

    create_synthetic_point_cloud(pcd_path, num_points=10000)
    create_synthetic_heatmaps(heatmap_dir, num_frames=10)
    create_synthetic_colmap_data(poses_path, num_frames=10)

    # Configure and run injection
    cfg = Config()
    cfg.dense_pcd_path = pcd_path
    cfg.heatmap_dir = heatmap_dir
    cfg.poses_path = poses_path
    cfg.output_dir = output_dir
    cfg.num_classes = 4
    cfg.spatial_kernel_size = 5
    cfg.confidence_threshold = 0.3

    print("\n" + "=" * 60)
    print("Running semantic injection...")
    print("=" * 60 + "\n")

    # Run injection (commented out to avoid actual execution in this test)
    # inject_semantics(cfg)

    print("\n" + "=" * 60)
    print("Test setup complete!")
    print("=" * 60)
    print(f"\nTest data created in: {test_dir}")
    print(f"To run the actual injection, uncomment the line:")
    print("    inject_semantics(cfg)")
    print(f"\nExpected output file: {os.path.join(output_dir, 'test_points3D_semantics.pt')}")


if __name__ == "__main__":
    test_inject_semantics()
