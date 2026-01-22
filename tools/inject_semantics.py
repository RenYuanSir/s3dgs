"""
Semantic Injection Script for 3D Gaussian Splatting

This script projects 3D dense point clouds onto 2D YOLO semantic heatmaps across
multiple frames to "lift" 2D semantics into 3D space using robust multi-view fusion.

Algorithm: Max-Pooling Bayesian Multi-view Fusion
- Projects 3D points to each camera view
- Samples semantic heatmaps with spatial robustness (KxK max-pooling)
- Fuses multi-view observations using Bayesian averaging
- Outputs semantic logits for 3DGS initialization

Author: Algorithm Engineer specializing in Multi-view Geometry
Date: 2026-01-21
"""

import os
import sys
import numpy as np
import torch
import cv2
import plyfile
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import struct


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration paths and parameters for semantic injection."""

    # Input paths
    dense_pcd_path: str = r"D:\path\to\points3D.ply"  # Dense point cloud from DA3
    heatmap_dir: str = r"D:\path\to\heatmaps"  # Directory containing .npy heatmaps
    poses_path: str = r"D:\path\to\poses.npz"  # Path to poses.npz (extrinsics + intrinsics)

    # Output path
    output_dir: str = r"D:\path\to\output"  # Directory to save output .pt files

    # Semantic parameters
    num_classes: int = 4  # Number of semantic classes (U, N, D, P)
    spatial_kernel_size: int = 5  # KxK window for max-pooling (robustness to jitter)
    confidence_threshold: float = 0.3  # Gating threshold for background

    # Depth visibility parameters
    depth_tolerance: float = 0.1  # Relative tolerance for depth checking (10%)
    max_depth_range: float = 1000.0  # Maximum depth range in world units

    # Coordinate system
    coordinate_system: str = "COLMAP"  # COLMAP: X-right, Y-up, Z-backward (OpenGL-style)

    # Processing
    batch_size_points: int = 100000  # Batch size for processing points (memory efficiency)
    enable_depth_filtering: bool = True  # Enable depth-based visibility filtering


# ============================================================================
# COLMAP Binary Parsing Functions
# ============================================================================

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_char="<"):
    """Read and unpack bytes from a binary file."""
    bytes_data = fid.read(num_bytes)
    if len(bytes_data) < num_bytes:
        raise ValueError(f"Expected {num_bytes} bytes but only got {len(bytes_data)}")
    return struct.unpack(endian_char + format_char_sequence, bytes_data)


def parse_cameras_bin(path_to_file: str) -> Dict[int, Dict]:
    """
    Parse COLMAP cameras.bin file.

    Returns:
        Dictionary mapping camera_id to camera parameters
    """
    cameras = {}
    with open(path_to_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_cameras):
            camera_id = read_next_bytes(fid, 4, "I")[0]
            camera_model = read_next_bytes(fid, 4, "I")[0]
            camera_width = read_next_bytes(fid, 8, "Q")[0]
            camera_height = read_next_bytes(fid, 8, "Q")[0]

            # Number of parameters depends on camera model
            num_params_map = {
                0: 3,  # SIMPLE_PINHOLE: f, cx, cy
                1: 4,  # PINHOLE: fx, fy, cx, cy
                2: 4,  # SIMPLE_RADIAL: f, cx, cy, k
                3: 5,  # RADIAL: f, cx, cy, k1, k2
                4: 8,  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
            }

            num_params = num_params_map.get(camera_model, 4)
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)

            cameras[camera_id] = {
                'model': camera_model,
                'width': int(camera_width),
                'height': int(camera_height),
                'params': np.array(params, dtype=np.float32)
            }

    return cameras


def parse_images_bin(path_to_file: str) -> Dict[int, Dict]:
    """
    Parse COLMAP images.bin file.

    Returns:
        Dictionary mapping image_id to image data with pose
    """
    images = {}
    with open(path_to_file, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "I")[0]

            # Read quaternion (qw, qx, qy, qz) - scalar first convention
            qw, qx, qy, qz = read_next_bytes(fid, 32, "dddd")
            qvec = np.array([qw, qx, qy, qz])

            # Read translation (tx, ty, tz)
            tx, ty, tz = read_next_bytes(fid, 24, "ddd")
            tvec = np.array([tx, ty, tz], dtype=np.float32)

            # Read camera_id
            camera_id = read_next_bytes(fid, 4, "I")[0]

            # Read image name (null-terminated string)
            name = ""
            while True:
                char = fid.read(1)
                if char == b'\0' or char == b"":
                    break
                name += char.decode("utf-8", errors="ignore")

            # Skip 2D points
            num_2d_points = read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_2d_points)

            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(qvec)

            images[image_id] = {
                'camera_id': camera_id,
                'name': name,
                'qvec': qvec,
                'tvec': tvec,
                'R': R
            }

    return images


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
    ], dtype=np.float32)


def get_intrinsics_from_camera(camera: Dict) -> np.ndarray:
    """Extract 3x3 intrinsic matrix from COLMAP camera parameters."""
    model = camera['model']
    params = camera['params']

    if model == 0:  # SIMPLE_PINHOLE
        f, cx, cy = params[:3]
        fx = fy = f
    elif model == 1:  # PINHOLE
        fx, fy, cx, cy = params[:4]
    elif model == 4:  # OPENCV
        fx, fy, cx, cy = params[:4]
    else:
        # Default to PINHOLE for other models
        if len(params) >= 4:
            fx, fy, cx, cy = params[:4]
        elif len(params) >= 3:
            f, cx, cy = params[:3]
            fx = fy = f
        else:
            raise ValueError(f"Unsupported camera model: {model}")

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


# ============================================================================
# Point Cloud Loading
# ============================================================================

def load_point_cloud(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud from PLY file.

    Args:
        ply_path: Path to .ply file

    Returns:
        xyz: numpy array [N, 3] of point coordinates
        rgb: numpy array [N, 3] of point colors (optional)
    """
    print(f"Loading point cloud from: {ply_path}")

    plydata = plyfile.PlyData.read(ply_path)
    vertex = plydata['vertex']

    # Extract XYZ coordinates
    xyz = np.stack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ], axis=1).astype(np.float32)

    print(f"Loaded {xyz.shape[0]} points")

    return xyz


# ============================================================================
# Multi-view Projection and Fusion
# ============================================================================

def project_points_to_camera(
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points to camera image plane.

    Args:
        points_3d: [N, 3] array of 3D points in world coordinates
        R: [3, 3] rotation matrix (World-to-Camera)
        t: [3,] translation vector (World-to-Camera)
        K: [3, 3] intrinsic matrix

    Returns:
        pixels: [N, 2] array of (u, v) pixel coordinates
        depths: [N,] array of depth values
        visible: [N,] boolean array indicating if points are in front of camera
    """
    N = points_3d.shape[0]

    # Transform to camera coordinates: X_cam = R @ X_world + t
    points_cam = (R @ points_3d.T).T + t  # [N, 3]

    # Check visibility (points must be in front of camera)
    visible = points_cam[:, 2] > 0

    # Project to image plane: x_img = K @ X_cam
    # Normalize by depth
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = (points_cam[:, 0] * fx / points_cam[:, 2]) + cx  # [N]
    v = (points_cam[:, 1] * fy / points_cam[:, 2]) + cy  # [N]
    depths = points_cam[:, 2]  # [N]

    pixels = np.stack([u, v], axis=1)  # [N, 2]

    return pixels, depths, visible


def dilate_heatmap(heatmap: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply max-pooling dilation to heatmap for spatial robustness.

    This achieves the KxK max-pooling effect efficiently using cv2.dilate.

    Args:
        heatmap: [H, W, C] semantic heatmap
        kernel_size: Size of dilation kernel (K)

    Returns:
        dilated_heatmap: [H, W, C] dilated heatmap
    """
    H, W, C = heatmap.shape

    # Create structuring element for dilation
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )

    # Apply dilation to each channel independently
    dilated = np.zeros_like(heatmap)
    for c in range(C):
        dilated[:, :, c] = cv2.dilate(heatmap[:, :, c], kernel)

    return dilated


def sample_heatmap_with_robustness(
    pixels: np.ndarray,
    heatmap: np.ndarray,
    dilated_heatmap: np.ndarray,
    img_height: int,
    img_width: int
) -> np.ndarray:
    """
    Sample heatmap at pixel locations using KxK max-pooling (dilated heatmap).

    Args:
        pixels: [N, 2] array of (u, v) pixel coordinates
        heatmap: [H, W, C] original semantic heatmap
        dilated_heatmap: [H, W, C] dilated heatmap (max-pooled)
        img_height: Image height
        img_width: Image width

    Returns:
        semantics: [N, C] array of semantic probability vectors
    """
    N = pixels.shape[0]
    C = heatmap.shape[2]

    # Round pixel coordinates to integers
    u = np.round(pixels[:, 0]).astype(int)
    v = np.round(pixels[:, 1]).astype(int)

    # Clip to image bounds
    u = np.clip(u, 0, img_width - 1)
    v = np.clip(v, 0, img_height - 1)

    # Sample from dilated heatmap (spatial robustness)
    semantics = dilated_heatmap[v, u, :]  # [N, C]

    return semantics


def check_depth_visibility(
    depths: np.ndarray,
    confidence_threshold: float = 0.1
) -> np.ndarray:
    """
    Check depth visibility using simple statistics-based gating.

    Points with extreme depth values (outliers) are marked as invisible.

    Args:
        depths: [N,] array of depth values
        confidence_threshold: Percentile threshold for outlier rejection

    Returns:
        visible: [N,] boolean array indicating depth validity
    """
    if len(depths) == 0:
        return np.array([], dtype=bool)

    # Compute depth statistics
    median_depth = np.median(depths)
    std_depth = np.std(depths) if len(depths) > 1 else 1.0

    # Mark points far from median as invisible (potential occlusion/outlier)
    valid = np.abs(depths - median_depth) < confidence_threshold * std_depth

    return valid


def fuse_multi_view_semantics(
    semantics_list: List[np.ndarray],
    weights_list: List[np.ndarray],
    num_classes: int,
    confidence_threshold: float
) -> np.ndarray:
    """
    Fuse semantic observations from multiple views using Bayesian averaging.

    Args:
        semantics_list: List of [N, C] semantic probability arrays (one per view)
        weights_list: List of [N,] weight arrays (visibility confidence per view)
        num_classes: Number of semantic classes
        confidence_threshold: Gating threshold for low-confidence predictions

    Returns:
        fused_semantics: [N, C] fused semantic probability vectors
    """
    if len(semantics_list) == 0:
        return np.zeros((0, num_classes), dtype=np.float32)

    N = semantics_list[0].shape[0]

    # Initialize accumulators
    sum_prob = np.zeros((N, num_classes), dtype=np.float32)
    sum_weights = np.zeros(N, dtype=np.float32)

    # Accumulate probabilities from all views
    for semantics, weights in zip(semantics_list, weights_list):
        # Expand weights to [N, 1] for broadcasting
        weights_expanded = weights[:, np.newaxis]  # [N, 1]

        # Weighted probability accumulation
        sum_prob += semantics * weights_expanded  # [N, C]
        sum_weights += weights  # [N]

    # Avoid division by zero
    sum_weights = np.maximum(sum_weights, 1e-8)

    # Compute average probability (Bayesian fusion)
    avg_prob = sum_prob / sum_weights[:, np.newaxis]  # [N, C]

    # Apply gating: low-confidence points -> background (uniform distribution)
    max_prob = np.max(avg_prob, axis=1)  # [N]
    low_confidence_mask = max_prob < confidence_threshold

    # Set low-confidence points to uniform background distribution
    if np.any(low_confidence_mask):
        avg_prob[low_confidence_mask] = 1.0 / num_classes

    return avg_prob


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def load_poses_and_intrinsics(poses_path: str) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """
    Load camera poses and intrinsics from COLMAP sparse directory.

    Args:
        poses_path: Path to COLMAP sparse directory (contains cameras.bin and images.bin)

    Returns:
        images_data: Dictionary of image poses
        cameras_data: Dictionary of camera intrinsics
    """
    cameras_path = os.path.join(poses_path, "cameras.bin")
    images_path = os.path.join(poses_path, "images.bin")

    print(f"Loading camera data from: {poses_path}")

    cameras = parse_cameras_bin(cameras_path)
    images = parse_images_bin(images_path)

    print(f"Loaded {len(cameras)} cameras, {len(images)} images")

    return images, cameras


def find_heatmap_files(heatmap_dir: str) -> Dict[str, str]:
    """
    Find all heatmap .npy files in directory.

    Args:
        heatmap_dir: Directory containing .npy heatmap files

    Returns:
        Dictionary mapping basename to filepath
    """
    heatmap_files = {}

    for filename in os.listdir(heatmap_dir):
        if filename.endswith('.npy'):
            basename = os.path.splitext(filename)[0]
            heatmap_files[basename] = os.path.join(heatmap_dir, filename)

    print(f"Found {len(heatmap_files)} heatmap files")

    return heatmap_files


def inject_semantics(cfg: Config):
    """
    Main semantic injection pipeline.

    Args:
        cfg: Configuration object
    """
    print("=" * 60)
    print("Semantic Injection Pipeline")
    print("=" * 60)

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. Load point cloud
    points_3d = load_point_cloud(cfg.dense_pcd_path)
    num_points = points_3d.shape[0]

    # 2. Load camera poses and intrinsics
    images_data, cameras_data = load_poses_and_intrinsics(cfg.poses_path)

    # 3. Find heatmap files
    heatmap_files = find_heatmap_files(cfg.heatmap_dir)

    # 4. Process each view
    print("\nProcessing views...")

    # Accumulators for multi-view fusion
    all_semantics = []  # List of [N, C] arrays
    all_weights = []  # List of [N,] arrays

    for image_id, img_data in tqdm(images_data.items(), desc="Injecting semantics"):
        # Get image basename
        image_name = img_data['name']
        image_basename = os.path.splitext(os.path.basename(image_name))[0]

        # Check if corresponding heatmap exists
        if image_basename not in heatmap_files:
            continue

        heatmap_path = heatmap_files[image_basename]

        # Load heatmap
        heatmap = np.load(heatmap_path)  # [H, W, C]
        H, W, C = heatmap.shape

        # Get camera parameters
        camera_id = img_data['camera_id']
        if camera_id not in cameras_data:
            continue

        camera = cameras_data[camera_id]
        K = get_intrinsics_from_camera(camera)

        # Extract pose
        R = img_data['R']  # [3, 3]
        t = img_data['tvec']  # [3,]

        # Dilate heatmap for spatial robustness (KxK max-pooling)
        dilated_heatmap = dilate_heatmap(heatmap, cfg.spatial_kernel_size)

        # Project points to camera view
        pixels, depths, visible_front = project_points_to_camera(points_3d, R, t, K)

        # Check depth visibility (optional)
        if cfg.enable_depth_filtering:
            visible_depth = check_depth_visibility(depths[visible_front])
            # Map back to original indices
            visible = np.zeros(num_points, dtype=bool)
            visible[np.where(visible_front)[0][visible_depth]] = True
        else:
            visible = visible_front

        # Filter visible points
        if not np.any(visible):
            continue

        visible_indices = np.where(visible)[0]
        visible_pixels = pixels[visible_indices]
        visible_depths = depths[visible_indices]

        # Sample heatmap with spatial robustness
        semantics = sample_heatmap_with_robustness(
            visible_pixels,
            heatmap,
            dilated_heatmap,
            H, W
        )  # [N_vis, C]

        # Create weight array (inverse depth weighting: closer points = higher confidence)
        weights = 1.0 / (visible_depths + 1e-8)  # [N_vis]
        weights = weights / (np.max(weights) + 1e-8)  # Normalize to [0, 1]

        # Expand to full point cloud size
        full_semantics = np.zeros((num_points, C), dtype=np.float32)
        full_weights = np.zeros(num_points, dtype=np.float32)

        full_semantics[visible_indices] = semantics
        full_weights[visible_indices] = weights

        # Accumulate
        all_semantics.append(full_semantics)
        all_weights.append(full_weights)

    # 5. Fuse multi-view observations
    print(f"\nFusing semantics from {len(all_semantics)} views...")

    fused_semantics = fuse_multi_view_semantics(
        all_semantics,
        all_weights,
        cfg.num_classes,
        cfg.confidence_threshold
    )  # [N, C]

    # 6. Convert to logits for 3DGS (logit = log(prob / (1 - prob)))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    probs = np.clip(fused_semantics, epsilon, 1 - epsilon)
    logits = np.log(probs / (1 - probs))  # [N, C]

    # 7. Save output
    output_basename = os.path.splitext(os.path.basename(cfg.dense_pcd_path))[0]
    output_path = os.path.join(cfg.output_dir, f"{output_basename}_semantics.pt")

    # Convert to torch tensor
    semantics_tensor = torch.from_numpy(logits.astype(np.float32))

    # Save as dictionary
    torch.save({
        'semantics': semantics_tensor,
        'num_points': num_points,
        'num_classes': cfg.num_classes
    }, output_path)

    print(f"\nSaved semantic logits to: {output_path}")
    print(f"Shape: {semantics_tensor.shape}")
    print("=" * 60)

    # Print statistics
    print("\nSemantic Statistics:")
    for c in range(cfg.num_classes):
        class_name = ['U', 'N', 'D', 'P'][c] if cfg.num_classes == 4 else f'Class_{c}'
        mean_prob = np.mean(fused_semantics[:, c])
        max_prob = np.max(fused_semantics[:, c])
        print(f"  {class_name}: mean={mean_prob:.4f}, max={max_prob:.4f}")

    print("\nDone!")


# ============================================================================
# Command-line Interface
# ============================================================================

def main():
    """Main entry point for script execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inject 2D semantics into 3D point clouds using multi-view fusion"
    )

    parser.add_argument(
        "--dense_pcd_path",
        type=str,
        help="Path to dense point cloud .ply file",
        default=None
    )

    parser.add_argument(
        "--heatmap_dir",
        type=str,
        help="Directory containing .npy heatmap files",
        default=None
    )

    parser.add_argument(
        "--poses_path",
        type=str,
        help="Path to COLMAP sparse directory (contains cameras.bin and images.bin)",
        default=None
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for .pt files",
        default=None
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        help="Number of semantic classes",
        default=4
    )

    parser.add_argument(
        "--spatial_kernel_size",
        type=int,
        help="Kernel size for spatial max-pooling (KxK)",
        default=5
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="Confidence threshold for background gating",
        default=0.3
    )

    args = parser.parse_args()

    # Create config
    cfg = Config()

    # Override with command-line arguments if provided
    if args.dense_pcd_path:
        cfg.dense_pcd_path = args.dense_pcd_path
    if args.heatmap_dir:
        cfg.heatmap_dir = args.heatmap_dir
    if args.poses_path:
        cfg.poses_path = args.poses_path
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.num_classes:
        cfg.num_classes = args.num_classes
    if args.spatial_kernel_size:
        cfg.spatial_kernel_size = args.spatial_kernel_size
    if args.confidence_threshold:
        cfg.confidence_threshold = args.confidence_threshold

    # Validate paths
    if not os.path.exists(cfg.dense_pcd_path):
        print(f"Error: Point cloud not found: {cfg.dense_pcd_path}")
        sys.exit(1)

    if not os.path.exists(cfg.heatmap_dir):
        print(f"Error: Heatmap directory not found: {cfg.heatmap_dir}")
        sys.exit(1)

    if not os.path.exists(cfg.poses_path):
        print(f"Error: Poses directory not found: {cfg.poses_path}")
        sys.exit(1)

    # Run injection
    inject_semantics(cfg)


if __name__ == "__main__":
    main()
