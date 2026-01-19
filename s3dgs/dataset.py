"""
Tomato Dataset for gsplat v1.5.3

This module provides a PyTorch Dataset class for loading RGB images, camera parameters,
and semantic heatmaps for tomato plant phenotype analysis.

COLMAP Parsing:
- Implements lightweight binary parser for cameras.bin and images.bin
- Returns World-to-Camera matrices and intrinsics compatible with gsplat
"""

import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional
import warnings


# ============================================================================
# COLMAP Binary Parsing Functions
# ============================================================================

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_char="<"):
    """
    Read and unpack bytes from a binary file.

    Args:
        fid: File object
        num_bytes: Number of bytes to read
        format_char_sequence: Format characters for struct.unpack
        endian_char: Endianness character ('<' for little-endian)

    Returns:
        Tuple of unpacked values
    """
    bytes_data = fid.read(num_bytes)
    return struct.unpack(endian_char + format_char_sequence, bytes_data)


def parse_cameras_bin(path_to_file: str) -> Dict[int, Dict]:
    """
    Parse COLMAP cameras.bin file.

    COLMAP binary format:
        - num_cameras (8 bytes, uint64)
        For each camera:
            - camera_id (4 bytes, uint32)
            - model (4 bytes, uint32)
            - width (8 bytes, uint64)
            - height (8 bytes, uint64)
            - params (N * 8 bytes, doubles)

    Camera model types:
        0: SIMPLE_PINHOLE
        1: PINHOLE
        2: SIMPLE_RADIAL
        3: RADIAL
        4: OPENCV
        5: OPENCV_FISHEYE
        6: FULL_OPENCV
        7: FOV
        8: SIMPLE_RADIAL_FISHEYE
        9: RADIAL_FISHEYE
        10: THIN_PRISM_FISHEYE

    Args:
        path_to_file: Path to cameras.bin

    Returns:
        Dictionary mapping camera_id to camera parameters:
        {
            camera_id: {
                'model': int,
                'width': int,
                'height': int,
                'params': array of parameters
            }
        }
    """
    cameras = {}
    with open(path_to_file, "rb") as fid:
        # Read number of cameras (uint64)
        num_cameras = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_cameras):
            # Read camera_id (uint32)
            camera_id = read_next_bytes(fid, 4, "I")[0]
            # Read model (uint32)
            camera_model = read_next_bytes(fid, 4, "I")[0]
            # Read width (uint64)
            camera_width = read_next_bytes(fid, 8, "Q")[0]
            # Read height (uint64)
            camera_height = read_next_bytes(fid, 8, "Q")[0]

            # Number of parameters depends on camera model
            num_params_map = {
                0: 3,  # SIMPLE_PINHOLE: f, cx, cy
                1: 4,  # PINHOLE: fx, fy, cx, cy
                2: 4,  # SIMPLE_RADIAL: f, cx, cy, k
                3: 5,  # RADIAL: f, cx, cy, k1, k2
                4: 8,  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
                5: 8,  # OPENCV_FISHEYE
                6: 12, # FULL_OPENCV
                7: 3,  # FOV
                8: 4,  # SIMPLE_RADIAL_FISHEYE
                9: 5,  # RADIAL_FISHEYE
                10: 12 # THIN_PRISM_FISHEYE
            }

            num_params = num_params_map.get(camera_model, 4)

            # Read parameters (N * 8 bytes, doubles)
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)

            cameras[camera_id] = {
                'model': camera_model,
                'width': camera_width,
                'height': camera_height,
                'params': np.array(params)
            }

    return cameras


def parse_images_bin(path_to_file: str) -> Dict[int, Dict]:
    """
    Parse COLMAP images.bin file.

    COLMAP images.bin format:
        - num_images (8 bytes, uint64)
        For each image:
            - image_id (4 bytes, uint32)
            - qw, qx, qy, qz (4 * 8 bytes, doubles) - quaternion (scalar-first)
            - tx, ty, tz (3 * 8 bytes, doubles) - translation vector
            - camera_id (4 bytes, uint32)
            - image_name (variable length, null-terminated string)
            - num_2d_points (8 bytes, uint64)
            - 2D points data (24 bytes each: x, y, point3d_id)

    Note: COLMAP stores data compactly without extra padding between fields.

    COLMAP stores World-to-Camera poses as:
        X_cam = R * X_world + T

    Args:
        path_to_file: Path to images.bin

    Returns:
        Dictionary mapping image_id to image data:
        {
            image_id: {
                'camera_id': int,
                'name': str (relative path to image),
                'qvec': np.ndarray (4,) quaternion [qw, qx, qy, qz],
                'tvec': np.ndarray (3,) translation [tx, ty, tz],
                'R': np.ndarray (3, 3) rotation matrix
            }
        }
    """
    images = {}
    with open(path_to_file, "rb") as fid:
        # Read number of images (uint64)
        num_images = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_images):
            # Read image_id (4 bytes, uint32)
            image_id = read_next_bytes(fid, 4, "I")[0]

            # Read quaternion (qw, qx, qy, qz) - scalar first convention
            qw, qx, qy, qz = read_next_bytes(fid, 32, "dddd")
            qvec = np.array([qw, qx, qy, qz])

            # Read translation (tx, ty, tz)
            tx, ty, tz = read_next_bytes(fid, 24, "ddd")
            tvec = np.array([tx, ty, tz], dtype=np.float32)

            # Read camera_id (4 bytes, uint32)
            camera_id = read_next_bytes(fid, 4, "I")[0]

            # Read image name (null-terminated string)
            name = ""
            while True:
                char = fid.read(1)
                if char == b'\0' or char == b"":
                    break
                name += char.decode("utf-8", errors="ignore")

            # Read 2D points (skip for our use case)
            num_2d_points = read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_2d_points)  # Skip 2D points: x, y, point3d_id (8+8+8 bytes each)

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
    """
    Convert quaternion to rotation matrix.

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        3x3 rotation matrix
    """
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])


def get_intrinsics_from_camera(camera: Dict) -> np.ndarray:
    """
    Extract 3x3 intrinsic matrix from COLMAP camera parameters.

    Args:
        camera: Camera dictionary from parse_cameras_bin

    Returns:
        3x3 intrinsic matrix K
    """
    model = camera['model']
    params = camera['params']
    width = camera['width']
    height = camera['height']

    # PINHOLE model (model=1): params = [fx, fy, cx, cy]
    # SIMPLE_PINHOLE model (model=0): params = [f, cx, cy]
    # OPENCV model (model=4): params = [fx, fy, cx, cy, k1, k2, p1, p2]

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
            raise ValueError(f"Unsupported camera model: {model} with params: {params}")

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


# ============================================================================
# Tomato Dataset Class
# ============================================================================

class TomatoDataset(Dataset):
    """
    PyTorch Dataset for loading RGB images, camera parameters, semantic heatmaps,
    and depth maps for tomato plant phenotype analysis with gsplat v1.5.3.

    Expected directory structure:
        colmap_dir/
            sparse/
                0/
                    cameras.bin
                    images.bin
        images_dir/
            image_001.jpg
            image_002.jpg
            ...
        heatmap_dir/
            image_001.npy
            image_002.npy
            ...
        confidence_path: confidence.json
        depth_dir/ (optional)
            image_001.png
            image_002.png
            ...

    The confidence JSON should have the format:
        {
            "image_001": [conf_U, conf_N, conf_D, conf_P],
            "image_002": [conf_U, conf_N, conf_D, conf_P],
            ...
        }
    """

    def __init__(
        self,
        colmap_dir: str,
        images_dir: str,
        heatmap_dir: str,
        confidence_path: str,
        depth_dir: str = None,
        split: str = "train"
    ):
        """
        Initialize the TomatoDataset.

        Args:
            colmap_dir: Path to COLMAP output directory (contains sparse/0/)
            images_dir: Path to RGB images directory
            heatmap_dir: Path to heatmap .npy files directory
            confidence_path: Path to confidence JSON file
            depth_dir: Path to depth map .png files directory (optional)
            split: Dataset split ("train" or "val"), currently unused
        """
        self.colmap_dir = colmap_dir
        self.images_dir = images_dir
        self.heatmap_dir = heatmap_dir
        self.depth_dir = depth_dir
        self.split = split

        # Parse COLMAP data
        sparse_dir = os.path.join(colmap_dir)
        if not os.path.exists(sparse_dir):
            raise FileNotFoundError(f"COLMAP sparse directory not found: {sparse_dir}")

        cameras_path = os.path.join(sparse_dir, "cameras.bin")
        images_path = os.path.join(sparse_dir, "images.bin")

        print(f"Parsing COLMAP data from: {colmap_dir}")
        self.cameras = parse_cameras_bin(cameras_path)
        self.images_data = parse_images_bin(images_path)
        print(f"Found {len(self.cameras)} cameras, {len(self.images_data)} images")

        # Load confidence data
        with open(confidence_path, 'r') as f:
            self.confidence_dict = json.load(f)
        print(f"Loaded confidence data for {len(self.confidence_dict)} frames")

        # Validate depth directory if provided
        if depth_dir is not None:
            if not os.path.exists(depth_dir):
                warnings.warn(f"Depth directory not found: {depth_dir}, disabling depth loading")
                self.depth_dir = None

        # Match and cache all data
        self._cache_data()

    def _cache_data(self):
        """
        Pre-load and cache all data in memory for faster training.
        Matches COLMAP images with heatmaps and confidence records.
        """
        self.cached_data = []

        for image_id, img_data in self.images_data.items():
            image_name = img_data['name']
            # Remove path prefix if present
            image_basename = os.path.basename(image_name)
            name_without_ext = os.path.splitext(image_basename)[0]

            # Check if corresponding heatmap exists
            heatmap_path = os.path.join(self.heatmap_dir, name_without_ext + ".npy")
            if not os.path.exists(heatmap_path):
                warnings.warn(f"Heatmap not found for image: {image_basename}, skipping")
                continue

            # Check if confidence data exists
            if name_without_ext not in self.confidence_dict:
                warnings.warn(f"Confidence data not found for: {name_without_ext}, skipping")
                continue

            # Check if depth map exists (if depth_dir is provided)
            has_depth = False
            if self.depth_dir is not None:
                depth_path = os.path.join(self.depth_dir, name_without_ext + ".png")
                has_depth = os.path.exists(depth_path)
                if not has_depth:
                    warnings.warn(f"Depth map not found for: {image_basename}, will skip depth for this frame")

            # Get camera parameters
            camera_id = img_data['camera_id']
            if camera_id not in self.cameras:
                warnings.warn(f"Camera {camera_id} not found for image: {image_basename}, skipping")
                continue

            camera = self.cameras[camera_id]

            # Store all necessary data
            self.cached_data.append({
                'image_id': image_id,
                'image_name': image_basename,
                'name_without_ext': name_without_ext,
                'camera': camera,
                'R': img_data['R'],
                'tvec': img_data['tvec'],
                'camera_id': camera_id,
                'has_depth': has_depth
            })

        print(f"Successfully cached {len(self.cached_data)} valid frames")

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.

        Returns:
            Dictionary containing:
                - image: Tensor [H, W, 3], RGB in [0, 1]
                - heatmap: Tensor [H, W, K], semantic heatmaps in [0, 1]
                - confidence: Tensor [K], per-class confidence scores
                - viewmat: Tensor [4, 4], World-to-Camera matrix
                - K: Tensor [3, 3], camera intrinsics
                - height: int
                - width: int
                - image_id: int
        """
        data = self.cached_data[idx]
        name_without_ext = data['name_without_ext']

        # Load RGB image
        image_path = os.path.join(self.images_dir, data['image_name'])
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        H, W = image_np.shape[:2]
        image_tensor = torch.from_numpy(image_np)  # [H, W, 3]

        # Load heatmap
        heatmap_path = os.path.join(self.heatmap_dir, name_without_ext + ".npy")
        heatmap_np = np.load(heatmap_path).astype(np.float32)  # [H, W, K]
        heatmap_tensor = torch.from_numpy(heatmap_np)  # [H, W, K]

        # Load confidence
        conf_vec = self.confidence_dict[name_without_ext]
        confidence_tensor = torch.tensor(conf_vec, dtype=torch.float32)  # [K]

        # Load depth map if available
        depth_tensor = None
        if data['has_depth'] and self.depth_dir is not None:
            depth_path = os.path.join(self.depth_dir, name_without_ext + ".png")
            try:
                # Load 16-bit PNG and normalize to [0, 1]
                depth_16bit = np.array(Image.open(depth_path), dtype=np.float32)
                depth_np = depth_16bit / 65535.0  # Convert from [0, 65535] to [0, 1]
                depth_tensor = torch.from_numpy(depth_np)  # [H, W]
            except Exception as e:
                warnings.warn(f"Failed to load depth map for {name_without_ext}: {e}")
                depth_tensor = None

        # Get camera intrinsics
        K_np = get_intrinsics_from_camera(data['camera'])
        K_tensor = torch.from_numpy(K_np)  # [3, 3]

        # Build World-to-Camera matrix (4x4) for gsplat
        # COLMAP convention: X_cam = R * X_world + T
        # gsplat expects OpenCV convention: Right-Down-Forward
        R = data['R']  # [3, 3]
        T = data['tvec']  # [3,]

        # Construct 4x4 view matrix
        # [R00 R01 R02 Tx]
        # [R10 R11 R12 Ty]
        # [R20 R21 R22 Tz]
        # [0   0   0   1 ]
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = T

        viewmat_tensor = torch.from_numpy(viewmat)  # [4, 4]

        result = {
            'image': image_tensor,
            'heatmap': heatmap_tensor,
            'confidence': confidence_tensor,
            'viewmat': viewmat_tensor,
            'K': K_tensor,
            'height': H,
            'width': W,
            'image_id': data['image_id']
        }

        # Add depth tensor if available
        if depth_tensor is not None:
            result['depth'] = depth_tensor

        return result


# ============================================================================
# Utility Functions
# ============================================================================

def create_dataloader(
    colmap_dir: str,
    images_dir: str,
    heatmap_dir: str,
    confidence_path: str,
    depth_dir: str = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the TomatoDataset.

    Args:
        colmap_dir: Path to COLMAP output directory
        images_dir: Path to RGB images directory
        heatmap_dir: Path to heatmap .npy files directory
        confidence_path: Path to confidence JSON file
        depth_dir: Path to depth map .png files directory (optional)
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader instance
    """
    dataset = TomatoDataset(
        colmap_dir=colmap_dir,
        images_dir=images_dir,
        heatmap_dir=heatmap_dir,
        confidence_path=confidence_path,
        depth_dir=depth_dir
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
