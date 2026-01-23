"""
Semantic Gaussian Model for 3D Gaussian Splatting with Semantic Supervision

This module implements the core 3DGS model with semantic-aware Gaussians.
Each Gaussian has:
- Geometry: position (xyz), scale, rotation, opacity
- Appearance: spherical harmonics for view-dependent color
- Semantics: class probability distribution over K classes

Refactored to use nn.ParameterDict for compatibility with gsplat v1.5.3 DefaultStrategy.

Key Naming Convention (matches gsplat requirements):
- means: Gaussian positions
- scales: Log-space scales
- quats: Rotation quaternions
- opacities: Logit-space opacity
- sh0: SH DC component [N, 1, 3]
- shN: SH higher order components [N, K, 3]
- semantic: Semantic logits [N, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial import KDTree
import plyfile
import os
from pathlib import Path


class SemanticGaussianModel(nn.Module):
    """
    3D Gaussian Splatting model with semantic annotations.

    Refactored to use nn.ParameterDict for gsplat DefaultStrategy compatibility.

    Attributes:
        params: nn.ParameterDict containing all Gaussian parameters
        active_sh_degree: Current active SH degree
        _max_sh_degree: Maximum SH degree (default 3)
        _num_classes: Number of semantic classes
    """

    def __init__(self, sh_degree: int = 3, num_classes: int = 4):
        """
        Initialize the Semantic Gaussian Model.

        Args:
            sh_degree: Maximum spherical harmonics degree (default 3)
            num_classes: Number of semantic classes (K=4 for our dataset)
        """
        super().__init__()

        self._max_sh_degree = sh_degree
        self._num_classes = num_classes
        self.active_sh_degree = sh_degree

        # Initialize ParameterDict (will be populated in create_from_pcd)
        self.params = nn.ParameterDict()

    def create_from_pcd(
        self,
        pcd_path: str,
        spatial_lr_scale: float = 1.0,
        min_scale_threshold: float = None
    ):
        """
        Initialize Gaussian parameters from a point cloud file.

        Supports robust initialization from dense point clouds (DA3) with:
        - Adaptive scale threshold based on scene scale
        - Optional semantic prior loading from precomputed .pt files

        Args:
            pcd_path: Path to PLY or TXT file containing points with RGB
            spatial_lr_scale: Spatial learning rate scale factor
            min_scale_threshold: Minimum scale threshold (DEPRECATED, now auto-computed).
                                 If None, automatically computed as scene_radius * 1e-4.
                                 This prevents Gaussian explosion in dense clouds.
        """
        # Load point cloud
        xyz, rgb = self._load_point_cloud(pcd_path)

        num_points = xyz.shape[0]

        # 1. Geometry Initialization
        # Position: directly from point cloud
        self.params['means'] = nn.Parameter(torch.tensor(xyz, dtype=torch.float32))

        # Scale: based on KNN distances with adaptive scale floor and ceiling
        kdtree = KDTree(xyz)
        distances, _ = kdtree.query(xyz, k=4)  # k=4 because first point is itself
        mean_distances = np.mean(distances[:, 1:], axis=1)  # Skip first (self)

        # CRITICAL: Adaptive scale floor to prevent Gaussian explosion
        # For dense point clouds (500k+ points), KNN distance can be ~1e-5,
        # leading to scales that cause massive screen-space overdraw (OOM).
        # Solution: Compute scene radius and set floor proportionally.
        if min_scale_threshold is None:
            # Compute scene radius
            min_xyz = xyz.min(axis=0)
            max_xyz = xyz.max(axis=0)
            scene_radius = np.linalg.norm(max_xyz - min_xyz)
            # Set floor to 1e-4 of scene radius (very conservative)
            # For typical scenes with radius ~5-10 units, this gives ~5e-4 to 1e-3
            min_scale_threshold = max(scene_radius * 1e-4, 1e-5)  # Safety floor at 1e-5

        # CRITICAL FIX #1: Hard reduction to prevent giant Gaussians
        # For sparse point clouds (e.g., 80k points), KNN distances can be very large,
        # causing Gaussians to cover ~75% of screen → massive overdraw → OOM.
        mean_distances = mean_distances * 0.1

        # CRITICAL FIX #2: Safety clamp to enforce maximum scale
        # Ensures no Gaussian can be initialized as a "giant sphere" covering the screen.
        # Maximum of 0.01 world units prevents screen-space explosion.
        mean_distances = np.clip(mean_distances, 1e-5, 0.01)

        # Apply adaptive scale floor (only for numerical stability)
        mean_distances = np.maximum(mean_distances, min_scale_threshold)

        # Convert to log-space (inverse of exp activation)
        scales = np.tile(np.log(mean_distances)[:, np.newaxis], (1, 3))
        self.params['scales'] = nn.Parameter(torch.tensor(scales, dtype=torch.float32))

        # Rotation: initialize as unit quaternions [w, x, y, z] = [1, 0, 0, 0]
        rotations = np.zeros((num_points, 4), dtype=np.float32)
        rotations[:, 0] = 1.0  # w component
        self.params['quats'] = nn.Parameter(torch.tensor(rotations, dtype=torch.float32))

        # Opacity: initialize as 0.1 (after sigmoid)
        # Use inverse_sigmoid: log(x / (1 - x))
        opacity_init = self.inverse_sigmoid(0.1 * np.ones((num_points, 1), dtype=np.float32))
        self.params['opacities'] = nn.Parameter(torch.tensor(opacity_init, dtype=torch.float32))

        # 2. Color Initialization (Spherical Harmonics)
        # Convert RGB (0-1) to SH DC component
        sh_dc = self.RGB2SH(rgb)
        # sh0 needs shape [N, 1, 3] for gsplat
        sh_dc = sh_dc[:, np.newaxis, :]  # [N, 1, 3]
        self.params['sh0'] = nn.Parameter(torch.tensor(sh_dc, dtype=torch.float32))

        # Initialize higher-order SH coefficients as zeros
        num_sh_bases = (self._max_sh_degree + 1) ** 2 - 1  # Exclude DC
        if num_sh_bases > 0:
            sh_rest = np.zeros((num_points, num_sh_bases, 3), dtype=np.float32)
            self.params['shN'] = nn.Parameter(torch.tensor(sh_rest, dtype=torch.float32))
        else:
            # Create empty parameter for consistency
            self.params['shN'] = nn.Parameter(
                torch.zeros((num_points, 0, 3), dtype=torch.float32)
            )

        # 3. Semantic Initialization with Pre-computed Prior Support
        # Try to load precomputed semantic priors from .pt file
        semantic_init = self._load_semantic_priors(pcd_path, num_points)

        self.params['semantic'] = nn.Parameter(semantic_init.to(torch.float32))

        print(f"Initialized {num_points} Gaussians with {self._num_classes} semantic classes")
        print(f"  Adaptive scale threshold: {min_scale_threshold:.6f} (max: 0.01)")
        print(f"  Semantic prior loaded: {semantic_init.shape}")

    def _load_semantic_priors(self, pcd_path: str, num_points: int) -> torch.Tensor:
        """
        Load precomputed semantic priors from a .pt file.

        Looks for a corresponding semantic file:
        - If pcd_path is 'point.ply', looks for 'point_semantic.pt'
        - If pcd_path is 'point.ply', looks for 'point_semantic.pt'

        The .pt file should contain a dictionary with:
            {'semantics': tensor [N, K]}  # Probability distribution

        Args:
            pcd_path: Path to the point cloud file
            num_points: Number of points for shape validation

        Returns:
            Semantic logits tensor [N, num_classes] in log-space
        """
        # Generate expected semantic file path
        pcd_path_obj = Path(pcd_path)

        # Try multiple naming conventions
        possible_names = [
            pcd_path_obj.stem + "_semantic.pt",  # point_semantic.pt
            pcd_path_obj.stem + "_semantics.pt",  # point_semantics.pt
            pcd_path_obj.name.replace(".ply", "_semantic.pt").replace(".txt", "_semantic.pt"),
        ]

        semantic_file = None
        for name in possible_names:
            candidate = pcd_path_obj.parent / name
            if candidate.exists():
                semantic_file = candidate
                break

        # If no semantic file found, use random initialization
        if semantic_file is None:
            print(f"  [Semantic] No precomputed semantic file found. Using random initialization.")
            semantic_init = torch.randn(num_points, self._num_classes) * 0.01
            return semantic_init

        # Load semantic priors
        print(f"  [Semantic] Loading precomputed semantics from: {semantic_file}")

        try:
            semantic_data = torch.load(semantic_file, map_location="cpu")

            # Support both direct tensor and dict format
            if isinstance(semantic_data, dict):
                if 'semantics' in semantic_data:
                    probs = semantic_data['semantics']
                else:
                    raise KeyError("Semantic dict must contain 'semantics' key")
            elif isinstance(semantic_data, torch.Tensor):
                probs = semantic_data
            else:
                raise TypeError(f"Unsupported semantic data type: {type(semantic_data)}")

            # Validate shape
            if probs.shape[0] != num_points:
                raise ValueError(
                    f"Semantic count mismatch: {probs.shape[0]} vs {num_points} points. "
                    f"Ensure the semantic file matches the point cloud."
                )

            if probs.shape[1] != self._num_classes:
                print(
                    f"  [Semantic] Warning: Class count mismatch ({probs.shape[1]} vs {self._num_classes}). "
                    f"Truncating or padding to match model config."
                )
                # Truncate or pad to match num_classes
                if probs.shape[1] > self._num_classes:
                    probs = probs[:, :self._num_classes]
                else:
                    pad_size = self._num_classes - probs.shape[1]
                    padding = torch.zeros(probs.shape[0], pad_size)
                    probs = torch.cat([probs, padding], dim=1)
                # Renormalize
                probs = probs / probs.sum(dim=1, keepdim=True)

            # Convert probabilities to logits (log-space for numerical stability)
            # Add epsilon to prevent log(0)
            semantic_logits = torch.log(probs + 1e-6)

            print(f"  [Semantic] Loaded semantic prior: {semantic_logits.shape}")
            print(f"  [Semantic] Stats - mean: {semantic_logits.mean():.4f}, std: {semantic_logits.std():.4f}")

            return semantic_logits

        except Exception as e:
            print(f"  [Semantic] Error loading semantic file: {e}")
            print(f"  [Semantic] Falling back to random initialization.")
            semantic_init = torch.randn(num_points, self._num_classes) * 0.01
            return semantic_init

    def _load_point_cloud(self, pcd_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load point cloud from PLY or TXT file.

        Args:
            pcd_path: Path to point cloud file

        Returns:
            xyz: Point coordinates [N, 3]
            rgb: RGB colors [N, 3] in range [0, 1]
        """
        if pcd_path.endswith('.ply'):
            return self._load_ply(pcd_path)
        elif pcd_path.endswith('.txt'):
            return self._load_txt(pcd_path)
        else:
            raise ValueError(f"Unsupported file format: {pcd_path}")

    def _load_ply(self, ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load point cloud from PLY file."""
        with open(ply_path, 'rb') as f:
            plydata = plyfile.PlyData.read(f)

        vertex = plydata['vertex']

        x = np.array(vertex['x'])
        y = np.array(vertex['y'])
        z = np.array(vertex['z'])
        xyz = np.stack([x, y, z], axis=1)

        r = np.array(vertex['red']) / 255.0
        g = np.array(vertex['green']) / 255.0
        b = np.array(vertex['blue']) / 255.0
        rgb = np.stack([r, g, b], axis=1)

        return xyz, rgb

    def _load_txt(self, txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load point cloud from TXT file."""
        data = np.loadtxt(txt_path)
        xyz = data[:, :3]
        rgb = data[:, 3:6] / 255.0
        return xyz, rgb

    # ========================================================================
    # Properties & Activations (compatible with gsplat)
    # ========================================================================

    @property
    def get_xyz(self):
        """Get Gaussian centers (means)."""
        return self.params['means']

    @property
    def get_scaling(self):
        """Get Gaussian scales."""
        return torch.exp(self.params['scales'])

    @property
    def get_rotation(self):
        """Get Gaussian rotations (quats)."""
        return F.normalize(self.params['quats'], dim=1)

    @property
    def get_opacity(self):
        """Get Gaussian opacity."""
        return torch.sigmoid(self.params['opacities'])

    @property
    def get_features(self):
        """
        Get spherical harmonics features.

        Concatenates sh0 and shN and flattens for the rasterizer.
        Returns shape [N, num_features] where num_features = 3 * (1 + num_sh_bases)
        """
        sh0 = self.params['sh0']  # [N, 1, 3]
        shN = self.params['shN']  # [N, K, 3]

        # Flatten: [N, 1, 3] -> [N, 3], [N, K, 3] -> [N, K*3]
        sh0_flat = sh0.squeeze(1)  # [N, 3]
        shN_flat = shN.reshape(shN.shape[0], -1)  # [N, K*3]

        # Concatenate: [N, 3] + [N, K*3] -> [N, 3*(K+1)]
        features = torch.cat([sh0_flat, shN_flat], dim=1)
        return features

    @property
    def get_semantic(self):
        """
        Get semantic class probabilities.
        Returns softmax(semantic) to get probability distribution over K classes.
        """
        return F.softmax(self.params['semantic'], dim=1)

    # ========================================================================
    # Optimization Setup
    # ========================================================================

    def get_param_groups(
        self,
        lr_scale: float = 1.0,
        spatial_lr_scale: float = 1.0
    ) -> Dict[str, List[Dict]]:
        """
        Get parameter groups for optimizer.

        Returns a dictionary mapping parameter names to their optimizer configs.

        Args:
            lr_scale: Global learning rate scale
            spatial_lr_scale: Spatial learning rate scale

        Returns:
            Dictionary with parameter names as keys and list of param group dicts as values
        """
        param_groups = {
            'means': [
                {
                    "params": [self.params['means']],
                    "lr": 0.00016 * spatial_lr_scale * lr_scale,
                    "name": "means"
                }
            ],
            'scales': [
                {
                    "params": [self.params['scales']],
                    "lr": 0.005 * lr_scale,
                    "name": "scales"
                }
            ],
            'quats': [
                {
                    "params": [self.params['quats']],
                    "lr": 0.001 * lr_scale,
                    "name": "quats"
                }
            ],
            'opacities': [
                {
                    "params": [self.params['opacities']],
                    "lr": 0.05 * lr_scale,
                    "name": "opacities"
                }
            ],
            'sh0': [
                {
                    "params": [self.params['sh0']],
                    "lr": 0.0025 * lr_scale,
                    "name": "sh0"
                }
            ],
            'shN': [
                {
                    "params": [self.params['shN']],
                    "lr": 0.000125 * lr_scale,  # 100x smaller than DC
                    "name": "shN"
                }
            ],
            'semantic': [
                {
                    "params": [self.params['semantic']],
                    "lr": 0.01 * lr_scale,  # Higher LR for fast convergence
                    "name": "semantic"
                }
            ]
        }

        return param_groups

    # ========================================================================
    # Helper Functions
    # ========================================================================

    @staticmethod
    def inverse_sigmoid(x: np.ndarray) -> np.ndarray:
        """Inverse sigmoid function."""
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))

    @staticmethod
    def RGB2SH(rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB colors to Spherical Harmonics DC component.
        SH(0) = 0.28209479177387814
        """
        SH_C0 = 0.28209479177387814
        return (rgb - 0.5) / SH_C0

    def train_setting(self):
        """Configure model for training mode."""
        self.train()

    def eval_setting(self):
        """Configure model for evaluation mode."""
        self.eval()

    def num_gaussians(self) -> int:
        """Get the number of Gaussians in the model."""
        if 'means' not in self.params:
            return 0
        return self.params['means'].shape[0]

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"SemanticGaussianModel(\n"
            f"  num_gaussians={self.num_gaussians()},\n"
            f"  max_sh_degree={self._max_sh_degree},\n"
            f"  num_classes={self._num_classes},\n"
            f"  uses_parameter_dict=True\n"
            f")"
        )
