"""
Rendering Functions for Semantic 3D Gaussian Splatting

This module contains rendering-related functions:
- Dual-pass rendering for RGB and semantic channels
- Depth rendering for depth supervision
- Helper functions for multi-channel semantic rendering
"""

import torch
from typing import Dict
from gsplat.rendering import rasterization


def render_dual_pass(
    model,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = 3,
    render_depth: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Perform dual-pass rendering for RGB and Semantic channels.

    Pass 1 (RGB): Render using SH features for view-dependent color.
    Pass 2 (Semantic): Render semantic probabilities using the same geometry.
    Pass 3 (Depth): Render depth map if enabled (for depth supervision).

    Args:
        model: SemanticGaussianModel
        viewmat: Camera view matrix [4, 4]
        K: Camera intrinsics [3, 3]
        width: Image width
        height: Image height
        sh_degree: Spherical harmonics degree
        render_depth: Whether to render depth map

    Returns:
        Dictionary containing:
            - rgb: Rendered RGB image [H, W, 3]
            - semantic: Rendered semantic probabilities [H, W, K]
            - depth: Rendered depth [H, W] (if render_depth=True)
            - alpha: Alpha channel [H, W]
            - meta: Intermediate results from Pass 1
    """
    device = model.params['means'].device

    # Prepare model parameters using gsplat naming convention
    # For packed=True: use flat [N, ...] tensors, not [1, N, ...]
    means = model.params['means']  # [N, 3]
    quats = model.params['quats']  # [N, 4]
    scales = model.params['scales']  # [N, 3]
    # IMPORTANT: opacities are stored in inverse_sigmoid space, need to apply sigmoid
    opacities = torch.sigmoid(model.params['opacities']).squeeze()  # [N]

    # ========================================================================
    # Pass 1: RGB Rendering
    # ========================================================================
    # Get SH features [N, K, 3] where K = (sh_degree + 1)^2
    features = model.get_features  # [N, num_features]
    num_features = features.shape[1]
    num_sh_bases = num_features // 3

    # Reshape to [1, N, K, 3] for rasterization (gsplat expects [..., (C,) N, K, 3])
    colors_sh = features.view(1, -1, num_sh_bases, 3)  # [1, N, K, 3]

    # Render RGB with packed=True for performance
    # For packed mode: viewmats=[C, 4, 4], Ks=[C, 3, 3] where C=1 for single camera
    rgb, alpha, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors_sh,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=True,  # Revert to packed=True for performance
        absgrad=True  # AbsGS: use absolute gradients for densification
    )

    # rgb: [1, H, W, 3] -> [H, W, 3] (packed mode output shape)
    rgb = rgb.squeeze(0)
    # alpha: [1, H, W, 1] -> [H, W]
    alpha = alpha.squeeze(0).squeeze(-1)

    # ========================================================================
    # Pass 2: Depth Rendering (if enabled)
    # ========================================================================
    depth_map = None
    if render_depth:
        depth_map = _render_depth_map(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            viewmat=viewmat,
            K=K,
            width=width,
            height=height
        )

    # ========================================================================
    # Pass 3: Semantic Rendering
    # ========================================================================
    sem_map = _render_semantic_map(
        model=model,
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        viewmat=viewmat,
        K=K,
        width=width,
        height=height,
        device=device
    )

    # Retain gradients for densification
    meta['means2d'].retain_grad()

    result = {
        'rgb': rgb,        # [H, W, 3]
        'semantic': sem_map,  # [H, W, 4]
        'alpha': alpha,     # [H, W]
        'meta': meta        # Dict containing intermediate results
    }

    # Add depth map if rendered
    if depth_map is not None:
        result['depth'] = depth_map

    return result


def _render_depth_map(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int
) -> torch.Tensor:
    """
    Render depth map using z-coordinate as color.

    Args:
        means: Gaussian centers [N, 3]
        quats: Rotation quaternions [N, 4]
        scales: Gaussian scales [N, 3]
        opacities: Gaussian opacities [N]
        viewmat: Camera view matrix [4, 4]
        K: Camera intrinsics [3, 3]
        width: Image width
        height: Image height

    Returns:
        Depth map [H, W]
    """
    # Extract depth (z-coordinate in camera space) from means
    # Transform means to camera space: X_cam = R * X_world + T
    viewmat_expanded = viewmat[None]  # [1, 4, 4]
    means_homo = torch.cat([means, torch.ones_like(means[:, :1])], dim=1)  # [N, 4]
    means_cam = (viewmat_expanded @ means_homo.T).T[:, :3]  # [N, 3]
    z_values = means_cam[:, 2:3]  # [N, 1] - depth in camera space

    # Normalize z-values to [0, 1] range for rendering
    z_min = z_values.min()
    z_max = z_values.max()
    if z_max > z_min:
        z_normalized = (z_values - z_min) / (z_max - z_min)  # [N, 1]
    else:
        z_normalized = torch.zeros_like(z_values)

    # Render as single-channel "color"
    colors_depth = z_normalized.repeat(1, 3)[None]  # [1, N, 3]

    depth_render, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors_depth,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        width=width,
        height=height,
        sh_degree=None,  # No SH for depth
        packed=True,
        absgrad=True
    )

    # depth_render: [1, H, W, 3] -> [H, W] (take R channel)
    depth_map = depth_render.squeeze(0)[..., 0]  # [H, W]

    # Denormalize back to actual depth values
    if z_max > z_min:
        depth_map = depth_map * (z_max - z_min) + z_min

    return depth_map


def _render_semantic_map(
    model,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device
) -> torch.Tensor:
    """
    Render semantic probabilities using multiplexing strategy.

    Since gsplat's rasterization only supports 3-channel RGB output, we use
    a multiplexing strategy to render 4 semantic channels:
    - Batch 1: Channels 0-2 (U, N, D) as RGB
    - Batch 2: Channel 3 (P) + zeros

    Args:
        model: SemanticGaussianModel
        means: Gaussian centers [N, 3]
        quats: Rotation quaternions [N, 4]
        scales: Gaussian scales [N, 3]
        opacities: Gaussian opacities [N]
        viewmat: Camera view matrix [4, 4]
        K: Camera intrinsics [3, 3]
        width: Image width
        height: Image height
        device: Torch device

    Returns:
        Semantic map [H, W, 4]
    """
    # Get semantic probabilities [N, K]
    sem_probs = model.get_semantic  # [N, K]
    K_sem = sem_probs.shape[1]  # Should be 4

    # Multiplexing strategy: pad to 6 channels (2 sets of RGB)
    if K_sem == 4:
        # Pad to [N, 6]
        sem_padded = torch.zeros(sem_probs.shape[0], 6, device=device)
        sem_padded[:, :3] = sem_probs[:, :3]  # U, N, D
        sem_padded[:, 3] = sem_probs[:, 3]   # P

        # Batch 1: Channels 0-2 (U, N, D) as RGB
        sem_batch1 = sem_padded[:, :3][None]  # [1, N, 3]

        # Batch 2: Channel 3 (P) + zeros
        sem_batch2 = sem_padded[:, 3:6][None]  # [1, N, 3]

        # Render both batches with packed=True
        sem_render1, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sem_batch1,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=width,
            height=height,
            sh_degree=None,
            packed=True,
            absgrad=True  # AbsGS: use absolute gradients for densification
        )

        sem_render2, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sem_batch2,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=width,
            height=height,
            sh_degree=None,
            packed=True,
            absgrad=True  # AbsGS: use absolute gradients for densification
        )

        # sem_render1: [1, H, W, 3] -> [H, W, 3] (packed mode output shape)
        sem_render1 = sem_render1.squeeze(0)
        # sem_render2: [1, H, W, 3] -> [H, W, 3]
        sem_render2 = sem_render2.squeeze(0)

        # Concatenate to get [H, W, 4]
        sem_map = torch.cat([
            sem_render1,     # [H, W, 3] (U, N, D)
            sem_render2[..., 0:1]  # [H, W, 1] (P)
        ], dim=-1)  # [H, W, 4]

    else:
        raise ValueError(f"Expected K_sem=4, got {K_sem}")

    return sem_map
