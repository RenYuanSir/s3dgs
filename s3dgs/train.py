"""
Training Script for Semantic 3D Gaussian Splatting

This script implements dual-pass rendering to train both RGB reconstruction
and semantic field prediction simultaneously.

Key Features:
- Dual-Pass Rendering: RGB + Semantic channels
- Confidence-Gated Semantic Loss with Foreground Weighting
- Scheduled Semantic Learning (warm-up + enable)
- Official gsplat DefaultStrategy for densification

Refactored for gsplat v1.5.3 DefaultStrategy compatibility.
"""

import os
import sys

# ========================================================================
# Auto-add project root to sys.path for imports
# ========================================================================
# This allows running the script directly from anywhere:
#   python s3dgs/train.py
#   python -m s3dgs.train
#   python train.py (if moved to root)
_current_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    print(f"[INFO] Added project root to sys.path: {_project_root}")

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

# Import gsplat components
import gsplat
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

# Import our modules (now works from any directory)
from s3dgs.model import SemanticGaussianModel
from s3dgs.dataset import TomatoDataset, create_dataloader


# ============================================================================
# Loss Functions
# ============================================================================

def scale_invariant_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Scale-invariant depth loss for monocular depth priors.

    Since monocular depth is relative (up to scale and shift), we solve for the
    optimal scale (s) and shift (t) that align prediction with ground truth:
        s * pred_depth + t ≈ gt_depth

    Then compute L1 loss on the aligned depth.

    Args:
        pred_depth: Rendered depth from 3DGS [H, W] (can be any scale)
        gt_depth: Monocular depth prior [H, W] (relative depth, normalized [0, 1])
        mask: Valid pixel mask [H, W] (optional, defaults to all valid pixels)

    Returns:
        Scalar loss value

    Reference:
        Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", 2014
    """
    # Flatten spatial dimensions
    pred_flat = pred_depth.reshape(-1)  # [H*W]
    gt_flat = gt_depth.reshape(-1)      # [H*W]

    # Create mask for valid pixels (non-inf, non-nan)
    if mask is None:
        valid_mask = torch.isfinite(pred_flat) & torch.isfinite(gt_flat)
    else:
        valid_mask = mask.reshape(-1) & torch.isfinite(pred_flat) & torch.isfinite(gt_flat)

    # Filter valid pixels
    pred_valid = pred_flat[valid_mask]
    gt_valid = gt_flat[valid_mask]

    if pred_valid.numel() < 10:
        # Not enough valid pixels, return zero loss
        return torch.tensor(0.0, device=pred_depth.device)

    # Solve for optimal scale (s) and shift (t) using least squares
    # We want to minimize: || s * pred + t - gt ||^2
    # This is a linear regression problem

    # Design matrix: [pred, 1]  -> [N, 2]
    A = torch.stack([pred_valid, torch.ones_like(pred_valid)], dim=1)  # [N, 2]

    # Target: gt  -> [N]
    b = gt_valid  # [N]

    # Solve: (A^T A)^{-1} A^T b
    # Using pseudo-inverse for numerical stability
    try:
        # A^T A: [2, N] @ [N, 2] -> [2, 2]
        ATA = A.T @ A
        # A^T b: [2, N] @ [N] -> [2]
        ATb = A.T @ b

        # Solve linear system
        params = torch.linalg.solve(ATA, ATb)  # [2]
        s, t = params[0], params[1]
    except RuntimeError:
        # Fallback if matrix is singular
        s = torch.tensor(1.0, device=pred_depth.device)
        t = torch.tensor(0.0, device=pred_depth.device)

    # Align prediction
    pred_aligned = s * pred_flat + t

    # Compute L1 loss on aligned depth (only on valid pixels)
    loss = torch.abs(pred_aligned[valid_mask] - gt_valid).mean()

    return loss


def semantic_loss_with_gating(
    pred_sem: torch.Tensor,
    gt_heatmap: torch.Tensor,
    confidence: torch.Tensor,
    fg_weight: float = 20.0
) -> torch.Tensor:
    """
    Compute semantic loss with confidence gating and foreground weighting.

    Args:
        pred_sem: Predicted semantic probabilities [H, W, K]
        gt_heatmap: Ground truth heatmap [H, W, K]
        confidence: Per-class confidence [K]
        fg_weight: Foreground pixel weight multiplier

    Returns:
        Scalar loss value
    """
    H, W, K = pred_sem.shape

    # 1. Compute MSE loss
    loss = (pred_sem - gt_heatmap) ** 2  # [H, W, K]

    # 2. Foreground weighting
    fg_mask = (gt_heatmap > 0.1).float()  # [H, W, K]
    loss = loss * (1.0 + fg_weight * fg_mask)  # [H, W, K]

    # 3. Confidence gating
    valid_mask = (confidence > 0.5).float()  # [K]
    valid_mask = valid_mask.view(1, 1, K)  # [1, 1, K]

    # Apply mask and compute mean
    loss = (loss * valid_mask).mean()

    return loss


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 loss for RGB reconstruction.

    Args:
        pred: Predicted RGB [H, W, 3]
        target: Ground truth RGB [H, W, 3]

    Returns:
        Scalar loss value
    """
    return torch.abs(pred - target).mean()


# ============================================================================
# Dual-Pass Rendering
# ============================================================================

def render_dual_pass(
    model: SemanticGaussianModel,
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
        # Render depth using z-coordinate as color
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

    # ========================================================================
    # Pass 3: Semantic Rendering
    # ========================================================================
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
            packed=True,  # Revert to packed=True
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
            packed=True,  # Revert to packed=True
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


# ============================================================================
# Training Loop
# ============================================================================

def train(
    # Data paths
    colmap_dir: str,
    images_dir: str,
    heatmap_dir: str,
    confidence_path: str,
    pcd_path: str,
    depth_dir: str = None,  # New: path to depth maps

    # Training settings
    num_iterations: int = 7000,
    warmup_iterations: int = 4000,  # "Geometry First": RGB builds cylindrical volume before semantic classification
    lambda_sem: float = 0.05,
    lambda_depth: float = 0.1,  # New: depth loss weight
    lr_scale: float = 1.0,
    spatial_lr_scale: float = 1.0,

    # Model settings
    sh_degree: int = 3,
    num_classes: int = 4,

    # Loss weights
    fg_weight: float = 20.0,

    # Performance optimization
    resolution_scale: float = 0.75,  # Full resolution for skeleton supervision (no downscaling)

    # Logging
    log_every: int = 100,
    save_every: int = 1000,
    output_dir: str = "./output",

    # Device
    device: str = "cuda"
):
    """
    Main training loop for semantic 3D Gaussian Splatting with depth supervision.

    Args:
        colmap_dir: Path to COLMAP output
        images_dir: Path to RGB images
        heatmap_dir: Path to heatmap .npy files
        confidence_path: Path to confidence JSON
        pcd_path: Path to COLMAP point cloud (.ply)
        depth_dir: Path to depth map .png files (optional)
        num_iterations: Total training iterations
        warmup_iterations: Iterations for geometric warm-up (lambda_sem=0)
        lambda_sem: Semantic loss weight after warm-up
        lambda_depth: Depth loss weight (scale-invariant)
        lr_scale: Global learning rate scale
        spatial_lr_scale: Spatial learning rate scale
        sh_degree: Spherical harmonics degree
        num_classes: Number of semantic classes
        fg_weight: Foreground pixel weight for semantic loss
        log_every: Log interval
        save_every: Save checkpoint interval
        output_dir: Output directory for checkpoints
        device: Device to use
    """
    print("="*60)
    print("Semantic 3D Gaussian Splatting Training")
    print("="*60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================================
    # Initialize Model
    # ========================================================================
    print("\nInitializing model...")
    model = SemanticGaussianModel(sh_degree=sh_degree, num_classes=num_classes)
    model.create_from_pcd(pcd_path, spatial_lr_scale=spatial_lr_scale)
    model = model.to(device)

    print(f"Model: {model.num_gaussians()} Gaussians")

    # ========================================================================
    # Initialize Dataset and DataLoader
    # ========================================================================
    print("\nInitializing dataset...")
    dataset = TomatoDataset(
        colmap_dir=colmap_dir,
        images_dir=images_dir,
        heatmap_dir=heatmap_dir,
        confidence_path=confidence_path,
        depth_dir=depth_dir  # Pass depth directory
    )

    dataloader = create_dataloader(
        colmap_dir=colmap_dir,
        images_dir=images_dir,
        heatmap_dir=heatmap_dir,
        confidence_path=confidence_path,
        depth_dir=depth_dir,  # Pass depth directory
        batch_size=1,
        num_workers=0,  # Windows compatibility: must be 0 to avoid multiprocessing issues
        shuffle=True
    )

    # Check if depth data is available
    has_depth = depth_dir is not None and any(data.get('has_depth', False) for data in dataset.cached_data)
    print(f"Dataset: {len(dataset)} frames")
    print(f"Depth supervision: {'Enabled' if has_depth else 'Disabled'}")

    # ========================================================================
    # Initialize Optimizers (Dictionary for DefaultStrategy)
    # ========================================================================
    print("\nInitializing optimizers...")
    param_groups = model.get_param_groups(
        lr_scale=lr_scale,
        spatial_lr_scale=spatial_lr_scale
    )

    # Create dictionary of optimizers (one per parameter group)
    optimizers = {}
    for param_name, groups in param_groups.items():
        optimizers[param_name] = torch.optim.Adam(groups)

    print(f"Created {len(optimizers)} optimizers")

    # ========================================================================
    # Initialize DefaultStrategy
    # ========================================================================
    print("\nInitializing DefaultStrategy...")
    strategy = DefaultStrategy(
        refine_every=100,
        grow_grad2d=0.00005,       # 5e-5: Recalibrated for skeleton supervision (stronger signal over lines)
        grow_scale3d=0.005,        # Split features > 0.025 units (5.0 * 0.005 = 0.025), enforce high density for stem volume
        prune_opa=0.005,
        prune_scale3d=3.0,         # Tolerance radius ~15.0 units (5.0 * 3.0 = 15.0)
        refine_start_iter=500,
        refine_stop_iter=15000,
        reset_every=3000,
        absgrad=True,              # AbsGS: use absolute gradients for robust densification
        verbose=True  # Enable verbose to see densification activity
    )

    # Check sanity
    strategy.check_sanity(model.params, optimizers)
    print("DefaultStrategy sanity check passed")

    # Initialize strategy state with object core scale
    strategy_state = strategy.initialize_state(scene_scale=5.0)
    print("Strategy state initialized with scene_scale=5.0 (object core radius)")

    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\nStarting training...")
    print(f"Total iterations: {num_iterations}")
    print(f"Warm-up iterations: {warmup_iterations} (lambda_sem=0)")
    print(f"Lambda_sem after warm-up: {lambda_sem}")
    if has_depth:
        print(f"Lambda_depth: {lambda_depth}")

    iteration = 0
    train_losses = []

    start_time = time.time()

    # Convert dataloader to infinite iterator
    data_iter = iter(dataloader)

    for iteration in range(num_iterations):
        try:
            data = next(data_iter)
        except StopIteration:
            # Reload dataloader
            data_iter = iter(dataloader)
            data = next(data_iter)

        # Extract data (still on CPU)
        # DataLoader adds batch dimension, so we need to squeeze it
        image = data['image'].squeeze(0)  # [H, W, 3], CPU
        heatmap = data['heatmap'].squeeze(0)  # [H, W, K], CPU
        confidence = data['confidence'].squeeze(0)  # [K], CPU
        viewmat = data['viewmat'].squeeze(0)  # [4, 4], CPU
        K = data['K'].clone().squeeze(0)  # [3, 3], CPU (clone to avoid in-place modification)

        # Extract depth if available
        gt_depth = None
        if 'depth' in data:
            gt_depth = data['depth'].squeeze(0)  # [H, W], CPU

        # ====================================================================
        # Apply Resolution Scaling (Performance Optimization)
        # ====================================================================
        # resolution_scale=0.75 provides good balance between quality and speed
        # Reduces pixels by ~44% (0.75^2 = 0.5625), resulting in ~2x speedup
        if resolution_scale != 1.0:
            import torch.nn.functional as F

            # Scale RGB: [H, W, 3] -> [H*r, W*r, 3]
            image = F.interpolate(
                image.permute(2, 0, 1)[None],  # [1, 3, H, W]
                scale_factor=resolution_scale,
                mode='bilinear',
                align_corners=False
            )[0].permute(1, 2, 0)  # [H*r, W*r, 3]

            # Scale Heatmap: [H, W, K] -> [H*r, W*r, K]
            heatmap = F.interpolate(
                heatmap.permute(2, 0, 1)[None],  # [1, K, H, W]
                scale_factor=resolution_scale,
                mode='bilinear',
                align_corners=False
            )[0].permute(1, 2, 0)  # [H*r, W*r, K]

            # Scale intrinsics: adjust fx, fy, cx, cy
            K[0, 0] *= resolution_scale  # fx
            K[1, 1] *= resolution_scale  # fy
            K[0, 2] *= resolution_scale  # cx
            K[1, 2] *= resolution_scale  # cy

        # Move to device
        image = image.to(device)
        heatmap = heatmap.to(device)
        confidence = confidence.to(device)
        viewmat = viewmat.to(device)
        K = K.to(device)
        if gt_depth is not None:
            gt_depth = gt_depth.to(device)

        H, W = image.shape[:2]

        # ====================================================================
        # Forward Pass: Dual-Pass Rendering
        # ====================================================================
        # Zero gradients for all optimizers
        for optimizer in optimizers.values():
            optimizer.zero_grad()

        # Render RGB, Semantic, and Depth (if available)
        renders = render_dual_pass(
            model=model,
            viewmat=viewmat,
            K=K,
            width=W,
            height=H,
            sh_degree=sh_degree,
            render_depth=(gt_depth is not None)  # Enable depth rendering if GT depth available
        )

        pred_rgb = renders['rgb']  # [H, W, 3]
        pred_sem = renders['semantic']  # [H, W, K]
        alpha = renders['alpha']  # [H, W]
        meta = renders['meta']  # Dict with intermediate results
        pred_depth = renders.get('depth', None)  # [H, W] if depth was rendered

        # ====================================================================
        # Compute Losses
        # ====================================================================
        # RGB loss (L1)
        loss_rgb = l1_loss(pred_rgb, image)

        # Depth loss (scale-invariant, if available)
        loss_depth = torch.tensor(0.0, device=device)
        if pred_depth is not None and gt_depth is not None:
            # Ensure GT depth matches rendered resolution
            if gt_depth.shape != pred_depth.shape:
                gt_depth_resized = torch.nn.functional.interpolate(
                    gt_depth.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                    size=pred_depth.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze()  # [H, W]
            else:
                gt_depth_resized = gt_depth

            loss_depth = scale_invariant_depth_loss(pred_depth, gt_depth_resized)

        # Semantic loss (if past warm-up)
        if iteration < warmup_iterations:
            loss_sem = torch.tensor(0.0, device=device)
            current_lambda_sem = 0.0
        else:
            loss_sem = semantic_loss_with_gating(
                pred_sem=pred_sem,
                gt_heatmap=heatmap,
                confidence=confidence,
                fg_weight=fg_weight
            )
            current_lambda_sem = lambda_sem

        # Total loss: RGB + Depth + Semantic
        total_loss = loss_rgb + lambda_depth * loss_depth + current_lambda_sem * loss_sem

        train_losses.append(total_loss.item())

        # ====================================================================
        # Pre-Backward Step (Densification)
        # ====================================================================
        strategy.step_pre_backward(
            params=model.params,
            optimizers=optimizers,
            state=strategy_state,
            step=iteration,
            info={**meta, 'width': W, 'height': H, 'n_cameras': 1}
        )

        # ====================================================================
        # Backward Pass
        # ====================================================================
        total_loss.backward()

        # ====================================================================
        # Gradient Check (Only when logging to avoid CPU-GPU sync every iteration)
        # ====================================================================
        # CRITICAL: Check gradients AFTER backward() but BEFORE step_post_backward()
        # This ensures we capture real gradient values before densification clears them
        # The .item() call forces CPU-GPU sync, so we ONLY do this when logging
        if iteration % log_every == 0:
            grad_3d_norm = 0.0
            if model.params['means'].grad is not None:
                grad_3d_norm = model.params['means'].grad.norm().item()

            # Print MAX gradient to see the strongest signal (per-point peak)
            grad_2d_max = 0.0
            if meta['means2d'].grad is not None:
                grad_2d_max = meta['means2d'].grad.abs().max().item()

            print(f"[DEBUG] Iter {iteration}: Grad3D_Norm={grad_3d_norm:.6f}, Grad2D_MAX={grad_2d_max:.8f}")

            # Max Grad Norm (with safety check)
            max_grad_norm = 0.0
            if model.params['means'].grad is not None:
                max_grad_norm = model.params['means'].grad.norm(dim=-1).max().item()

        # ====================================================================
        # Post-Backward Step (Densification)
        # ====================================================================
        strategy.step_post_backward(
            params=model.params,
            optimizers=optimizers,
            state=strategy_state,
            step=iteration,
            info={**meta, 'width': W, 'height': H, 'n_cameras': 1},
            packed=True  # Restore packed=True for performance
        )

        # IMPORTANT: After densification (split/clone/prune), ensure requires_grad=True
        # PERFORMANCE: Only check this when densification actually happens (every 100 iters)
        # DefaultStrategy may create new parameters which don't have requires_grad set
        # Checking requires_grad every iteration is expensive (causes CUDA sync), so we only do it
        # when densification operations actually occur
        if iteration % 100 == 0 and iteration >= 500:
            for param_name, param in model.params.items():
                if not param.requires_grad:
                    param.requires_grad_(True)

        # ====================================================================
        # Optimizer Step
        # ====================================================================
        for optimizer in optimizers.values():
            optimizer.step()

        # ====================================================================
        # Logging
        # ====================================================================
        if iteration % log_every == 0:
            elapsed = time.time() - start_time
            remaining = elapsed / (iteration + 1) * (num_iterations - iteration - 1)

            # Max Grad Norm (with safety check)
            # Note: Gradients were already computed and printed earlier in the loop
            max_grad_norm = 0.0
            if model.params['means'].grad is not None:
                max_grad_norm = model.params['means'].grad.norm(dim=-1).max().item()

            print(f"Iter {iteration:5d} | "
                  f"Loss: {total_loss.item():.6f} (RGB: {loss_rgb.item():.6f}, "
                  f"Depth: {loss_depth.item():.6f}, "
                  f"Sem: {loss_sem.item():.6f}, lambda_sem: {current_lambda_sem:.3f}) | "
                  f"Gaussians: {model.num_gaussians()} | "
                  f"Max Grad Norm: {max_grad_norm:.6f} | "
                  f"Time: {elapsed:.1f}s / ETA: {remaining:.1f}s")

        # ====================================================================
        # Save Checkpoint
        # ====================================================================
        if iteration % save_every == 0 and iteration > 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{iteration:05d}.pth")
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_states_dict': {k: v.state_dict() for k, v in optimizers.items()},
                'strategy_state': strategy_state,
                'losses': train_losses
            }, checkpoint_path)
            print(f"  -> Saved checkpoint: {checkpoint_path}")

    # ========================================================================
    # Final Save
    # ========================================================================
    final_checkpoint = os.path.join(output_dir, "checkpoint_final.pth")
    torch.save({
        'iteration': num_iterations,
        'model_state_dict': model.state_dict(),
        'optimizer_states_dict': {k: v.state_dict() for k, v in optimizers.items()},
        'strategy_state': strategy_state,
        'losses': train_losses
    }, final_checkpoint)

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Final checkpoint: {final_checkpoint}")
    print("="*60)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Configuration
    train(
        # Data paths (modify these for your setup)
        colmap_dir=r"D:\PythonProject\PythonProject\data\video_data\colmap_data\video2_output_ply",
        images_dir=r"D:\PythonProject\PythonProject\data\video_data\frames\video2_frame",
        heatmap_dir=r"D:\PythonProject\PythonProject\data\heatmaps\heatmap_video2_stem",
        confidence_path=r"D:\PythonProject\PythonProject\data\heatmaps\confidence_video2_stem.json",
        pcd_path=r"D:\PythonProject\PythonProject\data\video_data\colmap_data\video2_output_ply\sparse\0\points3D.ply",
        depth_dir=r"D:\PythonProject\PythonProject\data\depths\video2_depths",  # New: depth maps directory

        # Training settings
        num_iterations=20000,
        warmup_iterations=4000,    # "Geometry First": RGB builds cylindrical volume before semantic classification
        lambda_sem=0.2,            # Standard weight (skeleton supervision is already strong)
        lambda_depth=0.1,          # New: depth loss weight (scale-invariant)
        lr_scale=1.0,
        spatial_lr_scale=1.0,

        # Model settings
        sh_degree=3,
        num_classes=4,

        # Loss weights
        fg_weight=20.0,

        # Logging
        log_every=100,
        save_every=5000,
        output_dir=r"D:\PythonProject\PythonProject\output\video2_depth_supervision",  # Updated output directory

        # Device
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
