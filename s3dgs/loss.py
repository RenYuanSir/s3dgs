"""
Loss Functions for Semantic 3D Gaussian Splatting

This module contains all loss functions used for training:
- L1 loss for RGB reconstruction
- Scale-invariant depth loss for depth supervision
- Semantic loss with confidence gating and foreground weighting
"""

import torch
import torch.nn.functional as F
from typing import Optional


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


def scale_invariant_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None
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
