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
from gsplat.strategy import DefaultStrategy

# Import our modules (now works from any directory)
from s3dgs.model import SemanticGaussianModel
from s3dgs.dataset import TomatoDataset, create_dataloader
from s3dgs.loss import l1_loss, scale_invariant_depth_loss, semantic_loss_with_gating
from s3dgs.render import render_dual_pass


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
    depth_dir: str = None,  # Path to individual depth maps
    depth_npz_path: str = None,  # New: path to unified DA3 NPZ file

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

    # Sparse Semantic Supervision
    confidence_threshold: float = 0.5,  # YOLO confidence threshold for semantic validity

    # Performance optimization
    resolution_scale: float = 1.0,  # Full resolution (1080P) - optimized rendering pipeline now handles this efficiently

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
        depth_dir: Path to individual depth map .npz/.png files (optional)
        depth_npz_path: Path to unified DA3 results NPZ file (optional, takes precedence)
        num_iterations: Total training iterations
        warmup_iterations: Iterations for geometric warm-up (lambda_sem=0)
        lambda_sem: Semantic loss weight after warm-up
        lambda_depth: Depth loss weight (scale-invariant)
        lr_scale: Global learning rate scale
        spatial_lr_scale: Spatial learning rate scale
        sh_degree: Spherical harmonics degree
        num_classes: Number of semantic classes
        fg_weight: Foreground pixel weight for semantic loss
        confidence_threshold: YOLO confidence threshold for semantic validity
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
        depth_dir=depth_dir,  # Pass depth directory
        depth_npz_path=depth_npz_path,  # Pass unified NPZ path
        confidence_threshold=confidence_threshold  # Pass confidence threshold
    )

    dataloader = create_dataloader(
        colmap_dir=colmap_dir,
        images_dir=images_dir,
        heatmap_dir=heatmap_dir,
        confidence_path=confidence_path,
        depth_dir=depth_dir,  # Pass depth directory
        depth_npz_path=depth_npz_path,  # Pass unified NPZ path
        confidence_threshold=confidence_threshold,  # Pass confidence threshold
        batch_size=1,
        num_workers=0,  # Windows compatibility: must be 0 to avoid multiprocessing issues
        shuffle=True
    )

    # Check if depth data is available
    has_depth = (depth_dir is not None or depth_npz_path is not None) and \
                any(data.get('has_depth', False) for data in dataset.cached_data)

    # Compute sparse semantic supervision statistics
    valid_semantic_frames = sum(
        1 for data in dataset.cached_data
        if max(dataset.confidence_dict[data['name_without_ext']]) >= confidence_threshold
    )
    print(f"Dataset: {len(dataset)} frames")
    print(f"Depth supervision: {'Enabled' if has_depth else 'Disabled'}")
    print(f"Sparse semantic supervision: {valid_semantic_frames}/{len(dataset)} frames ({100*valid_semantic_frames/len(dataset):.1f}%)")
    print(f"Confidence threshold: {confidence_threshold}")

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
    print("Strategy: Pruning-focused (Dense Initialization from DA3)")
    print("  - Starting with ~500k dense points from DA3")
    print("  - Aggressive pruning to remove depth-error floaters")
    print("  - Conservative splitting (only on strong error signals)")
    print("  - Delayed refinement to let geometry align first")

    strategy = DefaultStrategy(
        refine_every=100,
        grow_grad2d=0.0002,        # 2e-4: Official default (conservative splitting)
                                   # Rationale: Starting dense (~500k points), only split if
                                   # there is a VERY strong 2D gradient error signal.
                                   # Previous aggressive growth (5e-5) caused OOM and noise.
        grow_scale3d=0.005,        # Split features > 0.025 units (5.0 * 0.005 = 0.025)
                                   # Keep conservative to avoid over-splitting dense regions
        prune_opa=0.005,           # Aggressively prune invisible/transparent Gaussians
                                   # Rationale: DA3 depth errors generate "floaters" in empty space.
                                   # Strong opacity pruning removes these artifacts early.
        prune_scale3d=3.0,         # Tolerance radius ~15.0 units (5.0 * 3.0 = 15.0)
        refine_start_iter=1000,    # Delay refinement to iteration 1000
                                   # Rationale: Start dense, let geometry align through
                                   # optimization before splitting. Prevents premature explosion.
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
    print(f"Geometry Warm-up Stage 1 (iter < 1000): RGB-only rendering")
    print(f"Geometry Warm-up Stage 2 (1000 <= iter < {warmup_iterations}): RGB + Depth, semantic loss disabled")
    print(f"Full Training (iter >= {warmup_iterations}): RGB + Semantic + Depth")
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
        sem_mask = data['sem_mask'].squeeze(0)  # [1], CPU - semantic validity mask

        # Extract depth if available
        gt_depth = None
        if 'depth' in data:
            gt_depth = data['depth'].squeeze(0)  # [H, W], CPU

        # ====================================================================
        # Apply Resolution Scaling (Performance Optimization)
        # ====================================================================
        # resolution_scale=0.5 provides safer training for dense clouds
        # Reduces pixels by ~75% (0.5^2 = 0.25), preventing OOM in isect_tiles
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
        sem_mask = sem_mask.to(device)  # Move sem_mask to device
        if gt_depth is not None:
            gt_depth = gt_depth.to(device)

        H, W = image.shape[:2]

        # ====================================================================
        # Geometry Warm-up: Staged Rendering Strategy
        # ====================================================================
        # Stage 1 (iter < 1000): RGB-only to minimize memory spike
        # Stage 2 (1000 <= iter < warmup_iterations): RGB + Depth, semantic disabled
        # Stage 3 (iter >= warmup_iterations): Full RGB + Semantic + Depth

        # Determine rendering flags based on iteration
        geometry_warmup_iter = 1000  # Stage 1: RGB-only for first 1000 iters
        render_semantic = (iteration >= geometry_warmup_iter)
        render_depth = (gt_depth is not None) and (iteration >= geometry_warmup_iter)

        # Zero gradients for all optimizers
        for optimizer in optimizers.values():
            optimizer.zero_grad()

        # ====================================================================
        # Memory Monitoring (Before Render)
        # ====================================================================
        if device == "cuda" and iteration == 0:
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n[MEM] Iter {iteration}: Before render - Allocated={mem_before:.2f}GiB, Reserved={mem_reserved:.2f}GiB")

        # Render with controlled auxiliary passes based on warm-up stage
        renders = render_dual_pass(
            model=model,
            viewmat=viewmat,
            K=K,
            width=W,
            height=H,
            sh_degree=sh_degree,
            render_depth=render_depth  # Controlled by warm-up stage
        )

        # ====================================================================
        # Memory Monitoring (After Render)
        # ====================================================================
        if device == "cuda" and iteration == 0:
            mem_after_render = torch.cuda.memory_allocated() / 1e9
            mem_reserved_after = torch.cuda.memory_reserved() / 1e9
            print(f"[MEM] Iter {iteration}: After render - Allocated={mem_after_render:.2f}GiB, Reserved={mem_reserved_after:.2f}GiB")
            print(f"[MEM] Render memory delta: {mem_after_render - mem_before:.2f}GiB")

        pred_rgb = renders['rgb']  # [H, W, 3]
        pred_sem = renders['semantic']  # [H, W, K]
        alpha = renders['alpha']  # [H, W]
        meta = renders['meta']  # Dict with intermediate results
        pred_depth = renders.get('depth', None)  # [H, W] if depth was rendered

        # ====================================================================
        # Compute Losses
        # ====================================================================
        # RGB loss (L1) - ALWAYS active
        loss_rgb = l1_loss(pred_rgb, image)

        # Depth loss (scale-invariant, controlled by warm-up stage)
        loss_depth = torch.tensor(0.0, device=device)
        if iteration >= geometry_warmup_iter and pred_depth is not None and gt_depth is not None:
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

        # Semantic loss (controlled by warm-up stage)
        # Stage 1 (iter < 1000): Disabled (no semantic rendering)
        # Stage 2 (1000 <= iter < warmup_iterations): Disabled (semantic rendered but loss=0)
        # Stage 3 (iter >= warmup_iterations): Enabled with lambda_sem
        if iteration < warmup_iterations:
            loss_sem = torch.tensor(0.0, device=device)
            current_lambda_sem = 0.0
        else:
            # Compute base semantic loss
            loss_sem_base = semantic_loss_with_gating(
                pred_sem=pred_sem,
                gt_heatmap=heatmap,
                confidence=confidence,
                fg_weight=fg_weight
            )

            # Apply sparse semantic supervision: Multiply by sem_mask
            # If frame is low-confidence (sem_mask=0), semantic loss is disabled
            # If frame is high-confidence (sem_mask=1), semantic loss is enabled
            # This prevents "Negative Supervision" from occluded/blurry frames
            loss_sem = loss_sem_base * sem_mask

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
            # Memory Monitoring (Periodic)
            # ====================================================================
            if device == "cuda":
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                num_gaussians = model.params['means'].shape[0]
                print(f"[MEM] Iter {iteration}: Gaussians={num_gaussians:,}, GPU={mem_allocated:.2f}GiB alloc / {mem_reserved:.2f}GiB reserv")

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

            # Determine training stage for logging
            if iteration < geometry_warmup_iter:
                stage = "STAGE1:RGB"
            elif iteration < warmup_iterations:
                stage = "STAGE2:RGB+D"
            else:
                stage = "STAGE3:FULL"

            # Log semantic mask status for debugging
            sem_mask_val = sem_mask.item()
            sem_status = "ENABLED" if sem_mask_val > 0.5 else "DISABLED"

            print(f"Iter {iteration:5d} [{stage}] | "
                  f"Loss: {total_loss.item():.6f} (RGB: {loss_rgb.item():.6f}, "
                  f"Depth: {loss_depth.item():.6f}, "
                  f"Sem: {loss_sem.item():.6f} [{sem_status}], lambda_sem: {current_lambda_sem:.3f}) | "
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
# Module Export
# ============================================================================
# To train from command line, use: python start.py
# This module can also be imported and used programmatically:
#
# from s3dgs.train import train
# train(
#     colmap_dir="./data/colmap",
#     images_dir="./data/images",
#     ...
# )
