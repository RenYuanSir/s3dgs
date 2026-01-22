#!/usr/bin/env python3
"""
Semantic 3D Gaussian Splatting - Training Entry Point
"""

import argparse
import sys
import os
from pathlib import Path
import traceback

# Add project root to path
_project_root = Path(__file__).parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from s3dgs.train import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic 3D Gaussian Splatting Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required data paths
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--heatmap_dir", type=str, required=True)
    parser.add_argument("--confidence_path", type=str, required=True)
    parser.add_argument("--pcd_path", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, default=None)

    # Training settings
    parser.add_argument("--num_iterations", type=int, default=20000)
    parser.add_argument("--warmup_iterations", type=int, default=4000)
    parser.add_argument("--lambda_sem", type=float, default=0.2)
    parser.add_argument("--lambda_depth", type=float, default=0.1)
    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--spatial_lr_scale", type=float, default=1.0)

    # Model settings
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=4)

    # Loss settings
    parser.add_argument("--fg_weight", type=float, default=20.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)

    # Performance
    parser.add_argument("--resolution_scale", type=float, default=0.75)

    # Logging
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="./output")

    # Device
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("Semantic 3D Gaussian Splatting Training")
    print("="*60)
    print()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        args.device = "cpu"

    # Validate paths
    print("Validating data paths...")
    required = {
        "COLMAP": args.colmap_dir,
        "Images": args.images_dir,
        "Heatmaps": args.heatmap_dir,
        "Confidence": args.confidence_path,
        "Point Cloud": args.pcd_path
    }
    if args.depth_dir:
        required["Depth"] = args.depth_dir

    for name, path in required.items():
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found: {path}")
            sys.exit(1)
        print(f"  ✓ {name}: {path}")

    print("\nStarting training...\n")

    try:
        train(
            colmap_dir=args.colmap_dir,
            images_dir=args.images_dir,
            heatmap_dir=args.heatmap_dir,
            confidence_path=args.confidence_path,
            pcd_path=args.pcd_path,
            depth_dir=args.depth_dir,
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations,
            lambda_sem=args.lambda_sem,
            lambda_depth=args.lambda_depth,
            lr_scale=args.lr_scale,
            spatial_lr_scale=args.spatial_lr_scale,
            sh_degree=args.sh_degree,
            num_classes=args.num_classes,
            fg_weight=args.fg_weight,
            confidence_threshold=args.confidence_threshold,
            resolution_scale=args.resolution_scale,
            log_every=args.log_every,
            save_every=args.save_every,
            output_dir=args.output_dir,
            device=args.device
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
