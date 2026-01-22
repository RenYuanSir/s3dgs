"""
Example usage script for semantic injection

This script shows how to integrate the semantic injection tool
into your 3DGS training pipeline.
"""

from tools.inject_semantics import Config, inject_semantics


def main():
    """
    Example: Inject semantics into a DA3 reconstructed point cloud
    """

    # ===== Configuration =====
    # Modify these paths to match your data structure
    cfg = Config()

    # Input: Dense point cloud from DA3 or COLMAP
    cfg.dense_pcd_path = r"D:\data\video_data\colmap_data\video3_output\sparse\0\points3D.ply"

    # Input: Directory containing semantic heatmaps (.npy files)
    cfg.heatmap_dir = r"D:\data\video_data\heatmaps\video3"

    # Input: COLMAP sparse directory (contains cameras.bin and images.bin)
    cfg.poses_path = r"D:\data\video_data\colmap_data\video3_output\sparse\0"

    # Output: Directory for semantic initialization file
    cfg.output_dir = r"D:\data\video_data\output"

    # ===== Semantic Parameters =====
    cfg.num_classes = 4  # U, N, D, P (tomato keypoints)
    cfg.spatial_kernel_size = 5  # 5x5 max-pooling for robustness
    cfg.confidence_threshold = 0.3  # Gating threshold

    # ===== Advanced Parameters =====
    cfg.depth_tolerance = 0.1  # 10% tolerance for depth checking
    cfg.batch_size_points = 100000  # Process 100k points at a time
    cfg.enable_depth_filtering = True  # Enable depth-based visibility

    # ===== Run Injection =====
    print("Starting semantic injection...")
    inject_semantics(cfg)

    print("\nSemantic injection complete!")
    print("You can now use the output .pt file to initialize your 3DGS model.")


if __name__ == "__main__":
    main()
