# S-3DGS: Semantic 3D Gaussian Splatting for Tomato Plant Phenotype Analysis

**Semantic-Guided 3D Scene Reconstruction with Skeleton-Based Supervision**

## Overview

This project implements **Semantic 3D Gaussian Splatting (S-3DGS)** for high-fidelity tomato plant 3D reconstruction and semantic segmentation. It augments standard 3DGS with dual-pass rendering and skeleton-based heatmap supervision to achieve fine-grained stem reconstruction (3-5mm diameter).

## Key Features

- **Dual-Pass Rendering**: RGB + Semantic channels
- **Skeleton Supervision**: Continuous line-based supervision (not isolated points)
- **Adaptive Heatmaps**: Topology-aware U-N-D-P connections
- **Official gsplat Integration**: Compatible with gsplat v1.5.3
- **High-Quality Reconstruction**: Preserves fine stem structures

## Quick Start

### Installation

```bash
# Create conda environment
conda create -n mygsplat python=3.10
conda activate mygsplat

# Install dependencies
pip install torch torchvision
pip install gsplat==1.5.3
pip install numpy==1.26.4  # Must be 1.x for pycolmap compatibility
pip install opencv-python pillow scipy plyfile tqdm
pip install fused_ssim
```

### Training

```bash
# Preprocess: Generate skeleton heatmaps
python preprocess/generate_heatmaps_stem.py

# Train semantic 3DGS model
python s3dgs/train.py
```

### Visualization

```bash
# Visualize heatmaps
python vis_heatmap_overlay.py

# View trained model (requires official gsplat examples)
cd official_gsplat/gsplat/examples
python simple_viewer.py --ckpt "path/to/ckpt.pt" --port 8080
```

## Project Structure

```
├── s3dgs/                    # Core implementation
│   ├── model.py             # SemanticGaussianModel
│   ├── dataset.py           # TomatoDataset + COLMAP parsers
│   └── train.py             # Training loop
├── preprocess/               # Data preprocessing
│   ├── extract_frames.py    # Video → frames
│   ├── yolo_inference.py    # Keypoint detection
│   └── generate_heatmaps_stem.py  # Skeleton heatmaps
├── docs/                     # Technical documentation
│   ├── README.md             # Documentation index
│   ├── PROJECT_TECHNICAL_SPECIFICATION.md
│   ├── DATA_FLOW_SPECIFICATION.md
│   └── PROJECT_STATUS_REPORT.md
└── vis_heatmap_overlay.py   # Visualization tool
```

## Technical Highlights

- **Skeleton Heatmaps**: Continuous lines connecting U-N-D-P keypoints with adaptive thickness
- **Dual-Pass Rendering**: SH-based RGB + softmax semantic rendering
- **Geometry-First Strategy**: 4000 iteration warm-up before semantic supervision
- **AbsGS Densification**: Absolute gradient accumulation for sparse regions

## Citation

If you use this code, please cite:

```bibtex
@software{s3dgs_tomato,
  title = {Semantic 3D Gaussian Splatting for Tomato Plant Phenotype Analysis},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/s3dgs}
}
```

## License

MIT License

## Acknowledgments

- **gsplat**: Official 3D Gaussian Splatting library (v1.5.3)
- **COLMAP**: Structure-from-Motion pipeline
- **Roboflow**: Keypoint detection inference
