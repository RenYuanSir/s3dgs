# Semantic 3D Gaussian Splatting - 快速开始

## 标准训练方式（推荐）

```bash
python start.py \
    --colmap_dir ./data/colmap_output \
    --images_dir ./data/frames \
    --heatmap_dir ./data/heatmaps \
    --confidence_path ./data/confidence.json \
    --pcd_path ./data/points3D.ply
```

## 带深度监督的训练

```bash
python start.py \
    --colmap_dir ./data/colmap_output \
    --images_dir ./data/frames \
    --heatmap_dir ./data/heatmaps \
    --confidence_path ./data/confidence.json \
    --pcd_path ./data/points3D.ply \
    --depth_dir ./data/depths
```

## 主要变化

### 之前的项目结构
```bash
python s3dgs/train.py  # 直接运行，路径硬编码在文件中
```

### 现在的标准结构
```bash
python start.py --colmap_dir ... --images_dir ...  # 命令行参数配置
```

或者作为模块导入：
```python
from s3dgs.train import train
train(...)
```

## 项目结构

```
s3dgs/
├── start.py          # 训练入口（推荐使用）
├── s3dgs/
│   ├── train.py      # 训练模块（可导入）
│   ├── model.py
│   ├── dataset.py
│   ├── render.py
│   └── loss.py
├── configs/          # 配置文件目录
└── tools/            # 预处理工具
```

## 查看所有参数

```bash
python start.py --help
```
