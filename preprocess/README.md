# Preprocessing Scripts

This directory contains preprocessing scripts for preparing data for S-3DGS training.

## Depth Map Generation

### Overview

Generate monocular depth maps using **Depth Anything V2** via the HuggingFace Space API.

### Why Depth Maps?

Monocular depth priors provide geometric constraints that improve 3DGS reconstruction:
- **Fixes broken stems**: Depth provides geometric consistency across frames
- **Reduces floaters**: Depth penalizes Gaussians in empty space
- **Improves thin structures**: Better reconstruction of fine details (stems, branches)

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install gradio_client
   ```

2. **Test API connection** (recommended):
   ```bash
   python test_depth_api.py
   ```

3. **Generate depth maps**:
   ```bash
   python generate_depth.py \
       path/to/images/ \
       path/to/depths/ \
       --verify
   ```

### Scripts

#### `test_depth_api.py`

Quick test script to verify the Depth Anything V2 API is working.

**Usage**:
```bash
# Test with sample image
python test_depth_api.py

# Test with your own image
python test_depth_api.py path/to/your/image.jpg
```

**What it does**:
- Tests connection to `depth-anything/Depth-Anything-V2` HuggingFace Space
- Verifies `gradio_client` is installed correctly
- Shows API response format (helps debug issues)

#### `generate_depth.py`

Main script for batch depth map generation.

**Usage**:
```bash
python generate_depth.py \
    <images_dir> \
    <output_dir> \
    [--verify] [--num_samples N]
```

**Arguments**:
- `images_dir`: Directory containing input RGB images
- `output_dir`: Directory to save generated depth maps
- `--verify`: Verify generated depth maps after processing
- `--num_samples`: Number of samples to verify (default: 5)

**Features**:
- Processes all images in a directory
- Saves depth maps as 16-bit PNG files (high precision)
- Skips already-processed images (resumable)
- Shows progress bar with ETA
- Includes API connection test before processing
- Verification mode to check output quality

**Example**:
```bash
python generate_depth.py \
    "D:\data\frames\video2" \
    "D:\data\depths\video2" \
    --verify --num_samples 10
```

**Expected runtime**:
- ~1-2 seconds per image (API rate limit)
- 500 images ≈ 10-15 minutes

### Output Format

**Depth Maps**:
- Format: 16-bit PNG
- Values: [0, 65535] (normalized [0, 1] after loading)
- Resolution: Matches input images
- Naming: Same as input files (e.g., `frame_0000000.png`)

**Directory Structure**:
```
output_dir/
├── frame_0000000.png  # 16-bit depth map
├── frame_0000001.png
├── ...
└── frame_0000499.png
```

### API Details

**Endpoint**: `depth-anything/Depth-Anything-V2` (HuggingFace Space)

**Method**: `gradio_client.Client.predict()`

**API Name**: `/on_submit`

**Example Call**:
```python
from gradio_client import Client, handle_file

client = Client("depth-anything/Depth-Anything-V2")
result = client.predict(
    image=handle_file('path/to/image.jpg'),
    api_name="/on_submit"
)
# result: Path to generated depth image (temp file)
```

**Limitations**:
- Rate-limited (≈1-2 seconds per request)
- Requires internet connection
- Queue times may vary (HuggingFace Space load)

### Troubleshooting

#### Error: `gradio_client not found`
```bash
pip install gradio_client
```

#### Error: `API connection failed`
- Check your internet connection
- Verify HuggingFace Space is online: https://huggingface.co/spaces/depth-anything/Depth-Anything-V2
- Try again later (Space may be overloaded)

#### Error: `No images found in directory`
- Check that `images_dir` path is correct
- Verify images have valid extensions: `.jpg`, `.jpeg`, `.png` (case-insensitive)

#### Depth maps are all black (zeros)
- API likely failed or returned invalid data
- Check the warning messages in console
- Try running `test_depth_api.py` to verify API is working

#### Processing is very slow
- This is normal! The API is rate-limited
- Expected speed: ~1-2 seconds per image
- Consider processing in batches if you have many images

### Integration with Training

Once depth maps are generated, enable depth supervision in training:

```python
# In s3dgs/train.py
train(
    depth_dir="path/to/depths",  # Add this
    lambda_depth=0.1,            # Add this
    # ... other parameters ...
)
```

See `../docs/DEPTH_ANYTHING_V2_INTEGRATION.md` for full integration guide.

### Advanced Usage

#### Custom Image Extensions

Edit `generate_depth.py`:
```python
image_extensions = ['.jpg', '.png', '.bmp']  # Add your extensions
```

#### Resume from Interruption

The script automatically skips existing depth maps. Just run the same command again:

```bash
# First run (interrupted after 100 images)
python generate_depth.py images/ depths/

# Second run (skips first 100, processes remaining)
python generate_depth.py images/ depths/
```

#### Parallel Processing (Not Recommended)

The API is rate-limited, so parallel processing won't help. It may even get your IP throttled. Stick to sequential processing.

### References

- **Depth Anything V2 Paper**: https://arxiv.org/abs/2406.09414
- **HuggingFace Space**: https://huggingface.co/spaces/depth-anything/Depth-Anything-V2
- **gradio_client Docs**: https://gradio.app/docs/

### License

Depth Anything V2 is licensed under Apache 2.0. See the original repository for details.

---

**Last Updated**: 2025-01-19
**Author**: Claude (Anthropic)
**Status**: ✅ Production Ready
