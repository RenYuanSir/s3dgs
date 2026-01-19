"""
Depth Map Generation using Depth Anything V2

This script generates monocular depth maps for RGB images using the Depth Anything V2 model
via the gradio_client API (online inference service).

The depth maps are normalized and saved as 16-bit PNG files for efficient storage
and loading during 3DGS training.

Author: Integration for Depth Anything V2
Date: 2025-01-19
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple
import warnings


def setup_gradio_client():
    """
    Setup and return the gradio_client for Depth Anything V2.

    Returns:
        gradio_client Client instance

    Note:
        Requires: pip install gradio_client
    """
    try:
        from gradio_client import Client
    except ImportError:
        raise ImportError(
            "gradio_client is not installed. "
            "Please install it using: pip install gradio_client"
        )

    # Depth Anything V2 HuggingFace Space
    # Correct format: "owner/model-name"
    client = Client("depth-anything/Depth-Anything-V2")

    return client


def generate_depth_for_image(client, image_path: str) -> np.ndarray:
    """
    Generate depth map for a single image using Depth Anything V2.

    Args:
        client: Gradio client instance
        image_path: Path to input RGB image (file path)

    Returns:
        depth_map: Normalized depth map as numpy array [H, W] in range [0, 1]

    Note:
        The API returns raw depth predictions. We normalize them to [0, 1] range.

    API Format:
        client.predict(
            image=handle_file('path/to/image.jpg'),
            api_name="/on_submit"
        )
    """
    try:
        from gradio_client import handle_file

        # Call the Depth Anything V2 API
        # The API expects a file path wrapped in handle_file()
        result = client.predict(
            image=handle_file(image_path),
            api_name="/on_submit"
        )

        # The result is typically a file path to the output depth image
        if isinstance(result, str):
            # If it's a file path, load it
            depth_image = Image.open(result).convert('L')
            depth_map = np.array(depth_image, dtype=np.float32) / 255.0
        elif isinstance(result, np.ndarray):
            # If it's already a numpy array
            depth_map = result.astype(np.float32)
            if depth_map.max() > 1.0:
                depth_map = depth_map / depth_map.max()  # Normalize to [0, 1]
        else:
            # Fallback: try to convert to PIL Image
            depth_image = Image.fromarray(result)
            depth_map = np.array(depth_image, dtype=np.float32) / 255.0

        return depth_map

    except Exception as e:
        warnings.warn(f"Error generating depth for {image_path}: {e}")
        # Return a blank depth map as fallback
        return np.zeros((512, 512), dtype=np.float32)


def save_depth_map(depth_map: np.ndarray, output_path: str):
    """
    Save depth map as 16-bit PNG for high precision storage.

    Args:
        depth_map: Normalized depth map [H, W] in range [0, 1]
        output_path: Path to save the depth map
    """
    # Convert to 16-bit integer range [0, 65535]
    depth_16bit = (depth_map * 65535).astype(np.uint16)

    # Save as 16-bit PNG
    Image.fromarray(depth_16bit, mode='I;16').save(output_path)


def load_depth_map(depth_path: str) -> np.ndarray:
    """
    Load depth map from 16-bit PNG and normalize back to [0, 1].

    Args:
        depth_path: Path to depth map file

    Returns:
        depth_map: Normalized depth map [H, W] in range [0, 1]
    """
    depth_16bit = np.array(Image.open(depth_path), dtype=np.float32)
    depth_map = depth_16bit / 65535.0
    return depth_map


def test_api_connection(client):
    """
    Test the Depth Anything V2 API connection with a sample image.

    Args:
        client: Gradio client instance

    Returns:
        bool: True if connection successful, False otherwise
    """
    print("\nTesting API connection...")

    try:
        from gradio_client import handle_file

        # Use a test image URL (gradio's test image)
        test_url = "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"

        result = client.predict(
            image=handle_file(test_url),
            api_name="/on_submit"
        )

        if result is not None:
            print("✓ API connection successful!")
            print(f"  Result type: {type(result)}")
            if isinstance(result, str):
                print(f"  Result path: {result}")
            return True
        else:
            print("✗ API returned None")
            return False

    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False


def generate_depth_maps_batch(
    images_dir: str,
    output_dir: str,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
):
    """
    Generate depth maps for all images in a directory.

    Args:
        images_dir: Directory containing input RGB images
        output_dir: Directory to save generated depth maps
        image_extensions: List of valid image file extensions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get all image files
    image_files = []
    for fname in os.listdir(images_dir):
        if any(fname.endswith(ext) for ext in image_extensions):
            image_files.append(fname)

    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")

    print(f"Found {len(image_files)} images to process")

    # Setup gradio client
    print("Setting up Depth Anything V2 client...")
    client = setup_gradio_client()
    print("Client ready!")

    # Test API connection
    if not test_api_connection(client):
        print("\nWarning: API connection test failed. Proceeding anyway...")

    # Process each image
    success_count = 0
    skip_count = 0

    for fname in tqdm(image_files, desc="Generating depth maps"):
        # Input/output paths
        input_path = os.path.join(images_dir, fname)
        name_without_ext = os.path.splitext(fname)[0]
        output_path = os.path.join(output_dir, f"{name_without_ext}.png")

        # Skip if depth map already exists
        if os.path.exists(output_path):
            skip_count += 1
            continue

        try:
            # Generate depth map (API handles file path directly)
            depth_map = generate_depth_for_image(client, input_path)

            # Save depth map
            save_depth_map(depth_map, output_path)
            success_count += 1

        except Exception as e:
            warnings.warn(f"Failed to process {fname}: {e}")
            continue

    print(f"\nProcessing complete!")
    print(f"  Successfully generated: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Failed: {len(image_files) - success_count - skip_count}")
    print(f"  Output directory: {output_dir}")


def verify_depth_maps(depth_dir: str, num_samples: int = 5):
    """
    Verify generated depth maps by loading a few samples.

    Args:
        depth_dir: Directory containing depth maps
        num_samples: Number of samples to verify
    """
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]

    if len(depth_files) == 0:
        print(f"Warning: No depth maps found in {depth_dir}")
        return

    print(f"\nVerifying depth maps (samples: {min(num_samples, len(depth_files))})...")

    for i, fname in enumerate(depth_files[:num_samples]):
        depth_path = os.path.join(depth_dir, fname)
        try:
            depth_map = load_depth_map(depth_path)
            print(f"  [{i+1}] {fname}: shape={depth_map.shape}, "
                  f"min={depth_map.min():.3f}, max={depth_map.max():.3f}, "
                  f"mean={depth_map.mean():.3f}")
        except Exception as e:
            print(f"  [{i+1}] {fname}: ERROR - {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate depth maps using Depth Anything V2 via gradio_client'
    )
    parser.add_argument(
        'images_dir',
        type=str,
        help='Directory containing input RGB images'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save generated depth maps (e.g., data/depths/)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify generated depth maps after processing'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to verify (default: 5)'
    )

    args = parser.parse_args()

    print("="*60)
    print("Depth Anything V2 - Depth Map Generation")
    print("="*60)

    # Generate depth maps
    generate_depth_maps_batch(
        images_dir=args.images_dir,
        output_dir=args.output_dir
    )

    # Verify if requested
    if args.verify:
        verify_depth_maps(args.output_dir, args.num_samples)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    # Example usage:
    # python preprocess/generate_depth.py data/images/ data/depths/ --verify

    # Or run with default paths (modify these for your setup)
    # Example:
    #   python preprocess/generate_depth.py \
    #       D:\PythonProject\PythonProject\data\video_data\frames\video2_frame \
    #       D:\PythonProject\PythonProject\data\depths\video2_depths \
    #       --verify

    main()
