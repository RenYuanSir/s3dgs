"""
Quick test script for Depth Anything V2 API

This script tests the API connection and generates a sample depth map.
"""

import os
from gradio_client import Client, handle_file

def test_depth_api():
    """Test the Depth Anything V2 API with a sample image."""

    print("="*60)
    print("Testing Depth Anything V2 API")
    print("="*60)

    # Initialize client
    print("\n1. Initializing client...")
    client = Client("depth-anything/Depth-Anything-V2")
    print("   ✓ Client initialized")

    # Test with a sample image URL
    print("\n2. Testing with sample image...")
    test_url = "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"

    try:
        result = client.predict(
            image=handle_file(test_url),
            api_name="/on_submit"
        )

        print(f"   ✓ API call successful!")
        print(f"   Result type: {type(result)}")
        print(f"   Result: {result}")

        return True

    except Exception as e:
        print(f"   ✗ API call failed: {e}")
        return False

def test_local_image(image_path: str):
    """Test with a local image file."""

    if not os.path.exists(image_path):
        print(f"\n✗ Image not found: {image_path}")
        return False

    print("\n3. Testing with local image...")
    print(f"   Image: {image_path}")

    try:
        from gradio_client import Client, handle_file

        client = Client("depth-anything/Depth-Anything-V2")
        result = client.predict(
            image=handle_file(image_path),
            api_name="/on_submit"
        )

        print(f"   ✓ API call successful!")
        print(f"   Result: {result}")

        # Try to load the result if it's a file path
        from PIL import Image
        import numpy as np

        if isinstance(result, str) and os.path.exists(result):
            depth_img = Image.open(result)
            depth_map = np.array(depth_img)
            print(f"   Depth map shape: {depth_map.shape}")
            print(f"   Depth map dtype: {depth_map.dtype}")
            print(f"   Depth range: [{depth_map.min()}, {depth_map.max()}]")

        return True

    except Exception as e:
        print(f"   ✗ API call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test 1: API connection
    success = test_depth_api()

    if success:
        # Test 2: Local image (if provided)
        import sys
        if len(sys.argv) > 1:
            local_image = sys.argv[1]
            test_local_image(local_image)
        else:
            print("\n3. To test with a local image:")
            print("   python preprocess/test_depth_api.py path/to/your/image.jpg")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
