# 1. Import the library
from inference_sdk import InferenceHTTPClient
import os
import json

# 2. Connect to your local server
client = InferenceHTTPClient(
    api_url="http://localhost:9001", # Local server address
    api_key="bfNITT0ossDHQRyk3vLf"
)

# 3. Configure input and output folders
input_folder = r"D:\PythonProject\PythonProject\video_data\frames\video3_frame"
output_folder = r"D:\PythonProject\PythonProject\video_data\detect_results\video3_frame_results"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image formats
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# 4. Process all images in the input folder
image_files = [f for f in os.listdir(input_folder)
               if os.path.splitext(f)[1].lower() in image_extensions]

print(f"Found {len(image_files)} images to process.")

for idx, image_file in enumerate(image_files, 1):
    image_path = os.path.join(input_folder, image_file)

    print(f"[{idx}/{len(image_files)}] Processing: {image_file}")

    try:
        # Run workflow on the image
        result = client.run_workflow(
            workspace_name="smartagrizust",
            workflow_id="custom-workflow-2",
            images={
                "image": image_path
            },
            use_cache=True
        )

        # Save result to output folder with same name as image (but .json extension)
        base_name = os.path.splitext(image_file)[0]
        output_path = os.path.join(output_folder, f"{base_name}.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  -> Result saved to: {output_path}")

    except Exception as e:
        print(f"  -> Error processing {image_file}: {str(e)}")

print("\nAll images processed!")
