import requests
import base64
import os
import json
import time

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Replace with your actual RunPod Endpoint ID (e.g. from https://www.runpod.io/console/serverless)
ENDPOINT_ID = ""

# 2. Replace with your RunPod API Key (or set it in your environment variables)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")

# 3. Path to the image you want to edit
IMAGE_PATH = "test_image.jpg" # You can change this or place a test_image.jpg in this folder

# 4. Your prompt for the model
PROMPT = "Change the shirt color to black"

def image_to_base64(filepath):
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def save_base64_image(base64_string, output_filepath):
    with open(output_filepath, "wb") as output_file:
        output_file.write(base64.b64decode(base64_string))

def main():
    if RUNPOD_API_KEY == "YOUR_RUNPOD_API_KEY":
        print("❌ Error: Please set your RUNPOD_API_KEY in the script or environment.")
        return

    if ENDPOINT_ID == "YOUR_ENDPOINT_ID":
        print("❌ Error: Please set your ENDPOINT_ID in the script.")
        return

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Could not find image at {IMAGE_PATH}. Please provide an image to test.")
        # Create a dummy image just so the script doesn't completely fail for the example
        print("Creating a tiny dummy image for testing...")
        from PIL import Image
        img = Image.new('RGB', (512, 512), color = 'red')
        img.save(IMAGE_PATH)

    print(f"Reading image from {IMAGE_PATH}...")
    base64_image = image_to_base64(IMAGE_PATH)

    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    payload = {
        "input": {
            "prompt": PROMPT,
            "image": base64_image,
            "steps": 8,
            "true_cfg_scale": 1.0
        }
    }

    print(f"Sending request to RunPod Endpoint {ENDPOINT_ID}...")
    start_time = time.time()
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"❌ HTTP Error {response.status_code}: {response.text}")
        return

    data = response.json()
    
    if data.get("status") == "COMPLETED" or data.get("status") == "IN_PROGRESS":
        # Note: runsync might return IN_PROGRESS if it takes too long, 
        # for a truly synchronous call that waits, sometimes you need to poll the /status endpoint.
        # But let's assume it returns COMPLETED for our 8 step model.
        output = data.get("output", {})
        if output.get("status") == "success":
            print(f"✅ Success! Generation took {output['meta']['generate_seconds']}s")
            
            output_file = "output_image.jpg"
            save_base64_image(output['image'], output_file)
            print(f"🎉 Saved edited image to {output_file}")
        else:
            print(f"❌ Inference Error: {output.get('message', 'Unknown Error')}")
    else:
        print(f"❌ Job Failed or Not Completed: {json.dumps(data, indent=2)}")

if __name__ == "__main__":
    main()
