# tryon-worker (Qwen Image Edit 2511 RunPod Worker)

RunPod serverless worker running **Qwen-Image-Edit-2511**.
This repository has been adapted into a production-ready RunPod Serverless worker.

## Local Docker Testing

To build and test the worker locally:

```bash
# Build the Docker image
docker build -t qwen-image-edit-worker .

# Run locally (requires NVIDIA GPU)
docker run --gpus all -p 8000:8000 --env-file .env qwen-image-edit-worker
```

## RunPod Serverless Deployment

This repository is ready to be deployed as a RunPod Serverless worker.

### 1. Building and Deployment Process

RunPod allows pulling from Docker Hub or building directly via GitHub integrations.
- Push the Docker image to a registry (like Docker Hub or GitHub Container Registry) or configure RunPod to build from this GitHub repository.
- Make sure no secrets (`.env`, `HF_TOKEN`, etc.) are committed to the repository.

### 2. Creating the Endpoint

When creating the Serverless endpoint in RunPod, use the following configuration:
- **GPU Type**: Select **NVIDIA RTX 4090** (24 GB)
- **GPU Count**: 1
- **Active Workers**: 0 (for scale-to-zero)
- **Max Workers**: 1 (initially)
- **FlashBoot**: Highly recommended to enable for faster cold starts.

### 3. Network Volume Configuration

Attach a Network Volume to the endpoint so model weights are cached. 
Map the volume path (e.g., `/runpod-volume`) to the worker.

### 4. Required Environment Variables

Add the following environment variables (or RunPod Secrets):
- `MODEL_DIR=/runpod-volume/models`
- `HF_HOME=/runpod-volume/huggingface`
- `HF_TOKEN=your_huggingface_token` (If the model requires authentication)

### 5. API Request Example

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "prompt": "Change the shirt color to black",
      "image": "BASE64_IMAGE"
    }
  }'
```
*Optional parameters include `seed`, `steps`, and `true_cfg_scale`.*

### 6. API Response Example

```json
{
  "image": "<base64 encoded generated image>",
  "status": "success",
  "meta": {
    "steps": 8,
    "true_cfg_scale": 1.0,
    "size": [1024, 1024],
    "seed": null,
    "inputs": 1,
    "strip_bg": false,
    "generate_seconds": 3.45,
    "total_seconds": 4.12
  }
}
```

### 7. Scaling

To handle more concurrent requests, you can easily increase the **Max Workers** from `1` to `2` (or more) in the RunPod Endpoint settings. RunPod will automatically spin up additional RTX 4090 instances when the queue length increases.

## Notes

- Lightning LoRA is distilled for **CFG 1.0**. Raising guidance degrades it.
- `OFFLOAD=1` keeps one component on the GPU at a time; peak VRAM is the transformer (~40GB bf16) rather than the full ~55GB pipeline. For a 24GB RTX 4090, CPU offloading is crucial.
