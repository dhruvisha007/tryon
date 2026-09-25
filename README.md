# tryon-worker

RunPod serverless worker running **Qwen-Image-Edit-2511** for TryTheClothes.
Replaces the ModelsLab `image_editing/qwen_edit` API call with a self-hosted GPU.

## Request

```json
{
  "input": {
    "init_image": ["<clothing url>", "<person url>"],
    "prompt": "the person from image 2 is wearing outfit from image 1",
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "max_side": 1024,
    "seed": 12345
  }
}
```

`init_image` takes two images for a try-on (clothing, person) or one for an
angle regeneration. Every generation parameter is a request field, so quality
tuning never requires a rebuild.

## Response

```json
{ "status": "success", "images": ["data:image/jpeg;base64,..."], "meta": { ... } }
```

Errors return `{"status": "error", "message": "..."}` rather than raising, so
the Laravel caller can fall back to ModelsLab instead of failing the customer.

## Notes

- Lightning LoRA is distilled for **CFG 1.0**. Raising guidance degrades it.
- `OFFLOAD=1` keeps one component on the GPU at a time; peak VRAM is the
  transformer (~40GB bf16) rather than the full ~55GB pipeline. Needs a 48GB
  card. Set `OFFLOAD=0` only on 80GB+.
- Attach a network volume so weights are downloaded once, not per cold start.
