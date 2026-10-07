import base64
import io
import os
import time
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from PIL import Image, ImageOps
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ID  = os.getenv("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
LORA_REPO = os.getenv("LORA_REPO", "lightx2v/Qwen-Image-Edit-2511-Lightning")
LORA_FILE = os.getenv("LORA_FILE", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
USE_LORA = os.getenv("USE_LORA", "1") == "1"
MODEL_DIR = os.getenv("MODEL_DIR", None)
MAX_SIDE = int(os.getenv("MAX_SIDE", "1024"))

PIPE = None

def load_pipeline():
    global PIPE
    if PIPE is not None:
        return PIPE

    logger.info("Loading Qwen model...")
    use_auth_token = os.getenv("HF_TOKEN") is not None
    
    # LOAD IN BFLOAT16 to avoid PEFT/Float8 crash. 
    # This allows LoRA to work correctly and enables model_cpu_offload to fit in 24GB VRAM.
    try:
        transformer = QwenImageTransformer2DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
            cache_dir=MODEL_DIR, use_auth_token=use_auth_token, local_files_only=True
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16, 
            cache_dir=MODEL_DIR, use_auth_token=use_auth_token, local_files_only=True
        )
    except Exception:
        transformer = QwenImageTransformer2DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
            cache_dir=MODEL_DIR, use_auth_token=use_auth_token, local_files_only=False
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16, 
            cache_dir=MODEL_DIR, use_auth_token=use_auth_token, local_files_only=False
        )

    if USE_LORA:
        logger.info(f"Loading LoRA from {LORA_REPO}")
        try:
            pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE, cache_dir=MODEL_DIR, local_files_only=True)
        except Exception:
            pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE, cache_dir=MODEL_DIR, local_files_only=False)
        
        # We CAN fuse here because we are using bfloat16!
        pipe.fuse_lora()
        pipe.unload_lora_weights()

    # Use model_cpu_offload which is faster than sequential_cpu_offload, 
    # but still saves enough memory to fit the inference on a 24GB RTX 4090.
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    PIPE = pipe
    return pipe

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_pipeline()
    yield
    
app = FastAPI(lifespan=lifespan)

class EditRequest(BaseModel):
    image_1: str # Garment
    image_2: str # Person
    prompt: Optional[str] = "the person from image 2 is wearing outfit from image 1 Keep the entire original background from image 2 exactly as it is — the same scene, colours, objects, depth and lighting behind the person. Do not blur, replace, remove, whiten or regenerate the background; change only the clothing. Keep the whole body and both feet fully visible"
    steps: int = 8
    cfg_scale: float = 1.5

def decode_image(b64: str) -> Image.Image:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    return ImageOps.exif_transpose(img).convert("RGB")

def to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()

def fit(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    nw = max(16, int(round(w * scale / 16)) * 16)
    nh = max(16, int(round(h * scale / 16)) * 16)
    return img.resize((nw, nh), Image.LANCZOS)

@app.post("/edit")
async def edit_image(req: EditRequest):
    pipe = PIPE
    if pipe is None:
        raise HTTPException(500, "Pipeline not loaded")

    try:
        img1 = fit(decode_image(req.image_1), MAX_SIDE)
        img2 = fit(decode_image(req.image_2), MAX_SIDE)
    except Exception as e:
        raise HTTPException(400, f"Invalid images: {e}")

    images = [img1, img2]
    width, height = img2.size

    torch.cuda.empty_cache()
    
    with torch.inference_mode():
        result = pipe(
            image=images,
            prompt=req.prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            true_cfg_scale=req.cfg_scale,
            width=width,
            height=height,
        )

    out_b64 = to_base64(result.images[0])
    return {"status": "success", "image": out_b64}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
