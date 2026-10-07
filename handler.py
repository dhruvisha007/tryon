"""
RunPod serverless worker - Qwen-Image-Edit-2511
"""

import base64
import io
import os
import time
import traceback
import logging

import requests
import runpod
import torch
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODEL_ID  = os.getenv("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
LORA_REPO = os.getenv("LORA_REPO", "lightx2v/Qwen-Image-Edit-2511-Lightning")
LORA_FILE = os.getenv("LORA_FILE", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")

USE_LORA = os.getenv("USE_LORA", "1") == "1"

DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "8"))
DEFAULT_CFG   = float(os.getenv("DEFAULT_CFG", "1.5"))
MAX_SIDE      = int(os.getenv("MAX_SIDE", "1024"))
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "30"))

MODEL_DIR = os.getenv("MODEL_DIR", None)

PIPE = None

BG_MODEL     = os.getenv("BG_MODEL", "u2netp")
BG_MASK_SIDE = int(os.getenv("BG_MASK_SIDE", "512"))
BG_SESSION   = None

def strip_background(img: Image.Image) -> Image.Image:
    global BG_SESSION
    from rembg import new_session, remove

    if BG_SESSION is None:
        BG_SESSION = new_session(BG_MODEL)

    small = img.copy()
    small.thumbnail((BG_MASK_SIDE, BG_MASK_SIDE), Image.LANCZOS)
    cut  = remove(small, session=BG_SESSION)
    mask = cut.getchannel("A").resize(img.size, Image.LANCZOS)
    white = Image.new("RGB", img.size, (255, 255, 255))
    white.paste(img, mask=mask)
    return white


def load_pipeline():
    """Load once per container, not per request. Cold start pays this; warm calls don't."""
    global PIPE
    if PIPE is not None:
        return PIPE

    logger.info("Worker startup: loading model...")
    t0 = time.time()
    
    use_auth_token = os.getenv("HF_TOKEN") is not None
    
    try:
        logger.info("Attempting to load model from local cache...")
        torch.set_default_device("cuda")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            torch_dtype=torch.float8_e4m3fn,
            cache_dir=MODEL_DIR,
            use_auth_token=use_auth_token,
            local_files_only=True,
            use_safetensors=True,
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID, 
            transformer=transformer,
            torch_dtype=torch.bfloat16, 
            cache_dir=MODEL_DIR,
            use_auth_token=use_auth_token,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="flash_attention_2"
        )
        torch.set_default_device("cpu")
    except Exception as e:
        logger.info(f"Local cache miss, downloading model... ({e})")
        torch.set_default_device("cuda")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            torch_dtype=torch.float8_e4m3fn,
            cache_dir=MODEL_DIR,
            use_auth_token=use_auth_token,
            local_files_only=False,
            use_safetensors=True,
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID, 
            transformer=transformer,
            torch_dtype=torch.bfloat16, 
            cache_dir=MODEL_DIR,
            use_auth_token=use_auth_token,
            local_files_only=False,
            use_safetensors=True,
            attn_implementation="flash_attention_2"
        )
        torch.set_default_device("cpu")

    if USE_LORA:
        logger.info(f"Loading LoRA from {LORA_REPO} ({LORA_FILE})")
        try:
            pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE, cache_dir=MODEL_DIR, use_auth_token=use_auth_token, local_files_only=True)
        except Exception:
            pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE, cache_dir=MODEL_DIR, use_auth_token=use_auth_token, local_files_only=False)
        pipe.fuse_lora()
        pipe.unload_lora_weights()

    logger.info("Moving model to CUDA (No CPU Offload)")
    pipe.to("cuda")

    pipe.set_progress_bar_config(disable=True)

    try:
        t1 = time.time()
        strip_background(Image.new("RGB", (64, 64), (255, 255, 255)))
        logger.info(f"BG session ready in {time.time() - t1:.1f}s")
    except Exception as e:
        logger.warning(f"BG session warm failed: {e}")

    logger.info(f"Model loading complete in {time.time() - t0:.1f}s")
    PIPE = pipe
    return PIPE


def load_image(src: str) -> Image.Image:
    """Accept an http(s) URL or a base64 blob."""
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        raw = r.content
    else:
        if "," in src and src.strip().startswith("data:"):
            src = src.split(",", 1)[1]
        raw = base64.b64decode(src)

    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img


def fit(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    nw = max(16, int(round(w * scale / 16)) * 16)
    nh = max(16, int(round(h * scale / 16)) * 16)
    return img.resize((nw, nh), Image.LANCZOS)


def to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def handler(job):
    started = time.time()
    logger.info("Received request")
    try:
        inp = job.get("input") or {}

        if inp.get("warmup"):
            load_pipeline()
            return {
                "status": "success",
                "warmed": True,
                "seconds": round(time.time() - started, 2),
            }
            
        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            logger.error("Missing prompt in request")
            return {"status": "error", "message": "prompt is required"}

        # Support image, images, or init_image for backward compatibility
        image_in = inp.get("image")
        if not image_in:
            images_in = inp.get("init_image") or inp.get("images") or []
            if isinstance(images_in, str):
                images_in = [images_in]
            if not images_in:
                logger.error("Missing image in request")
                return {"status": "error", "message": "image is required"}
            images_in = images_in
        else:
            if isinstance(image_in, str):
                images_in = [image_in]
            else:
                images_in = image_in

        negative = inp.get("negative_prompt") or " "
        steps    = int(inp.get("steps") or inp.get("num_inference_steps") or DEFAULT_STEPS)
        cfg      = float(inp.get("true_cfg_scale") or inp.get("guidance_scale") or inp.get("cfg") or DEFAULT_CFG)
        max_side = int(inp.get("max_side") or MAX_SIDE)
        seed     = inp.get("seed")

        pipe = load_pipeline()
        
        try:
            images = [fit(load_image(s), max_side) for s in images_in]
        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            return {"status": "error", "message": f"Image decoding failed: {e}"}

        strip_bg = bool(inp.get("strip_bg", False))
        if strip_bg and len(images) > 1:
            try:
                images[0] = strip_background(images[0])
            except Exception as e:
                logger.warning(f"strip_bg failed, using original: {e}")

        ref = images[1] if len(images) > 1 else images[0]
        width, height = ref.size

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        logger.info("Inference start")
        gen_started = time.time()
        
        torch.cuda.empty_cache()
        
        with torch.inference_mode():
            result = pipe(
                image=images,
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=cfg,
                true_cfg_scale=cfg,
                width=width,
                height=height,
                generator=generator,
            )
            
        gen_secs = time.time() - gen_started
        logger.info(f"Inference completion in {gen_secs:.1f}s")

        out = result.images[0]
        
        return {
            "image": to_base64(out),
            "status": "success",
            "meta": {
                "steps": steps,
                "true_cfg_scale": cfg,
                "size": [width, height],
                "seed": seed,
                "inputs": len(images),
                "strip_bg": strip_bg,
                "generate_seconds": round(gen_secs, 2),
                "total_seconds": round(time.time() - started, 2),
            }
        }

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Inference failed: {type(e).__name__}: {e}",
            "total_seconds": round(time.time() - started, 2),
        }


if os.getenv("PRELOAD", "1") == "1":
    load_pipeline()

runpod.serverless.start({
    "handler": handler
})
