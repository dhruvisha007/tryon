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
from diffusers import QwenImageEditPlusPipeline
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODEL_ID  = os.getenv("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
LORA_REPO = os.getenv("LORA_REPO", "lightx2v/Qwen-Image-Edit-2511-Lightning")
LORA_FILE = os.getenv("LORA_FILE", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")

USE_LORA = os.getenv("USE_LORA", "1") == "1"

# Step count must match the LoRA. Running the 4-step LoRA at 8 steps, or the
# 8-step one at 4, degrades output badly and silently - so derive the default
# from the filename unless DEFAULT_STEPS is set explicitly.
#
# Halving steps roughly halves generation time AND cost, since serverless bills
# per second. On a slow card that is the difference between ~30s and ~15s.
def _steps_from_lora(name: str, fallback: int = 8) -> int:
    for n in (4, 8):
        if f"{n}steps" in name.lower():
            return n
    return fallback


DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS") or _steps_from_lora(LORA_FILE))
# Lightning LoRAs are DISTILLED FOR CFG 1.0. Above that they degrade visibly -
# worth A/B-ing 1.0 against the current 1.5 on a fixed seed.
DEFAULT_CFG   = float(os.getenv("DEFAULT_CFG", "1.5"))

# Denoising cost scales with pixel count, so 768 instead of 1024 is roughly
# 1.8x less work - and on serverless, which bills per second, 1.8x less money.
# Upscaling the result afterwards is far cheaper than generating big.
MAX_SIDE      = int(os.getenv("MAX_SIDE", "1024"))
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "30"))

MODEL_DIR = os.getenv("MODEL_DIR", None)

# Shrink the transformer so the pipeline fits a smaller card with BOTH the
# transformer and the text encoder resident - no CPU offload (measured: offload
# on 48GB times out, staging ~60GB through system RAM thrashes).
#
#   none   transformer 40GB + encoder 16GB + ~5GB = ~61GB  -> needs 80GB+
#   fp8    transformer 20GB + encoder 16GB + ~5GB = ~41GB  -> fits 48GB
#   int8   similar to fp8, and runs on Ampere too
#   nf4    transformer ~11GB                      = ~32GB  -> fits 40GB
#
# The text encoder CANNOT be dropped or precomputed: Qwen-Image-EDIT encodes the
# prompt together with the input images (encode_prompt takes image=), so the
# embeddings differ for every customer's photos.
#
# fp8 (torchao) needs Ada/Blackwell, sm_89+. On Ampere (A40/A6000) use int8/nf4.
QUANT         = os.getenv("QUANT", "none").lower()
QUANT_ENCODER = os.getenv("QUANT_ENCODER", "0") == "1"

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


def build_quant_config():
    """Quantisation config for from_pretrained, or None when QUANT=none."""
    if QUANT == "none":
        return None

    from diffusers import PipelineQuantizationConfig

    targets = ["transformer"] + (["text_encoder"] if QUANT_ENCODER else [])

    if QUANT == "fp8":
        return PipelineQuantizationConfig(
            quant_backend="torchao",
            quant_kwargs={"quant_type": "float8dq_e4m3_row"},
            components_to_quantize=targets,
        )

    if QUANT in ("int8", "nf4"):
        kw = (
            {"load_in_8bit": True}
            if QUANT == "int8"
            else {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            }
        )
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit" if QUANT == "int8" else "bitsandbytes_4bit",
            quant_kwargs=kw,
            components_to_quantize=targets,
        )

    raise ValueError(f"unknown QUANT={QUANT!r} (expected none|fp8|int8|nf4)")


def load_pipeline():
    """Load once per container, not per request. Cold start pays this; warm calls don't."""
    global PIPE
    if PIPE is not None:
        return PIPE

    logger.info("Worker startup: loading model...")
    t0 = time.time()
    
    use_auth_token = os.getenv("HF_TOKEN") is not None

    quant_cfg = build_quant_config()
    if quant_cfg is not None:
        logger.info(f"Quantising with QUANT={QUANT} (encoder={QUANT_ENCODER})")

    try:
        logger.info("Attempting to load model from local cache...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_DIR,
            use_auth_token=use_auth_token,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="flash_attention_2",
            quantization_config=quant_cfg
        )
    except Exception as e:
        logger.info(f"Local cache miss, downloading model... ({e})")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            cache_dir=MODEL_DIR,
            use_auth_token=use_auth_token,
            local_files_only=False,
            use_safetensors=True,
            attn_implementation="flash_attention_2",
            quantization_config=quant_cfg
        )

    if USE_LORA:
        logger.info(f"Loading LoRA from {LORA_REPO} ({LORA_FILE})")
        try:
            pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE, cache_dir=MODEL_DIR, use_auth_token=use_auth_token, local_files_only=True)
        except Exception:
            pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE, cache_dir=MODEL_DIR, use_auth_token=use_auth_token, local_files_only=False)
        # fuse_lora() writes back into the transformer's weights, which
        # quantised layers reject. Keep the adapter attached instead - a little
        # slower per step, but the only way to combine LoRA with QUANT.
        if QUANT == "none":
            pipe.fuse_lora()
            pipe.unload_lora_weights()
        else:
            logger.info(f"LoRA left unfused (QUANT={QUANT})")

    logger.info("Moving model to CUDA (No CPU Offload)")
    pipe.to("cuda")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info(
            f"VRAM after load: {(total - free) / 2**30:.1f}GB used of "
            f"{total / 2**30:.1f}GB (quant={QUANT})"
        )

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
