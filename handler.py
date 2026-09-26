"""
RunPod serverless worker - Qwen-Image-Edit-2511 virtual try-on for TryTheClothes.

Deliberately mirrors the ModelsLab qwen_edit contract the Laravel app already
speaks, so TryOnService can switch engines without changing how it submits a
job or polls for the result.

Everything worth tuning (steps, guidance, size, prompt) is a REQUEST parameter,
not a build-time constant - so quality tuning never needs a rebuild.
"""

import base64
import io
import os
import time
import traceback

import requests
import runpod
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image, ImageOps

MODEL_ID  = os.getenv("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
LORA_REPO = os.getenv("LORA_REPO", "lightx2v/Qwen-Image-Edit-2511-Lightning")
LORA_FILE = os.getenv("LORA_FILE", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")

USE_LORA = os.getenv("USE_LORA", "1") == "1"
# Keep one component on the GPU at a time. Peak VRAM is then the transformer
# (~40GB in bf16) instead of the whole ~55GB pipeline, which is what lets this
# run on a 48GB card. Turn off only on 80GB+.
OFFLOAD = os.getenv("OFFLOAD", "1") == "1"

DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "8"))
# Lightning LoRAs are distilled for CFG 1.0 - raising it degrades them badly.
DEFAULT_CFG   = float(os.getenv("DEFAULT_CFG", "1.0"))
MAX_SIDE      = int(os.getenv("MAX_SIDE", "1024"))
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "30"))

PIPE = None


def load_pipeline():
    """Load once per container, not per request. Cold start pays this; warm calls don't."""
    global PIPE
    if PIPE is not None:
        return PIPE

    t0 = time.time()
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

    if USE_LORA:
        pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE)
        pipe.fuse_lora()
        pipe.unload_lora_weights()

    if OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    pipe.set_progress_bar_config(disable=True)
    print(f"[worker] pipeline ready in {time.time() - t0:.1f}s "
          f"(lora={USE_LORA}, offload={OFFLOAD})", flush=True)

    PIPE = pipe
    return PIPE


def load_image(src: str) -> Image.Image:
    """Accept an http(s) URL or a (possibly data:-prefixed) base64 blob."""
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        raw = r.content
    else:
        if "," in src and src.strip().startswith("data:"):
            src = src.split(",", 1)[1]
        raw = base64.b64decode(src)

    img = Image.open(io.BytesIO(raw))
    # Phone photos carry EXIF rotation; without this people come out sideways.
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img


def fit(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    # Qwen's VAE wants dimensions on a multiple of 16.
    nw = max(16, int(round(w * scale / 16)) * 16)
    nh = max(16, int(round(h * scale / 16)) * 16)
    return img.resize((nw, nh), Image.LANCZOS)


# u2netp is the lightweight u2net - far quicker on CPU and plenty accurate for
# a garment on a product-shot background. BG_MASK_SIDE caps the resolution the
# cutout is computed at.
BG_MODEL     = os.getenv("BG_MODEL", "u2netp")
BG_MASK_SIDE = int(os.getenv("BG_MASK_SIDE", "512"))

BG_SESSION = None


def strip_background(img: Image.Image) -> Image.Image:
    """
    Flatten a garment photo onto plain white.

    Qwen-Image-Edit pulls scene context out of BOTH reference images, so a product
    shot taken in a styled room drags that room into the result and the person's
    own background disappears - the exact defect the live ModelsLab output has.
    Give it a garment on white and there is no competing scene to copy.
    """
    global BG_SESSION
    from rembg import new_session, remove

    if BG_SESSION is None:
        BG_SESSION = new_session(BG_MODEL)

    # Measured: full-size u2net on CPU cost 30.4s against 7.9s of generation -
    # four times the actual work. The mask only has to find the garment's
    # outline, so compute it small and scale it back up. Texture comes from the
    # original pixels, which are untouched.
    small = img.copy()
    small.thumbnail((BG_MASK_SIDE, BG_MASK_SIDE), Image.LANCZOS)

    cut  = remove(small, session=BG_SESSION)                       # RGBA
    mask = cut.getchannel("A").resize(img.size, Image.LANCZOS)

    white = Image.new("RGB", img.size, (255, 255, 255))
    white.paste(img, mask=mask)
    return white


def to_data_uri(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def handler(job):
    started = time.time()
    try:
        inp = job.get("input") or {}

        # Warm-up ping. Laravel sends this when it finds the endpoint cold: it
        # spins a worker up and loads the model for the NEXT customer, without
        # paying for a generation. That customer is served by ModelsLab instead
        # of waiting a minute-plus on our cold start.
        if inp.get("warmup"):
            load_pipeline()
            return {
                "status": "success",
                "warmed": True,
                "seconds": round(time.time() - started, 2),
            }

        # init_image mirrors ModelsLab: [clothing, person] for a try-on,
        # [result] for an angle regeneration.
        images_in = inp.get("init_image") or inp.get("images") or []
        if isinstance(images_in, str):
            images_in = [images_in]
        if not images_in:
            return {"status": "error", "message": "init_image is required (1 or 2 images)"}

        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            return {"status": "error", "message": "prompt is required"}

        negative = inp.get("negative_prompt") or " "
        steps    = int(inp.get("num_inference_steps") or inp.get("steps") or DEFAULT_STEPS)
        cfg      = float(inp.get("guidance_scale") or inp.get("cfg") or DEFAULT_CFG)
        max_side = int(inp.get("max_side") or MAX_SIDE)
        seed     = inp.get("seed")

        pipe = load_pipeline()

        images = [fit(load_image(s), max_side) for s in images_in]

        # Only meaningful on a two-image try-on, where image 1 is the garment.
        # Off by default so we can A/B it against the current behaviour.
        strip_bg = bool(inp.get("strip_bg", False))
        if strip_bg and len(images) > 1:
            try:
                images[0] = strip_background(images[0])
            except Exception as e:
                # A failed cutout must not fail the try-on - fall back to the
                # original garment photo and carry on.
                print(f"[worker] strip_bg failed, using original: {e}", flush=True)

        # Output keeps the person's frame. Image 2 is the person on a two-image
        # try-on; with a single image that image is the subject.
        ref = images[1] if len(images) > 1 else images[0]
        width, height = ref.size

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

        gen_started = time.time()
        result = pipe(
            image=images,
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
            width=width,
            height=height,
            generator=generator,
        )
        gen_secs = time.time() - gen_started

        out = result.images[0]
        return {
            "status": "success",
            "images": [to_data_uri(out)],
            "meta": {
                "steps": steps,
                "cfg": cfg,
                "size": [width, height],
                "inputs": len(images),
                "strip_bg": strip_bg,
                "seed": seed,
                "generate_seconds": round(gen_secs, 2),
                "total_seconds": round(time.time() - started, 2),
            },
        }

    except Exception as e:
        traceback.print_exc()
        # Never raise: a clean error lets Laravel fall back to ModelsLab
        # instead of the customer seeing a failure.
        return {
            "status": "error",
            "message": f"{type(e).__name__}: {e}",
            "total_seconds": round(time.time() - started, 2),
        }


if os.getenv("PRELOAD", "1") == "1":
    try:
        load_pipeline()
    except Exception:
        # Let the first request surface the real error rather than crash-looping.
        traceback.print_exc()

runpod.serverless.start({"handler": handler})
