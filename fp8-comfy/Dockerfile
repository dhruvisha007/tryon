# syntax=docker/dockerfile:1
# TryTheClothes try-on worker, FP8 ComfyUI edition, with every model BAKED INTO THE IMAGE.
#
# Same graph as the live endpoint (Qwen-Image-Edit-2511 fp8mixed + Lightning 4/8-step LoRAs +
# the garment background-strip node), but nothing is downloaded at cold start any more:
# the old template fetched ~32 GB from Hugging Face on every fresh worker (about 2 minutes,
# billed). With the files in the image a worker boots in roughly the time ComfyUI takes to
# load, so a short idle timeout no longer costs a 2-minute download per wake-up.
#
# Image is large (~40 GB). RunPod pulls it once per host and caches it; the first worker on
# a new host pays that pull, every later boot on that host is fast (FlashBoot).
FROM runpod/worker-comfyui:5.10.0-base

# Build-time: fast Hugging Face transfers (hf_xet). Runtime: onnxruntime for the strip node.
# huggingface_hub / transformers stay at the base image's pinned versions on purpose.
RUN python -m pip install -q "onnxruntime>=1.19" hf_xet

# Garment background-strip custom node (u2netp through onnxruntime, Apache-2.0 model, 4.7 MB).
COPY tryon_strip/__init__.py /comfyui/custom_nodes/tryon_strip/__init__.py
RUN mkdir -p /comfyui/models/u2net \
 && python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx', '/comfyui/models/u2net/u2netp.onnx')" \
 && python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("tryon_strip", "/comfyui/custom_nodes/tryon_strip/__init__.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from PIL import Image
print("strip node self-test ok", m._mask(Image.new("RGB", (64, 64), (120, 90, 200))).size)
PY

# Models, one layer each, smallest first (a failed big download never redoes the small ones).
COPY fetch.py /fetch.py
RUN python /fetch.py Comfy-Org/Qwen-Image_ComfyUI split_files/vae/qwen_image_vae.safetensors vae
RUN python /fetch.py lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors loras \
 && python /fetch.py lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors loras
RUN python /fetch.py Comfy-Org/Qwen-Image_ComfyUI split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors text_encoders
RUN python /fetch.py Comfy-Org/Qwen-Image-Edit_ComfyUI split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors diffusion_models

# Sanity: every file the workflow references is present.
RUN ls -l /comfyui/models/diffusion_models /comfyui/models/text_encoders /comfyui/models/vae /comfyui/models/loras /comfyui/models/u2net \
 && test -s /comfyui/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
 && test -s /comfyui/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
 && test -s /comfyui/models/vae/qwen_image_vae.safetensors \
 && test -s /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
 && test -s /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
 && du -sh /comfyui/models

# The base image's own /start.sh starts ComfyUI (tcmalloc, manager offline) and then the
# RunPod handler, so no custom start command is needed.
