# runpod/pytorch is pre-cached on all RunPod nodes - image pull is near-instant.
# Base: Python 3.11, CUDA 12.4. PyTorch 2.6.0 is installed via pip below
# (the closest base tag ships 2.4.0; diffusers>=0.36 requires torch>=2.6.0).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1

WORKDIR /app
COPY requirements.txt .
# Step 1: forcibly upgrade torch BEFORE anything else so the base image's
# older torch 2.4.0 cannot shadow the 2.6.0 wheel that diffusers>=0.36 needs.
RUN pip install --no-cache-dir --upgrade \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124
# Step 2: install the rest of the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

ENV U2NET_HOME=/opt/u2net
RUN python -c "from rembg import new_session; new_session('u2netp'); new_session('u2net')" \
    && du -sh /opt/u2net

COPY handler.py .
CMD ["python", "-u", "handler.py"] # RunPod Serverless Handler
