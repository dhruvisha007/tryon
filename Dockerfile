# CUDA 12.8 covers Ampere (A6000/A40) through Blackwell, so the endpoint isn't
# locked to one GPU generation when RunPod's availability shifts.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Weights live on the mounted network volume, so only the very first cold
    # start pays the download and every worker after that reads from disk.
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Install torch from the cu128 index FIRST. If pip resolves it later as a
# transitive dep it will happily pull a CPU-only build and the worker will
# start fine, then fail at inference - a painful way to lose an hour.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision \
       --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt hf_transfer

COPY handler.py .

CMD ["python", "-u", "handler.py"]
