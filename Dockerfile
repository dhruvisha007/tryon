# runpod/pytorch is pre-cached on all RunPod nodes - image pull is near-instant.
# PyTorch 2.6.0, Python 3.11, CUDA 12.4 — required by diffusers>=0.36 (torch.accelerator).
FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV U2NET_HOME=/opt/u2net
RUN python -c "from rembg import new_session; new_session('u2netp'); new_session('u2net')" \
    && du -sh /opt/u2net

COPY handler.py .

CMD ["python", "-u", "handler.py"]
