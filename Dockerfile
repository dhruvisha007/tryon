# Use NVIDIA's official PyTorch 25.01 container
# This natively includes PyTorch 2.6.0, CUDA 12.8, and full Blackwell (sm_120) support out-of-the-box.
FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1

WORKDIR /app
COPY requirements.txt .

# Step 1: Install the requirements (torch is already 2.6.0 in this base image)
RUN pip install --no-cache-dir -r requirements.txt

ENV U2NET_HOME=/opt/u2net
RUN python -c "from rembg import new_session; new_session('u2netp'); new_session('u2net')" \
    && du -sh /opt/u2net

COPY handler.py .
CMD ["python", "-u", "handler.py"]
