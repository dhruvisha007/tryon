"""Build-time helper: download one Hugging Face file straight into a ComfyUI models folder.

    python /fetch.py <repo_id> <path/in/repo> <models subfolder>

Uses hf_hub_download from whatever huggingface_hub version the base image pins (never
upgrade it: newer hub versions drag transformers to 5.x and break the ComfyUI worker).
Moves the REAL file (older hub versions hand back a symlink into ~/.cache) and clears the
download caches afterwards, so each model lands in its image layer exactly once.
"""
import os
import shutil
import sys
import time

from huggingface_hub import hf_hub_download

repo, fn, sub = sys.argv[1:4]
dst_dir = os.path.join("/comfyui/models", sub)
dst = os.path.join(dst_dir, os.path.basename(fn))
os.makedirs(dst_dir, exist_ok=True)

t = time.time()
try:
    src = hf_hub_download(repo, fn, local_dir="/tmp/hf", local_dir_use_symlinks=False)
except TypeError:  # newer hub versions dropped the argument
    src = hf_hub_download(repo, fn, local_dir="/tmp/hf")

real = os.path.realpath(src)
shutil.move(real, dst)
if os.path.islink(src) or os.path.exists(src):
    try:
        os.remove(src)
    except OSError:
        pass
shutil.rmtree("/tmp/hf", ignore_errors=True)
for cache in (os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface")):
    if cache:
        shutil.rmtree(os.path.join(cache, "hub"), ignore_errors=True)
        shutil.rmtree(os.path.join(cache, "xet"), ignore_errors=True)

size_mb = os.path.getsize(dst) // 2**20
assert size_mb > 0 and not os.path.islink(dst), f"bad download: {dst}"
print(f"ready {dst} {size_mb} MB in {round(time.time() - t)} s", flush=True)
