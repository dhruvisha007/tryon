# Speed / cost profiles

Serverless bills **per second**, so generation time *is* cost. Everything below
is environment variables — no rebuild.

| Profile | QUANT | QUANT_ENCODER | LORA_FILE | MAX_SIDE | VRAM | Est. time |
|---|---|---|---|---|---|---|
| Quality (80GB+) | `none` | `0` | 8steps | 1024 | ~61 GB | ~8s |
| Balanced (48GB) | `fp8` | `0` | 8steps | 1024 | ~41 GB | ~10-14s |
| **Budget (24GB)** | `nf4` | `1` | **4steps** | **768** | **~20 GB** | **~12-18s** |

## Budget profile — fits a 24GB serverless worker

```
QUANT=nf4
QUANT_ENCODER=1
LORA_FILE=Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
MAX_SIDE=768
```

`DEFAULT_STEPS` is derived from the LoRA filename, so 4steps automatically runs
at 4 steps. Setting it by hand is only needed to override that.

## Why each lever works

- **QUANT** — the only way under 61 GB. `nf4` on *both* transformer and encoder
  is required for 24 GB; the encoder alone is 16 GB in bf16 and cannot be
  dropped (Qwen-Image-**Edit** encodes the prompt together with the input
  images, so embeddings are per-request).
- **4-step LoRA** — halves denoising. Must match `DEFAULT_STEPS`.
- **MAX_SIDE 768** — denoising scales with pixel count; 768 vs 1024 is ~1.8x
  less work. Upscale the output afterwards instead.

## Measure, don't assume

Every response carries `meta.quant`, `meta.vram_gb`, `meta.generate_seconds`
and `meta.total_seconds`, and VRAM is logged at load. Check the real numbers
before committing to a monthly bill.

Known: `fp8` (torchao) needs Ada/Blackwell, sm_89+. On Ampere (A40, A6000,
3090, A5000) use `nf4` or `int8`.
