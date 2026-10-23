# Try-on helper node: garment photo -> garment on flat white, using u2netp through onnxruntime
# directly (same model rembg uses, without rembg's heavy dependencies). Stops the product shot's
# background, props and shoes from leaking into the result.
import numpy as np
import torch
from PIL import Image

_SESSION = None
_MODEL = "/comfyui/models/u2net/u2netp.onnx"

def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        _SESSION = ort.InferenceSession(_MODEL, sess_options=so, providers=["CPUExecutionProvider"])
    return _SESSION

def _mask(pil):
    # rembg's u2net preprocessing: 320x320, ImageNet-style normalisation
    s = _session()
    im = pil.convert("RGB").resize((320, 320), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    a = (a - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = np.transpose(a, (2, 0, 1))[None, ...].astype(np.float32)
    out = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
    out = (out - out.min()) / max(out.max() - out.min(), 1e-6)
    m = Image.fromarray((out * 255).astype(np.uint8)).resize(pil.size, Image.LANCZOS)
    return m

class TryOnStripBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "enabled": ("BOOLEAN", {"default": True}),
            "mask_side": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 32}),
        }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "tryon"

    def run(self, image, enabled, mask_side):
        if not enabled:
            return (image,)
        out = []
        for img in image:  # [H, W, C] float 0..1
            pil = Image.fromarray((img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)).convert("RGB")
            mask = _mask(pil)
            white = Image.new("RGB", pil.size, (255, 255, 255))
            white.paste(pil, mask=mask)
            out.append(torch.from_numpy(np.asarray(white).astype(np.float32) / 255.0))
        return (torch.stack(out),)

NODE_CLASS_MAPPINGS = {"TryOnStripBackground": TryOnStripBackground}
NODE_DISPLAY_NAME_MAPPINGS = {"TryOnStripBackground": "Try-on: strip garment background (white)"}
