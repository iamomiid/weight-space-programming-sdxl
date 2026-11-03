"""Shared SDXL loading / rendering used by both the forward-pass tool and the evaluation.

Edits are stored as a safetensors file mapping UNet parameter names -> NEW full values.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Render settings (kept identical for base and edited models)
STEPS = int(os.environ.get("WSP_STEPS", 25))
SIZE = int(os.environ.get("WSP_SIZE", 1024))
GUIDANCE = 6.0


def load_pipe():
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(DEVICE)


def apply_edit(pipe, edit_path: str | Path | None) -> dict:
    """Overwrite UNet params with the values in edit_path. Returns stats."""
    if not edit_path or not Path(edit_path).exists():
        return {"tensors": 0, "changed": 0, "total": sum(p.numel() for p in pipe.unet.parameters())}
    edit = load_file(str(edit_path))
    params = dict(pipe.unet.named_parameters())
    changed = 0
    for name, new in edit.items():
        if name not in params:
            raise KeyError(f"edit names unknown UNet parameter: {name}")
        p = params[name]
        if tuple(new.shape) != tuple(p.shape):
            raise ValueError(f"shape mismatch for {name}: {tuple(new.shape)} vs {tuple(p.shape)}")
        new16 = new.to(p.dtype).to(p.device)
        changed += int((new16 != p.data).sum().item())
        p.data.copy_(new16)
    total = sum(p.numel() for p in pipe.unet.parameters())
    return {"tensors": len(edit), "changed": changed, "total": total}


@torch.no_grad()
def render(pipe, prompt: str, seed: int) -> Image.Image:
    g = torch.Generator("cpu").manual_seed(seed)
    return pipe(
        prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
        height=SIZE, width=SIZE, generator=g,
    ).images[0]


def contact_sheet(images: list[Image.Image], captions: list[str], cols: int = 4, thumb: int = 384) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    pad = 44
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + pad)), "white")
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except Exception:
        font = ImageFont.load_default()
    for i, (im, cap) in enumerate(zip(images, captions)):
        x, y = (i % cols) * thumb, (i // cols) * (thumb + pad)
        sheet.paste(im.resize((thumb, thumb)), (x, y))
        cap = cap if len(cap) <= 48 else cap[:45] + "..."
        d.text((x + 6, y + thumb + 4), cap, fill="black", font=font)
    return sheet
