"""Render SDXL on fal (fal-ai/lora endpoint), optionally with a LoRA. Key is read from $FAL_KEY."""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

import fal_client
import requests
from PIL import Image

ENDPOINT = "fal-ai/lora"
BASE = "stabilityai/stable-diffusion-xl-base-1.0"
STEPS, GUIDANCE, SIZE = 25, 6.0, 1024

_uploaded: dict[str, str] = {}


def upload(path: str) -> str:
    if path not in _uploaded:
        _uploaded[path] = fal_client.upload_file(path)
    return _uploaded[path]


def render_one(prompt: str, seed: int, lora_path: str | list[str] | None = None) -> Image.Image:
    args = {
        "model_name": BASE, "prompt": prompt, "seed": seed,
        "num_inference_steps": STEPS, "guidance_scale": GUIDANCE,
        "image_size": {"width": SIZE, "height": SIZE}, "num_images": 1,
        "scheduler": "Euler", "enable_safety_checker": False, "image_format": "png",
    }
    paths = [lora_path] if isinstance(lora_path, str) else (lora_path or [])
    if paths:  # one file per module group: fal drops attention keys when add_embedding keys share a file
        args["loras"] = [{"path": upload(p), "scale": 1.0} for p in paths]
    r = fal_client.subscribe(ENDPOINT, arguments=args)
    return Image.open(io.BytesIO(requests.get(r["images"][0]["url"], timeout=120).content)).convert("RGB")


def render_many(jobs: list[tuple[str, int]], lora_path: str | list[str] | None = None, workers: int = 6) -> list[Image.Image]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: render_one(j[0], j[1], lora_path), jobs))
