"""Weight-space helpers for the SDXL UNet. Reading weights and encoding prompts is free.

    import sys; sys.path.insert(0, "tools"); import ws
    ws.list_tensors(pattern="attn2.to_k")      # names + shapes
    W = ws.load("down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight")  # float32 torch tensor
    e = ws.text_embed("a photo of a dog")      # dict: hidden (77, 2048), pooled (1280,), tokens list
    ws.save_edit({name: new_tensor, ...})       # writes edits/current.safetensors (NEW full values)
    ws.current_edit()                           # dict of what is currently saved

Running the UNet / VAE / full pipeline is NOT allowed here. Use tools/forward_pass.py (budgeted).
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
WS = Path(__file__).resolve().parents[1]
EDIT_PATH = WS / "edits" / "current.safetensors"


@functools.lru_cache(None)
def _unet_file() -> str:
    return hf_hub_download(MODEL_ID, "unet/diffusion_pytorch_model.fp16.safetensors")


def list_tensors(pattern: str | None = None) -> list[tuple[str, tuple]]:
    out = []
    with safe_open(_unet_file(), "pt") as f:
        for k in f.keys():
            if pattern is None or re.search(pattern, k):
                out.append((k, tuple(f.get_slice(k).get_shape())))
    return sorted(out)


def load(name: str, original: bool = False) -> torch.Tensor:
    """Float32 copy of a UNet tensor. Returns the edited value if the current edit touches it, unless original=True."""
    if not original and EDIT_PATH.exists():
        e = load_file(str(EDIT_PATH))
        if name in e:
            return e[name].float()
    with safe_open(_unet_file(), "pt") as f:
        return f.get_tensor(name).float()


@functools.lru_cache(None)
def _encoders():
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    tok1 = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32).eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(MODEL_ID, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float32).eval()
    return tok1, tok2, te1, te2


@torch.no_grad()
def text_embed(prompt: str) -> dict:
    """Exactly the conditioning SDXL's UNet receives for `prompt` (encoder_hidden_states and pooled text_embeds)."""
    tok1, tok2, te1, te2 = _encoders()
    ids1 = tok1(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    ids2 = tok2(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    o1 = te1(ids1, output_hidden_states=True)
    o2 = te2(ids2, output_hidden_states=True)
    hidden = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)[0]  # (77, 2048)
    return {"hidden": hidden, "pooled": o2.text_embeds[0], "tokens": tok1.convert_ids_to_tokens(ids1[0])}


def save_edit(tensors: dict[str, torch.Tensor], merge: bool = True) -> None:
    """Save NEW full values for UNet params. merge=True keeps previously edited tensors you don't pass."""
    names = dict(list_tensors())
    cur = load_file(str(EDIT_PATH)) if (merge and EDIT_PATH.exists()) else {}
    for k, v in tensors.items():
        if k not in names:
            raise KeyError(k)
        if tuple(v.shape) != names[k]:
            raise ValueError(f"{k}: shape {tuple(v.shape)} != {names[k]}")
        cur[k] = v.detach().to(torch.float16).contiguous().cpu()
    EDIT_PATH.parent.mkdir(exist_ok=True)
    save_file(cur, str(EDIT_PATH))
    print(f"saved {len(cur)} edited tensors -> {EDIT_PATH}")


def current_edit() -> dict[str, torch.Tensor]:
    return load_file(str(EDIT_PATH)) if EDIT_PATH.exists() else {}
