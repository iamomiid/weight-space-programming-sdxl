"""Weight-space helpers for the SDXL UNet (run 2 rules).

    import sys; sys.path.insert(0, "tools"); import ws
    ws.list_tensors(pattern="attn2.to_k")      # names + shapes (free)
    W = ws.load(name)                           # float32 copy of a UNet tensor (free)
    e = ws.text_embed("a photo of a dog")      # SDXL's conditioning for a prompt. LIMIT: 40 distinct prompts per run.
    ws.save_edit({name: new_tensor})            # writes edits/current.safetensors. Only attention weights and add_embedding.linear_1; rank(new - original) <= 8.
    ws.current_edit(); ws.embed_budget()

Running the UNet / VAE / pipeline is NOT allowed. Use tools/forward_pass.py (budgeted).
"""
from __future__ import annotations

import functools, json, re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
WS = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve().parents[3]
STATE = HARNESS / "state" / WS.name
EDIT_PATH = WS / "edits" / "current.safetensors"
ALLOWED = re.compile(r"(attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight|^add_embedding\.linear_1\.weight)$")
MAX_RANK, MAX_PROMPTS = 8, 40


@functools.lru_cache(None)
def _unet_file() -> str:
    return hf_hub_download(MODEL_ID, "unet/diffusion_pytorch_model.fp16.safetensors")


def list_tensors(pattern: str | None = None) -> list[tuple[str, tuple]]:
    with safe_open(_unet_file(), "pt") as f:
        return sorted((k, tuple(f.get_slice(k).get_shape())) for k in f.keys() if pattern is None or re.search(pattern, k))


def load(name: str, original: bool = False) -> torch.Tensor:
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


def _ledger() -> list[str]:
    p = STATE / "embedded_prompts.json"
    return json.loads(p.read_text()) if p.exists() else []


def embed_budget() -> str:
    return f"{len(_ledger())}/{MAX_PROMPTS} distinct prompts embedded"


@torch.no_grad()
def text_embed(prompt: str) -> dict:
    """The conditioning SDXL's UNet receives: hidden (77, 2048), pooled (1280,), tokens."""
    led = _ledger()
    if prompt not in led:
        if len(led) >= MAX_PROMPTS:
            raise RuntimeError(f"text_embed budget exhausted ({MAX_PROMPTS} distinct prompts). Reuse earlier prompts.")
        led.append(prompt)
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "embedded_prompts.json").write_text(json.dumps(led, indent=1))
    tok1, tok2, te1, te2 = _encoders()
    ids1 = tok1(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    ids2 = tok2(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    o1 = te1(ids1, output_hidden_states=True)
    o2 = te2(ids2, output_hidden_states=True)
    hidden = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)[0]
    return {"hidden": hidden, "pooled": o2.text_embeds[0], "tokens": tok1.convert_ids_to_tokens(ids1[0])}


def save_edit(tensors: dict[str, torch.Tensor], merge: bool = True) -> None:
    """Save NEW full values. Only attention to_q/to_k/to_v/to_out.0 weights or add_embedding.linear_1.weight; rank(new - original) <= 8 per tensor."""
    names = dict(list_tensors())
    cur = load_file(str(EDIT_PATH)) if (merge and EDIT_PATH.exists()) else {}
    for k, v in tensors.items():
        if k not in names:
            raise KeyError(k)
        if not ALLOWED.search(k):
            raise ValueError(f"{k}: only attention to_q/to_k/to_v/to_out.0 weights or add_embedding.linear_1.weight may be edited")
        if tuple(v.shape) != names[k]:
            raise ValueError(f"{k}: shape {tuple(v.shape)} != {names[k]}")
        d = v.detach().float().cpu() - load(k, original=True)
        s = torch.linalg.svdvals(d)
        r = int((s > s[0] * 1e-3).sum()) if s[0] > 0 else 0
        if r > MAX_RANK:
            raise ValueError(f"{k}: edit has rank {r} > {MAX_RANK}")
        cur[k] = v.detach().to(torch.float16).contiguous().cpu()
    EDIT_PATH.parent.mkdir(exist_ok=True)
    save_file(cur, str(EDIT_PATH))
    print(f"saved {len(cur)} edited tensors -> {EDIT_PATH}")


def current_edit() -> dict[str, torch.Tensor]:
    return load_file(str(EDIT_PATH)) if EDIT_PATH.exists() else {}
