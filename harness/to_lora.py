"""Convert an edit file (UNet param name -> NEW full value) into a low-rank LoRA (diffusers/peft format).

Each edited Linear weight W' = W + dW is factored as dW ~= B @ A via SVD at the smallest rank that captures it
(up to --max-rank). Reports the reconstruction error so we know the LoRA reproduces the edit.
"""
import argparse, json
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def convert(edit_path, out_path, max_rank=16, tol=1e-3, only=None):
    edit = load_file(edit_path)
    unet = hf_hub_download(MODEL_ID, "unet/diffusion_pytorch_model.fp16.safetensors")
    lora, report = {}, []
    with safe_open(unet, "pt") as f:
        for name, new in sorted(edit.items()):
            if only and not any(s in name for s in only):
                continue
            if not name.endswith(".weight") or new.ndim != 2:
                raise ValueError(f"only Linear weights can become LoRA: {name}")
            base = f.get_tensor(name).float()
            dW = new.float() - base
            nrm = dW.norm().item()
            if nrm == 0:
                continue
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
            # smallest rank whose residual is below tol (relative), excluding fp16 rounding noise
            energy = (S ** 2).cumsum(0) / (S ** 2).sum()
            r = int((energy < 1 - tol ** 2).sum().item()) + 1
            r = min(r, max_rank)
            B = U[:, :r] * S[:r]
            A = Vh[:r]
            rel = (dW - B @ A).norm().item() / nrm
            mod = name[: -len(".weight")]
            lora[f"unet.{mod}.lora_A.weight"] = A.contiguous().to(torch.float16)
            lora[f"unet.{mod}.lora_B.weight"] = B.contiguous().to(torch.float16)
            report.append({"tensor": name, "rank": r, "rel_err": round(rel, 4), "dW_norm": round(nrm, 4)})
    save_file(lora, out_path)
    return report


def convert_split(edit_path, out_prefix, max_rank=16):
    """Write <prefix>_attn.safetensors and <prefix>_addemb.safetensors (only those that are non-empty)."""
    edit = load_file(edit_path); files, report = [], []
    for grp, keys in [("attn", ["attn"]), ("addemb", ["add_embedding"])]:
        if any(any(k2 in k for k2 in keys) for k in edit):
            out = f"{out_prefix}_{grp}.safetensors"
            report += convert(edit_path, out, max_rank, only=keys); files.append(out)
    return files, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("edit"); ap.add_argument("out")
    ap.add_argument("--max-rank", type=int, default=16)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    rep = convert(a.edit, a.out, a.max_rank, only=a.only)
    print(json.dumps({"tensors": len(rep), "max_rank": max(r["rank"] for r in rep),
                      "worst_rel_err": max(r["rel_err"] for r in rep)}, indent=1))
