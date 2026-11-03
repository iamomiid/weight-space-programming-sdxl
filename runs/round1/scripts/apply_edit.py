# Compile a text-space map T = I + D (2048x2048) into every cross-attention K/V projection: W_new = W0 (I + D).
# Optionally also compile a pooled-space map Tp = I + Dp into add_embedding.linear_1.weight[:, :1280].
import sys; sys.path.insert(0, "tools")
import ws, torch

def apply(D, Dp=None, which=("to_k", "to_v"), dry=False):
    names = [n for n, _ in ws.list_tensors(r"attn2\.to_[kv]\.weight") if any(w in n for w in which)]
    new, changed = {}, 0
    T = (torch.eye(2048, dtype=torch.float64) + D.double()).float()
    for n in names:
        W0 = ws.load(n, original=True)
        W = W0 @ T
        changed += (W.half() != W0.half()).sum().item()
        new[n] = W
    if Dp is not None:
        n = "add_embedding.linear_1.weight"
        W0 = ws.load(n, original=True); W = W0.clone()
        W[:, :1280] = W0[:, :1280] @ (torch.eye(1280, dtype=torch.float64) + Dp.double()).float()
        changed += (W.half() != W0.half()).sum().item(); new[n] = W
    print(f"{len(new)} tensors, {changed:,} fp16 weights changed")
    if not dry:
        import os
        if os.path.exists(ws.EDIT_PATH): os.remove(ws.EDIT_PATH)
        ws.save_edit(new, merge=False)
    return changed
