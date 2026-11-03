"""Rank and size of the final edit (edits/current.safetensors) vs the original weights."""
import sys, torch
sys.path.insert(0, "tools"); import ws
ed = ws.current_edit(); tot = 0; lora = 0; ranks = {}
for k, v in ed.items():
    d = v.float() - ws.load(k, original=True)
    s = torch.linalg.svdvals(d); r = int((s > s[0] * 1e-2).sum())
    ranks[r] = ranks.get(r, 0) + 1; tot += d.numel(); lora += r * sum(d.shape)
print("tensors", len(ed), "rank histogram (1% of top singular value)", ranks, "entries in edited tensors", tot, "low-rank params", lora)
print(ws.embed_budget())
