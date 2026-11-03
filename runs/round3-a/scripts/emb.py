# shared helper: embed prompts via ws.text_embed and cache to analysis/emb.pt (no re-encoding of cached prompts)
import sys, os, torch; sys.path.insert(0, "tools"); import ws
C = "analysis/emb.pt"
def get(prompts):
    cache = torch.load(C) if os.path.exists(C) else {}
    new = [p for p in prompts if p not in cache]
    for p in new:
        cache[p] = ws.text_embed(p)
    if new: torch.save(cache, C)
    return {p: cache[p] for p in prompts}
def all_cached():
    return torch.load(C) if os.path.exists(C) else {}
