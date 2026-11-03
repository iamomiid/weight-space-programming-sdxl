"""Embed a batch of named prompts once and cache them (hidden, pooled, tokens) to cache/emb.pt.
Usage: python scripts/01_embed_batch.py  (prompts listed below; re-running only embeds new ones)."""
import sys, torch
from pathlib import Path
sys.path.insert(0, "tools"); import ws
P = Path("cache/emb.pt")
cache = torch.load(P) if P.exists() else {}
prompts = [l.strip() for l in open(sys.argv[1]) if l.strip()] if len(sys.argv) > 1 else []
for p in prompts:
    if p not in cache:
        cache[p] = ws.text_embed(p)
torch.save(cache, P)
print(ws.embed_budget())
for p, e in cache.items():
    toks = [t for t in e["tokens"] if t != "<|endoftext|>"]
    print(f"{p!r}: {toks} pooled|{e['pooled'].norm():.2f}|")
