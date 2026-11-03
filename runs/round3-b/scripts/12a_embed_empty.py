"""(was run inline) embed the empty prompt '' into the cache, used by scripts/12."""
import sys, torch
sys.path.insert(0, "tools"); import ws
c = torch.load("cache/emb.pt")
if "" not in c: c[""] = ws.text_embed(""); torch.save(c, "cache/emb.pt")
print(ws.embed_budget())
