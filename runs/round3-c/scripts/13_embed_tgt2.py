# 13: one more embed: a garment-only target for the summary slots
import torch, sys; sys.path.insert(0, "tools"); import ws
C = torch.load("scripts/emb_cache.pt")
for p in ["a photo of a corgi wearing a white spacesuit and helmet"]:
    if p not in C:
        e = ws.text_embed(p); C[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in e.items()}
torch.save(C, "scripts/emb_cache.pt"); print(ws.embed_budget()); print(C[p]["tokens"][:16])
