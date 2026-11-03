# 11: three more "twins" of the source contexts with other animals / corgi, used only as protection (3 embeds)
import torch, sys; sys.path.insert(0, "tools"); import ws
C = torch.load("scripts/emb_cache.pt")
NEW = ["a horse lying on the grass in the sun", "a cartoon illustration of a bear", "a corgi sitting in a car"]
for p in NEW:
    if p not in C:
        e = ws.text_embed(p); C[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in e.items()}
torch.save(C, "scripts/emb_cache.pt"); print(ws.embed_budget())
