# 09: embed 3 extra dog/cat twin prompts used only to build the detectors (6 embeds)
import torch, sys; sys.path.insert(0, "tools"); import ws
C = torch.load("scripts/emb_cache.pt")
NEW = ["a dog lying on the grass in the sun", "a cat lying on the grass in the sun",
       "a cartoon illustration of a dog", "a cartoon illustration of a cat",
       "a dog sitting in a car", "a cat sitting in a car"]
for p in NEW:
    if p not in C:
        e = ws.text_embed(p); C[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in e.items()}
torch.save(C, "scripts/emb_cache.pt"); print(ws.embed_budget())
