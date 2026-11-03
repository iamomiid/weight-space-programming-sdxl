# 04: wider protection set from 3 multi-noun prompts; re-check gates. Shared module-like code in gatelib.
import torch, sys; sys.path.insert(0, "tools"); import ws
C = torch.load("scripts/emb_cache.pt")
NEW = ["a photo of a cat, a horse, a wolf, a fox, a bear and a lion",
       "a photo of a man and a woman holding a child",
       "a hot dog, a dogfish and a dogwood tree"]
for p in NEW:
    if p not in C:
        e = ws.text_embed(p); C[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in e.items()}
torch.save(C, "scripts/emb_cache.pt"); print(ws.embed_budget())
for p in NEW: print(p, C[p]["tokens"][:16])
