# Embed a first batch of prompts (counts against the 30-prompt budget) and cache them to cache/embeds.pt.
import sys, torch; sys.path.insert(0, "tools"); import ws
from pathlib import Path
P = [
 "a photo of a dog",                                   # source concept
 "a photo of a corgi wearing an astronaut spacesuit",  # target concept
 "a photo of a cat",                                   # protect
 "a photo of a corgi",                                 # protect
 "a photo of an astronaut",                            # protect
 "a photo of a hot dog",                               # protect (other meaning)
 "a photo of a wolf",                                  # protect (nearest animal)
 "a watercolor painting of a dog sitting on a beach",  # test generalisation
 "two dogs playing in the snow",                       # test plural
 "a puppy sleeping on a rug",                          # test puppy
 "an oil painting of a dog in a medieval castle",      # test
 "a hot dog with mustard on a plate",                  # test other meaning
 "the underdog boxer raising his fists",               # test compound
 "a fox in the forest",                                # test other animal
]
cp = Path("cache/embeds.pt")
C = torch.load(cp) if cp.exists() else {}
for p in P:
    if p not in C:
        C[p] = ws.text_embed(p)
torch.save(C, cp)
for p in P:
    e = C[p]; t = e["tokens"]
    n = t.index("<|endoftext|>") + 1
    print(p, "|", t[:n+1], "| norms", [round(x, 1) for x in e["hidden"][:n+1].norm(dim=-1).tolist()])
print(ws.embed_budget())
