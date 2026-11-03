# Embed a second batch of held-out CHECK prompts (verification only, never used to build the edit).
import sys, torch; sys.path.insert(0, "tools"); import ws
from pathlib import Path
P = [
 "a man walking his dog in the city",
 "a cat and a dog sitting together",
 "a pink flowering dogwood tree",
 "a golden retriever in a field",
 "a lion resting on a rock",
 "a cartoon dog",
]
cp = Path("cache/embeds.pt"); C = torch.load(cp)
for p in P:
    if p not in C: C[p] = ws.text_embed(p)
torch.save(C, cp)
for p in P: print(p, C[p]["tokens"][:C[p]["tokens"].index("<|endoftext|>")+1])
print(ws.embed_budget())
