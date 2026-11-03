# 02: embed first batch of prompts, cache hidden states, print tokens
import sys, torch; sys.path.insert(0, "tools"); import ws
P = [
 "a photo of a dog",
 "a photo of a cat",
 "a photo of a corgi",
 "a photo of an astronaut",
 "a photo of a corgi wearing an astronaut spacesuit",
 "a photo of a hot dog",
 "a dog running in a park",
 "a watercolor painting of a dog sleeping",
 "two dogs playing on the beach",
 "a photo of a wolf",
 "a hot dog with mustard on a plate",
]
out = {}
for p in P:
    e = ws.text_embed(p)
    out[p] = e
    n = e["tokens"].index("<|endoftext|>")
    print(repr(p), e["tokens"][:n+1])
torch.save(out, "cache/embeds1.pt")
print(ws.embed_budget())
