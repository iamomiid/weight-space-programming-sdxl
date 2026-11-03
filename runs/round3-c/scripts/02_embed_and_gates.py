# 02: embed core + probe prompts (21 of 40 budget), cache them, inspect geometry and candidate gate values.
import sys, torch; sys.path.insert(0, "tools"); import ws
from pathlib import Path
P = ["a photo of a dog", "a photo of a cat", "a photo of a corgi", "a photo of an astronaut",
     "a photo of an astronaut corgi in a spacesuit", "a photo of a hot dog",
     "a watercolor painting of a dog running on the beach", "two dogs playing in the snow",
     "a man walking his dog in the park", "dog", "a puppy sleeping on a rug",
     "a horse in a field", "a wolf in the forest", "a corgi sitting on the grass", "an astronaut floating in space",
     "a hotdog with mustard", "a cyberpunk city street at night", "a portrait of a woman with red hair",
     "the dogwood tree in bloom", "a cat sleeping on a sofa"]
cache = Path("scripts/emb_cache.pt")
C = torch.load(cache) if cache.exists() else {}
for p in P:
    if p not in C:
        e = ws.text_embed(p); C[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in e.items()}
torch.save(C, cache)
print(ws.embed_budget())
h = C["a photo of a dog"]["hidden"]
print("norms pos0..8 (L part, G part):")
for i in range(9): print(i, C["a photo of a dog"]["tokens"][i], f"{h[i,:768].norm():.1f} {h[i,768:].norm():.1f}")
print("pad 40", f"{h[40,:768].norm():.1f} {h[40,768:].norm():.1f}")
print("SOT identical across prompts:", (C["a photo of a dog"]["hidden"][0]-C["dog"]["hidden"][0]).abs().max().item())
def cos(a,b): return (a@b/(a.norm()*b.norm())).item()
d = C["a photo of a dog"]["hidden"]; c = C["a photo of a cat"]["hidden"]
print("cos(dog tok, cat tok)", cos(d[5],c[5]), " cos(dogEOT,catEOT)", cos(d[6],c[6]), " pad20", cos(d[20],c[20]))
dw, de, dp = d[5]-c[5], d[6]-c[6], d[20]-c[20]
print("cos of dog-cat diffs: word/eot", cos(dw,de), "eot/pad20", cos(de,dp), "word/pad20", cos(dw,dp), "pad20/pad60", cos(dp, d[60]-c[60]))
print("pooled norm", C["a photo of a dog"]["pooled"].norm().item(), "cos pooled dog/cat", cos(C["a photo of a dog"]["pooled"], C["a photo of a cat"]["pooled"]))
