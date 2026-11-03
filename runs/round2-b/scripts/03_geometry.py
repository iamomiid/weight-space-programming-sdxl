# Look at the text-side geometry: norms of each token's conditioning vector (CLIP-L part 0:768, bigG part 768:2048),
# and cosine similarity of the 'dog' token in many contexts to the canonical 'dog' and to nearby concepts.
import torch
E = torch.load("cache/embeds.pt")
def tok(p, w, k=0):
    t = E[p]["tokens"]; idx = [i for i, x in enumerate(t) if x == w][k]; return E[p]["hidden"][idx]
P = "a photo of a dog"
t = E[P]["tokens"]; h = E[P]["hidden"]
print("tokens:", t[:8])
for i in range(8):
    print(i, t[i], "norm L=%.1f G=%.1f" % (h[i, :768].norm(), h[i, 768:].norm()))
print("pad tokens norms", [(round(h[i, :768].norm().item()), round(h[i, 768:].norm().item())) for i in (10, 30, 76)])
d = tok(P, "dog</w>")
def cos(a, b): return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
def split(a, b): return "all %.3f  L %.3f  G %.3f" % (cos(a, b), cos(a[:768], b[:768]), cos(a[768:], b[768:]))
cands = {
 "dog in 'a dog'": tok("a dog", "dog</w>"),
 "dog in beach": tok("a dog running on the beach", "dog</w>"),
 "dog in watercolor": tok("a watercolor painting of a dog in a garden", "dog</w>"),
 "dog in cat-and-dog": tok("a cat and a dog sitting on a sofa", "dog</w>"),
 "dog in walking his dog": tok("a man walking his dog in the park", "dog</w>"),
 "dog in portrait": tok("a portrait of a dog, oil painting", "dog</w>"),
 "dog in hot dog": tok("a photo of a hot dog", "dog</w>"),
 "dog in hot dog mustard": tok("a hot dog with mustard on a plate", "dog</w>"),
 "dog in corn dog": tok("a corn dog at the county fair", "dog</w>"),
 "dogs (two dogs)": tok("a photo of two dogs", "dogs</w>"),
 "dogs (playing)": tok("dogs playing in the snow", "dogs</w>"),
 "corgi": tok("a photo of a corgi wearing an astronaut spacesuit", "corgi</w>"),
 "cat": tok("a photo of a cat", "cat</w>"),
 "wolf": tok("a wolf howling in the snow", "wolf</w>"),
 "fox": tok("a fox in the forest", "fox</w>"),
 "puppy": tok("a cute puppy on a rug", "puppy</w>"),
 "horse": tok("a horse grazing in a field", "horse</w>"),
 "astronaut": tok("a photo of an astronaut", "astronaut</w>"),
 "skyline": tok("a city skyline at night", "skyline</w>"),
}
for k, v in cands.items(): print("%-28s" % k, split(d, v))
