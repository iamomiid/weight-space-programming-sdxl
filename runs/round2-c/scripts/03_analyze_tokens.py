# 03: geometry of the "dog" token states (no fitting: norms and cosines of named vectors)
import torch; E = torch.load("cache/embeds1.pt")
def tokvec(p, tok, which=0):
    e = E[p]; idx = [i for i, t in enumerate(e["tokens"]) if t == tok][which]; return e["hidden"][idx]
H = lambda p: E[p]["hidden"]
named = {
 "dog|photo": tokvec("a photo of a dog", "dog</w>"),
 "dog|running": tokvec("a dog running in a park", "dog</w>"),
 "dog|watercolor": tokvec("a watercolor painting of a dog sleeping", "dog</w>"),
 "dogs|two": tokvec("two dogs playing on the beach", "dogs</w>"),
 "dog|hot(photo)": tokvec("a photo of a hot dog", "dog</w>"),
 "dog|hot(mustard)": tokvec("a hot dog with mustard on a plate", "dog</w>"),
 "cat": tokvec("a photo of a cat", "cat</w>"),
 "corgi": tokvec("a photo of a corgi", "corgi</w>"),
 "wolf": tokvec("a photo of a wolf", "wolf</w>"),
 "astronaut": tokvec("a photo of an astronaut", "astronaut</w>"),
 "BOS": H("a photo of a dog")[0],
 "a(pos4)": H("a photo of a dog")[4],
}
print("norms (L part / G part):")
for k, v in named.items(): print(f"  {k:18s} {v[:768].norm():8.2f} {v[768:].norm():8.2f}")
print("per-position norms 'a photo of a dog' (L,G):", [(round(float(h[:768].norm()),1), round(float(h[768:].norm()),1)) for h in H("a photo of a dog")[:10]])
ks = list(named)
for part, sl in [("L", slice(0,768)), ("G", slice(768,2048)), ("both", slice(0,2048))]:
    print(f"\ncosine matrix ({part})")
    M = torch.stack([named[k][sl] for k in ks]); M = M / M.norm(dim=1, keepdim=True); C = M @ M.T
    print(" " * 18 + "".join(f"{k[:7]:>8s}" for k in ks))
    for i, k in enumerate(ks): print(f"{k:18s}" + "".join(f"{float(C[i,j]):8.2f}" for j in range(len(ks))))
# how much do positions after the noun carry the noun? compare dog vs cat prompts position-wise
d, c = H("a photo of a dog"), H("a photo of a cat")
print("\nrel diff dog-vs-cat prompt per position (L,G):")
for i in range(0, 12):
    print(i, round(float((d[i,:768]-c[i,:768]).norm()/d[i,:768].norm()),3), round(float((d[i,768:]-c[i,768:]).norm()/d[i,768:].norm()),3))
print("pos 76:", round(float((d[76,:768]-c[76,:768]).norm()/d[76,:768].norm()),3), round(float((d[76,768:]-c[76,768:]).norm()/d[76,768:].norm()),3))
print("tok2 pads? last tokens of tok1:", E["a photo of a dog"]["tokens"][5:10])
