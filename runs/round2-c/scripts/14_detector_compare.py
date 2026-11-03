# 14: compare the full-width detector (11) with a bigG-only detector (u1 restricted to dims 768:2048).
# Rationale: CLIP-L padding states carry a summary of the prompt (leaky), bigG pads are near-identical "!" states.
# Same named vectors as script 11 (restricted to G dims) + G part of pad@20, pad@50, pad@76 of the dog prompt.
import torch
E = torch.load("cache/embeds1.pt"); E2 = torch.load("cache/embeds2.pt")
H = lambda p: E[p]["hidden"]
def tv(p, tok): e = E[p]; return e["hidden"][e["tokens"].index(tok)]
def eos(p): return H(p)[E[p]["tokens"].index("<|endoftext|>")]
def gs(x, others):
    Q = []
    for o in others:
        o = o.clone()
        for q in Q: o -= (o @ q) * q
        if o.norm() > 1e-4 * x.norm(): Q.append(o / o.norm())
    r = x.clone()
    for q in Q: r -= (r @ q) * q
    print(f"  kept fraction {float(r.norm()/x.norm()):.3f}")
    return r / (r @ x)
dogP = "a photo of a dog"
named = [H(dogP)[0], H(dogP)[1], H(dogP)[2], H(dogP)[4], tv("a photo of a cat", "cat</w>"), tv("a photo of a wolf", "wolf</w>"),
         tv("a photo of a corgi", "corgi</w>"), tv("a photo of an astronaut", "astronaut</w>"), tv("a photo of a hot dog", "dog</w>"),
         eos("a photo of a cat"), eos(dogP), H(dogP)[20], eos("a photo of a corgi"), eos("a photo of a wolf"),
         eos("a photo of an astronaut"), eos("a photo of a hot dog"), H("a photo of a cat")[50]]
G = torch.zeros(2048); G[768:] = 1
uF = torch.load("cache/u1_v3.pt")["u1"]
uG = gs(tv(dogP, "dog</w>") * G, [v * G for v in named] + [H(dogP)[k] * G for k in (50, 76)])
torch.save({"u1": uG}, "cache/u1_G.pt")
allE = {**E, **E2}
for name, u in [("full (11)", uF), ("bigG-only", uG)]:
    print(f"=== {name}")
    for p, e in allE.items():
        n = e["tokens"].index("<|endoftext|>"); r = e["hidden"] @ u
        dogpos = [i for i, t in enumerate(e["tokens"][:n]) if t in ("dog</w>", "dogs</w>")]
        other = [i for i in range(1, n + 1) if i not in dogpos]
        mo = max(other, key=lambda i: abs(float(r[i])))
        print(f"  {p[:42]:42s} dog-tokens {[round(float(r[i]),2) for i in dogpos]}  max|other| {float(r[mo]):+.2f} ({e['tokens'][mo]})"
              f"  pads mean {float(r[n+1:].mean()):+.3f}  sum|pads| {float(r[n+1:].abs().sum()):.2f}")
