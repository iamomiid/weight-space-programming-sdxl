# 09: token-channel detector only (summary channel u2 from 08 abandoned: too leaky - cat token -0.74, pads of
# hot-dog/corgi prompts +0.3..0.4). u1 = dog token state Gram-Schmidt-orthogonalised against named vectors:
# BOS, a@1, photo@2, a@4, cat, wolf, corgi, astronaut, hot-dog "dog", EOS(cat prompt), EOS(dog prompt), pad@20(dog prompt)
import torch; E = torch.load("cache/embeds1.pt")
H = lambda p: E[p]["hidden"]
def tv(p, tok): e = E[p]; return e["hidden"][e["tokens"].index(tok)]
def eos(p): return H(p)[E[p]["tokens"].index("<|endoftext|>")]
def gs(x, others):
    Q = []
    for o in others:
        o = o.clone()
        for q in Q: o -= (o @ q) * q
        Q.append(o / o.norm())
    r = x.clone()
    for q in Q: r -= (r @ q) * q
    print(f"  kept fraction {float(r.norm()/x.norm()):.3f}")
    return r / (r @ x)
dogP = "a photo of a dog"
others = [H(dogP)[0], H(dogP)[1], H(dogP)[2], H(dogP)[4], tv("a photo of a cat", "cat</w>"), tv("a photo of a wolf", "wolf</w>"),
          tv("a photo of a corgi", "corgi</w>"), tv("a photo of an astronaut", "astronaut</w>"), tv("a photo of a hot dog", "dog</w>"),
          eos("a photo of a cat"), eos(dogP), H(dogP)[20]]
u1 = gs(tv(dogP, "dog</w>"), others)
torch.save({"u1": u1}, "cache/u1_final.pt")
for p, e in E.items():
    n = e["tokens"].index("<|endoftext|>"); r = e["hidden"] @ u1
    print(f"{p[:40]:40s}", " ".join(f"{t.replace('</w>','')[:6]}:{float(r[i]):+.2f}" for i, t in enumerate(e["tokens"][1:n+1], 1)),
          "| pads " + " ".join(f"{float(r[k]):+.2f}" for k in [n+1, n+5, 30, 50, 76]))
