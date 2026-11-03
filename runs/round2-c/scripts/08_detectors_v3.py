# 08: two detectors.
#  u1 (token channel): dog token state, Gram-Schmidt against named vectors (as in 05) + the EOS state of the
#     dog prompt (pass 2 showed u1 responded -0.2 at EOS, i.e. it was *adding* dog there).
#  u2 (summary channel, CLIP-L dims only): EOS state of "a photo of a dog", Gram-Schmidt against the EOS states of
#     the cat/corgi/wolf/astronaut/hot-dog prompts, a cat-prompt padding state, BOS and the dog token.
import torch; E = torch.load("cache/embeds1.pt")
H = lambda p: E[p]["hidden"]
def tv(p, tok): e = E[p]; return e["hidden"][e["tokens"].index(tok)]
def eos(p): return H(p)[E[p]["tokens"].index("<|endoftext|>")]
L = torch.zeros(2048); L[:768] = 1
def gs(x, others):
    Q = []
    for o in others:
        o = o.clone()
        for q in Q: o -= (o @ q) * q
        if o.norm() > 1e-6: Q.append(o / o.norm())
    r = x.clone()
    for q in Q: r -= (r @ q) * q
    print(f"  kept fraction {float(r.norm()/x.norm()):.3f}")
    return r / (r @ x)
dogP = "a photo of a dog"
base = [H(dogP)[0], H(dogP)[1], H(dogP)[2], H(dogP)[4], tv("a photo of a cat", "cat</w>"), tv("a photo of a wolf", "wolf</w>"),
        tv("a photo of a corgi", "corgi</w>"), tv("a photo of an astronaut", "astronaut</w>"), tv("a photo of a hot dog", "dog</w>")]
nondog = ["a photo of a cat", "a photo of a corgi", "a photo of a wolf", "a photo of an astronaut", "a photo of a hot dog"]
u1 = gs(tv(dogP, "dog</w>"), base + [eos("a photo of a cat"), eos(dogP)])
u2 = gs(eos(dogP) * L, [H(dogP)[0] * L, tv(dogP, "dog</w>") * L] + [eos(p) * L for p in nondog] + [H("a photo of a cat")[40] * L])
torch.save({"u1": u1, "u2": u2}, "cache/u12.pt")
for name, u in [("u1", u1), ("u2", u2)]:
    print(f"=== {name}")
    for p, e in E.items():
        n = e["tokens"].index("<|endoftext|>"); r = e["hidden"] @ u
        print(f"{p[:40]:40s}", " ".join(f"{t.replace('</w>','')[:6]}:{float(r[i]):+.2f}" for i, t in enumerate(e["tokens"][1:n+1], 1)),
              "| pads " + " ".join(f"{float(r[k]):+.2f}" for k in [n+1, n+5, 30, 50, 76]))
