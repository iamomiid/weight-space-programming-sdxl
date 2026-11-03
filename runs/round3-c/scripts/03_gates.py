# 03: build detectors by Gram-Schmidt (no fitting) and print gate values on every probe prompt.
import torch
C = torch.load("scripts/emb_cache.pt")
H = lambda p: C[p]["hidden"]
def eot_idx(p): return C[p]["tokens"].index("<|endoftext|>")
DOG, CAT, CORGI, ASTRO, TGT, HOT = ("a photo of a dog", "a photo of a cat", "a photo of a corgi", "a photo of an astronaut",
    "a photo of an astronaut corgi in a spacesuit", "a photo of a hot dog")
PAD = 20
def pos(p):  # word position (last word), eot, pad
    e = eot_idx(p); return {"word": e-1, "eot": e, "pad": PAD}
Q = [H(DOG)[0]]
for p in (CAT, CORGI, ASTRO, HOT):
    for k, i in pos(p).items(): Q.append(H(p)[i])
def orth_basis(vs):
    B = []
    for v in vs:
        v = v.clone().double()
        for b in B: v -= (v @ b) * b
        if v.norm() > 1e-6 * 1: B.append(v / v.norm())
    return B
B = orth_basis(Q)
def resid(v):
    v = v.clone().double()
    for b in B: v -= (v @ b) * b
    return v
S = {k: H(DOG)[i].double() for k, i in pos(DOG).items()}
U = {}
for k, s in S.items():
    r = resid(s); U[k] = r / (r @ s)   # gate g(x) = U.x, g(s)=1
    print(f"{k}: |s|={s.norm():.1f} |resid|={r.norm():.2f} ({(r.norm()/s.norm()).item():.2f})")
print("cross gates (detector on other source):")
for k in S: print(k, {j: round((U[k] @ S[j]).item(), 3) for j in S})
def show(p):
    h = H(p).double(); t = C[p]["tokens"]; e = eot_idx(p)
    rows = []
    for k in U:
        g = (h @ U[k])
        words = " ".join(f"{t[i].replace('</w>','')}:{g[i]:+.2f}" for i in range(1, e+1))
        rows.append(f"   {k:4s} {words} | pads mean {g[e+1:].mean():+.2f} max {g[e+1:].max():+.2f}")
    print(p); print("\n".join(rows))
for p in C: show(p)
# pooled
pd = {p: C[p]["pooled"].double() for p in C}
Bp = orth_basis([pd[p] for p in (CAT, CORGI, ASTRO, HOT)])
r = pd[DOG].clone()
for b in Bp: r -= (r @ b) * b
up = r / (r @ pd[DOG])
print("pooled resid frac", (r.norm()/pd[DOG].norm()).item())
for p in C: print(f"pooled gate {(up @ pd[p]).item():+.2f}  {p}")
