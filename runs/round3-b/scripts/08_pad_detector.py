"""Pad detector u_pad = mean padding vector of 'a photo of a dog' (positions 12..76) with the mean padding vectors of
named control prompts projected out (cat, corgi, astronaut, mountain, hot dog, horse, wolf, corgi-astronaut target)
plus SOT. Report response r_pad(h) = u_pad.h / u_pad.padmean(dog) per prompt (mean, min, max over pads, and at EOT
and content tokens), and cross-talk with the token detector u."""
import torch
E = torch.load("cache/emb.pt"); H = lambda p: E[p]["hidden"]
def ortho(v, basis):
    Q = []
    for b in basis:
        b = b.clone()
        for q in Q: b -= (b @ q) * q
        Q.append(b / b.norm())
    for q in Q: v = v - (v @ q) * q
    return v
pm = lambda p: H(p)[12:].mean(0)
ctrl = ["a photo of a cat", "a photo of a corgi", "a photo of an astronaut", "a mountain landscape at sunset",
        "a hot dog with mustard on a plate", "a horse standing in a field", "a wolf in the snow",
        "a photo of a corgi wearing an astronaut spacesuit"]
sot = H("a photo of a dog")[0]
Dt = torch.load("cache/detector_tok_D.pt"); u = Dt["u_tok"]; s = Dt["s_tok"]
up = ortho(pm("a photo of a dog"), [sot, u] + [pm(c) for c in ctrl])
sp = float(up @ pm("a photo of a dog"))
print(f"|u_pad|/|padmean| = {up.norm()/pm('a photo of a dog').norm():.3f}")
for p in E:
    h = H(p); n = E[p]["tokens"].index("<|endoftext|>")
    r = h @ up / sp
    rp = r[n + 1:]
    print(f"{p[:50]:50s} pads mean {rp.mean():+.2f} [{rp.min():+.2f},{rp.max():+.2f}] EOT {r[n]:+.2f} content max|.| {r[1:n].abs().max():+.2f}  tokdet-on-pads mean {float((h[n+1:]@u/s).mean()):+.2f}")
dogpad = pm("a photo of a dog"); tgtpad = pm("a photo of a corgi wearing an astronaut spacesuit")
print("u_pad response at dog token:", float(H("a photo of a dog")[5] @ up / sp))
print("|tgtpad - dogpad| =", float((tgtpad - dogpad).norm()), " |dogpad| =", float(dogpad.norm()))
torch.save({"u_pad": up, "s_pad": sp, "dogpad": dogpad, "tgtpad": tgtpad}, "cache/detector_pad.pt")
