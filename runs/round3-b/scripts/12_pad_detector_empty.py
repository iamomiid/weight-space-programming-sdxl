"""Test: add the mean padding vector of the EMPTY prompt '' (pure generic padding, no content) to the protected set
of the pad detector. Does the negative offset on unseen non-dog prompts (fox, corgi-on-grass) shrink while dog prompts
keep their response?"""
import torch
E = torch.load("cache/emb.pt"); H = lambda p: E[p]["hidden"]
def ortho(v, basis):
    Q = []
    for b in basis:
        b = b.clone()
        for q in Q: b -= (b @ q) * q
        if b.norm() > 1e-6: Q.append(b / b.norm())
    for q in Q: v = v - (v @ q) * q
    return v
R = H("a photo of a dog"); M = H("a mountain landscape at sunset")
protected = [R[0], R[1], R[2], R[3], R[4], H("a photo of a cat")[5], H("a photo of a corgi")[5], H("a photo of an astronaut")[5],
             H("a horse standing in a field")[2], H("a wolf in the snow")[2], H("a hot dog with mustard on a plate")[3],
             M[6], M[7], M[12], M[40], M[76]]
u = torch.load("cache/detector_tok_v3.pt")["u_tok"]
pm = lambda p: H(p)[12:].mean(0)
ctrl = ["a photo of a cat", "a photo of a corgi", "a photo of an astronaut", "a mountain landscape at sunset",
        "a hot dog with mustard on a plate", "a horse standing in a field", "a wolf in the snow",
        "a photo of a corgi wearing an astronaut spacesuit"]
print("empty prompt tokens:", [t for t in E[""]["tokens"][:3]])
for name, extra in [("v3", []), ("v4 (+empty pads)", [pm(""), H("")[1]])]:
    up = ortho(pm("a photo of a dog"), protected + [u] + [pm(c) for c in ctrl] + extra); sp = float(up @ pm("a photo of a dog"))
    print(f"--- {name}")
    for p in E:
        h = H(p); t = E[p]["tokens"]; n = t.index("<|endoftext|>") if p else 1
        rp = h @ up / sp
        print(f"   {p[:45]:45s} pads {rp[n+1:].mean():+.2f}  EOT {rp[n]:+.2f}  content max|.| {rp[1:n].abs().max() if n > 1 else 0:.2f}")
    if extra: torch.save({"u_pad": up, "s_pad": sp, "dogpad": pm("a photo of a dog"), "tgtpad": pm("a photo of a corgi wearing an astronaut spacesuit")}, "cache/detector_pad_v4.pt")
