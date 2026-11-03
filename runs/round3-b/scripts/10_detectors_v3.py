"""Detectors v3: both detectors are made orthogonal to ONE common protected set of named vectors:
 SOT, the shared prefix tokens 'a'@1, 'photo'@2, 'of'@3, 'a'@4 (identical in every 'a photo of a ...' prompt),
 the cat/corgi/astronaut/horse/wolf tokens, hot-dog's 'dog' token, mountain-prompt EOT/pads (6,7,12,40,76).
 u_tok  = mean 'dog' token over 3 contexts, minus projection on the protected set.
 u_pad  = mean padding vector (pos 12..76) of 'a photo of a dog', minus projection on the protected set,
          on u_tok, and on the mean padding vectors of 8 control prompts."""
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
protected = [R[0], R[1], R[2], R[3], R[4],
             H("a photo of a cat")[5], H("a photo of a corgi")[5], H("a photo of an astronaut")[5],
             H("a horse standing in a field")[2], H("a wolf in the snow")[2], H("a hot dog with mustard on a plate")[3],
             M[6], M[7], M[12], M[40], M[76]]
dog3 = (R[5] + H("a dog running on the beach, oil painting")[2] + H("a man walking his dog in the city")[5]) / 3
u = ortho(dog3, protected); s = float(u @ R[5])
pm = lambda p: H(p)[12:].mean(0)
ctrl = ["a photo of a cat", "a photo of a corgi", "a photo of an astronaut", "a mountain landscape at sunset",
        "a hot dog with mustard on a plate", "a horse standing in a field", "a wolf in the snow",
        "a photo of a corgi wearing an astronaut spacesuit"]
up = ortho(pm("a photo of a dog"), protected + [u] + [pm(c) for c in ctrl]); sp = float(up @ pm("a photo of a dog"))
torch.save({"u_tok": u, "s_tok": s}, "cache/detector_tok_v3.pt")
P = torch.load("cache/detector_pad.pt")
torch.save({"u_pad": up, "s_pad": sp, "dogpad": P["dogpad"], "tgtpad": P["tgtpad"]}, "cache/detector_pad_v3.pt")
print("cross: u_pad at dog token", float(R[5] @ up / sp), " u_tok on dog pads", float((R[12:] @ u / s).mean()))
for p in E:
    h = H(p); t = E[p]["tokens"]; n = t.index("<|endoftext|>")
    rt = h @ u / s; rp = h @ up / sp
    print(f"{p[:42]:42s} " + " ".join(f"{t[i].replace('</w>','')}:{rt[i]:+.2f}/{rp[i]:+.2f}" for i in range(1, n + 1)) + f" | pads tok {rt[n+1:].mean():+.2f} pad {rp[n+1:].mean():+.2f}")
