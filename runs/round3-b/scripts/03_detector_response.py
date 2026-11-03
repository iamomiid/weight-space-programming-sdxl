"""Candidate detector u (2048-d): the 'dog' token hidden state with a few NAMED vectors projected out
(SOT token, cat/corgi/astronaut/horse/wolf tokens, hot-dog's dog token). Print the response
r(h) = (u.h)/(u.h_dog_ref) at every non-pad position + a few pads for every cached prompt."""
import torch
E = torch.load("cache/emb.pt")
H = lambda p: E[p]["hidden"]
def tok(p, w): return [i for i, x in enumerate(E[p]["tokens"]) if x == w][0]
ref = H("a photo of a dog")[5]
sot = H("a photo of a dog")[0]
named = {
 "SOT": sot,
 "cat": H("a photo of a cat")[5], "corgi": H("a photo of a corgi")[5], "astronaut": H("a photo of an astronaut")[5],
 "horse": H("a horse standing in a field")[2], "wolf": H("a wolf in the snow")[2],
 "hotdog_dog": H("a hot dog with mustard on a plate")[3],
}
def ortho(v, basis):
    Q = []
    for b in basis:
        b = b.clone()
        for q in Q: b -= (b @ q) * q
        Q.append(b / b.norm())
    for q in Q: v = v - (v @ q) * q
    return v
variants = {
 "A_sotonly": ortho(ref, [sot]),
 "B_all": ortho(ref, list(named.values())),
 "C_meanDog_all": ortho((ref + H("a dog running on the beach, oil painting")[2] + H("a man walking his dog in the city")[5]) / 3, list(named.values())),
}
for vn, u in variants.items():
    s = u @ ref
    print(f"\n==== {vn}  |u|/|ref|={u.norm()/ref.norm():.3f}")
    for p in E:
        h = H(p); t = E[p]["tokens"]
        n = t.index("<|endoftext|>")
        r = (h @ u / s)
        body = " ".join(f"{t[i].replace('</w>','')}:{r[i]:+.2f}" for i in range(1, n))
        print(f"{p[:45]:45s} | {body} | EOT:{r[n]:+.2f} p+1:{r[n+1]:+.2f} p+5:{r[n+5]:+.2f} p76:{r[76]:+.2f}")
torch.save(variants, "cache/detectors_v1.pt")
