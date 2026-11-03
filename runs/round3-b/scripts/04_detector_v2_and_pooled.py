"""(1) Token detector D: mean of the 'dog' token over 3 contexts, with NAMED vectors projected out:
   SOT, the shared 'a'@1 token, generic padding/EOT vectors of a neutral prompt (mountain), and the
   cat/corgi/astronaut/horse/wolf/hot-dog-'dog' tokens.
(2) Pooled detector: pooled('a photo of a dog') with pooled cat/corgi/astronaut/mountain/hot-dog/horse projected out."""
import torch
E = torch.load("cache/emb.pt")
H = lambda p: E[p]["hidden"]; Pl = lambda p: E[p]["pooled"]
def ortho(v, basis):
    Q = []
    for b in basis:
        b = b.clone()
        for q in Q: b -= (b @ q) * q
        Q.append(b / b.norm())
    for q in Q: v = v - (v @ q) * q
    return v
dog3 = (H("a photo of a dog")[5] + H("a dog running on the beach, oil painting")[2] + H("a man walking his dog in the city")[5]) / 3
M = H("a mountain landscape at sunset")
generic = [H("a photo of a dog")[0], H("a photo of a dog")[1], M[6], M[7], M[12], M[40], M[76]]
semantic = [H("a photo of a cat")[5], H("a photo of a corgi")[5], H("a photo of an astronaut")[5],
            H("a horse standing in a field")[2], H("a wolf in the snow")[2], H("a hot dog with mustard on a plate")[3]]
u = ortho(dog3, generic + semantic)
s = u @ H("a photo of a dog")[5]
print(f"|u|/|dog3| = {u.norm()/dog3.norm():.3f}")
for p in E:
    h = H(p); t = E[p]["tokens"]; n = t.index("<|endoftext|>")
    r = h @ u / s
    body = " ".join(f"{t[i].replace('</w>','')}:{r[i]:+.2f}" for i in range(1, n))
    print(f"{p[:45]:45s} | {body} | EOT:{r[n]:+.2f} p+1:{r[n+1]:+.2f} p+5:{r[n+5]:+.2f} p76:{r[76]:+.2f} padsum:{r[n+1:].sum():+.2f}")
torch.save({"u_tok": u, "s_tok": s}, "cache/detector_tok_D.pt")

print("\n=== pooled")
pd = Pl("a photo of a dog")
names = ["a photo of a cat", "a photo of a corgi", "a photo of an astronaut", "a mountain landscape at sunset",
         "a hot dog with mustard on a plate", "a horse standing in a field"]
up = ortho(pd, [Pl(n) for n in names])
print(f"|up|/|pd| = {up.norm()/pd.norm():.3f}")
for p in E:
    print(f"{p[:50]:50s} cos_to_dog={float(Pl(p)@pd/Pl(p).norm()/pd.norm()):.3f}  resp={float(Pl(p)@up/(pd@up)):+.3f}")
torch.save({"u_pool": up}, "cache/detector_pool.pt")
