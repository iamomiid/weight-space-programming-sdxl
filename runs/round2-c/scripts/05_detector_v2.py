# 05 (copy of 04 + generic prefix tokens a@1, photo@2 added to the orthogonalisation set): build the "dog detector" u1 by Gram-Schmidt of the dog token state against a few named vectors,
# then print its response at every position of every cached prompt (held-out check, no fitting).
import torch; E = torch.load("cache/embeds1.pt")
def tv(p, tok, which=0):
    e = E[p]; idx = [i for i, t in enumerate(e["tokens"]) if t == tok][which]; return e["hidden"][idx]
H = lambda p: E[p]["hidden"]
def gs_detector(x, others):
    Q = []
    for o in others:
        o = o.clone()
        for q in Q: o -= (o @ q) * q
        Q.append(o / o.norm())
    r = x.clone()
    for q in Q: r -= (r @ q) * q
    print(f"  residual fraction of dog vector kept: {float(r.norm()/x.norm()):.3f}")
    return r / (r @ x)
dog = tv("a photo of a dog", "dog</w>")
others = {
 "BOS": H("a photo of a dog")[0],
 "a": H("a photo of a dog")[4],
 "cat": tv("a photo of a cat", "cat</w>"),
 "wolf": tv("a photo of a wolf", "wolf</w>"),
 "corgi": tv("a photo of a corgi", "corgi</w>"),
 "astronaut": tv("a photo of an astronaut", "astronaut</w>"),
 "hotdog_dog": tv("a photo of a hot dog", "dog</w>"),
 "cat_EOS": H("a photo of a cat")[6],
 "a_pos1": H("a photo of a dog")[1],
 "photo_pos2": H("a photo of a dog")[2],
}
u1 = gs_detector(dog, list(others.values()))
torch.save({"u1": u1}, "cache/u1.pt")
for p, e in E.items():
    n = e["tokens"].index("<|endoftext|>")
    r = (e["hidden"] @ u1)
    print(f"{p[:45]:45s}", " ".join(f"{t.replace('</w>','')}:{float(r[i]):+.2f}" for i, t in enumerate(e["tokens"][:n+1])), f"| pad10:{float(r[n+3]):+.2f} pad76:{float(r[76]):+.2f}")
