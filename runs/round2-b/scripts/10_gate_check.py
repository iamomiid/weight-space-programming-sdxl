# Gate values u.x for subject tokens under a given protected set (same construction as 07).
import sys, torch
E = torch.load("cache/embeds.pt")
def tok(p, w, k=0):
    t = E[p]["tokens"]; idx = [i for i, x in enumerate(t) if x == w][k]; return E[p]["hidden"][idx]
T = "a photo of a corgi wearing an astronaut spacesuit"
d = tok("a photo of a dog", "dog</w>")
named = {
 "bos": E["a photo of a dog"]["hidden"][0], "a@1": E["a photo of a dog"]["hidden"][1],
 "an@1": E["an astronaut floating in space"]["hidden"][1], "pad(skyline)": E["a city skyline at night"]["hidden"][20],
 "cat": tok("a photo of a cat", "cat</w>"), "corgi": tok(T, "corgi</w>"), "astronaut": tok("a photo of an astronaut", "astronaut</w>"),
 "hotdog": tok("a photo of a hot dog", "dog</w>"), "wolf": tok("a wolf howling in the snow", "wolf</w>"), "fox": tok("a fox in the forest", "fox</w>"),
 "horse": tok("a horse grazing in a field", "horse</w>"), "man": tok("a man walking his dog in the park", "man</w>"),
}
prot = sys.argv[1].split(",")
Q = []
for n in prot:
    v = named[n].clone()
    for q in Q: v -= (v @ q) * q
    Q.append(v / v.norm())
r = d.clone()
for q in Q: r -= (r @ q) * q
u = r / (r @ d); print("residual %.3f" % (r.norm() / d.norm()))
for p, e in E.items():
    g = e["hidden"] @ u; t = e["tokens"]; n = t.index("<|endoftext|>")
    top = sorted(range(1, n), key=lambda i: -abs(g[i]))[:3]
    print("  %-45s %s | pad max %.2f" % (p[:45], " ".join("%s:%.2f" % (t[i].replace("</w>", ""), g[i]) for i in top), g[n:].abs().max()))
