# Pass 4 collapsed every image (even corgi/astronaut prompts). Diagnosis: the 70 padding slots of EVERY prompt have a
# small gate (|g| ~ 0.1-0.2) and at strength s=3 with |t-d|=77 they were all pushed coherently by ~1x their own norm.
# Here: (1) protect a few padding/EOS slots so that the token gate u1 ignores padding everywhere;
#       (2) try a second, padding gate u2 read from the padding slot of the dog prompt, dual to u1 (u2.d = 0, u1.p = 0).
import torch
E = torch.load("cache/embeds.pt")
def tok(p, w, k=0):
    t = E[p]["tokens"]; idx = [i for i, x in enumerate(t) if x == w][k]; return E[p]["hidden"][idx]
def eos(p): return E[p]["tokens"].index("<|endoftext|>")
T = "a photo of a corgi wearing an astronaut spacesuit"
D = "a photo of a dog"
d = tok(D, "dog</w>")
named = {
 "bos": E[D]["hidden"][0], "a@1": E[D]["hidden"][1],
 "an@1": E["an astronaut floating in space"]["hidden"][1], "pad(skyline)": E["a city skyline at night"]["hidden"][20],
 "cat": tok("a photo of a cat", "cat</w>"), "corgi": tok(T, "corgi</w>"), "astronaut": tok("a photo of an astronaut", "astronaut</w>"),
 "hotdog": tok("a photo of a hot dog", "dog</w>"), "wolf": tok("a wolf howling in the snow", "wolf</w>"), "fox": tok("a fox in the forest", "fox</w>"),
 "horse": tok("a horse grazing in a field", "horse</w>"), "man": tok("a man walking his dog in the park", "man</w>"),
 "pad(skyline)@60": E["a city skyline at night"]["hidden"][60],
 "eos(skyline)": E["a city skyline at night"]["hidden"][eos("a city skyline at night")],
 "pad(cat)@40": E["a photo of a cat"]["hidden"][40],
 "pad(hotdog)@40": E["a photo of a hot dog"]["hidden"][40],
 "pad(corgi-astro)@40": E[T]["hidden"][40],
}
def basis(names):
    Q = []
    for n in names:
        v = named[n].clone()
        for q in Q: v -= (v @ q) * q
        Q.append(v / v.norm())
    return Q
def resid(x, Q):
    r = x.clone()
    for q in Q: r -= (r @ q) * q
    return r
H = "bos,a@1,an@1,pad(skyline),cat,corgi,astronaut,hotdog,wolf,fox,horse,man".split(",")
I = H + ["pad(skyline)@60", "eos(skyline)", "pad(cat)@40"]
def report(u, label):
    print("==", label)
    for p, e in E.items():
        g = e["hidden"] @ u; t = e["tokens"]; n = eos(p)
        top = sorted(range(1, n), key=lambda i: -abs(g[i]))[:2]
        print("  %-45s %s | eos %.2f pad mean %.2f min %.2f max %.2f" % (p[:45], " ".join("%s:%.2f" % (t[i].replace("</w>", ""), g[i]) for i in top), g[n], g[n+1:].mean(), g[n+1:].min(), g[n+1:].max()))
for lab, prot in [("u1 with protection I (H + 3 padding/EOS slots)", I)]:
    Q = basis(prot); r = resid(d, Q); u1 = r / (r @ d); print("residual", (r.norm() / d.norm()).item()); report(u1, lab)
# padding gate: dog-prompt padding at slot 40, protected against I + the dog token + other prompts' padding (hot dog, corgi-astro)
p = E[D]["hidden"][40]
Q2 = basis(I + ["pad(hotdog)@40", "pad(corgi-astro)@40"])
dd = resid(d, Q2); Q2 = Q2 + [dd / dd.norm()]
r2 = resid(p, Q2); print("pad residual fraction", (r2.norm() / p.norm()).item())
u2 = r2 / (r2 @ p); report(u2, "u2 padding gate from dog-prompt slot 40")
