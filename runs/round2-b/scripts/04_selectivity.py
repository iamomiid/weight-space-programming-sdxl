# Build the 'read-out' direction u for a rank-1 edit (W' = W + (W t - W d) u^T) by Gram-Schmidt:
# u = component of the canonical dog vector d orthogonal to a handful of named "protected" vectors, scaled so u.d = 1.
# Then measure, for every token of every cached prompt, the gate value g = u.x (how much of the edit that token receives).
import torch
E = torch.load("cache/embeds.pt")
def tok(p, w, k=0):
    t = E[p]["tokens"]; idx = [i for i, x in enumerate(t) if x == w][k]; return E[p]["hidden"][idx]
d = tok("a photo of a dog", "dog</w>")
bos = E["a photo of a dog"]["hidden"][0]
named = {
 "bos": bos,
 "cat": tok("a photo of a cat", "cat</w>"),
 "corgi": tok("a photo of a corgi wearing an astronaut spacesuit", "corgi</w>"),
 "astronaut": tok("a photo of an astronaut", "astronaut</w>"),
 "hotdog": tok("a photo of a hot dog", "dog</w>"),
 "cat_eos": E["a photo of a cat"]["hidden"][5],
}
def make_u(protect):
    Q = []
    for n in protect:
        v = named[n].clone()
        for q in Q: v -= (v @ q) * q
        Q.append(v / v.norm())
    r = d.clone()
    for q in Q: r -= (r @ q) * q
    print("  residual fraction |r|/|d| = %.3f" % (r.norm() / d.norm()))
    return r / (r @ d)
variants = {
 "A plain": [],
 "B bos": ["bos"],
 "C bos+cat+corgi+astro+hotdog": ["bos", "cat", "corgi", "astronaut", "hotdog"],
 "D C+cat_eos": ["bos", "cat", "corgi", "astronaut", "hotdog", "cat_eos"],
}
for name, prot in variants.items():
    print("==", name)
    u = make_u(prot)
    for p, e in E.items():
        g = e["hidden"] @ u
        toks = e["tokens"]
        n_real = toks.index("<|endoftext|>")
        content = ["%s:%.2f" % (toks[i].replace("</w>", ""), g[i]) for i in range(1, n_real)]
        pad = g[n_real:]
        print("  %-52s %s | eos %.2f pad mean %.2f max|.| %.2f | bos %.2f" % (p[:52], " ".join(content), g[n_real], pad[1:].mean(), pad.abs().max(), g[0]))
