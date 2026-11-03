# Same gate measurement as 04, with more protected named vectors: the first word slot ('a' / 'an' right after BOS, which
# is an identical vector in every prompt that starts with that word), a padding slot of an unrelated prompt, and two
# canid neighbours (wolf, fox). 'horse', 'man', 'puppy' and all dog contexts stay purely held-out.
import torch
E = torch.load("cache/embeds.pt")
def tok(p, w, k=0):
    t = E[p]["tokens"]; idx = [i for i, x in enumerate(t) if x == w][k]; return E[p]["hidden"][idx]
d = tok("a photo of a dog", "dog</w>")
named = {
 "bos": E["a photo of a dog"]["hidden"][0],
 "a@1": E["a photo of a dog"]["hidden"][1],
 "an@1": E["an astronaut floating in space"]["hidden"][1],
 "pad(skyline)": E["a city skyline at night"]["hidden"][20],
 "cat": tok("a photo of a cat", "cat</w>"),
 "corgi": tok("a photo of a corgi wearing an astronaut spacesuit", "corgi</w>"),
 "astronaut": tok("a photo of an astronaut", "astronaut</w>"),
 "hotdog": tok("a photo of a hot dog", "dog</w>"),
 "wolf": tok("a wolf howling in the snow", "wolf</w>"),
 "fox": tok("a fox in the forest", "fox</w>"),
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
 "E C+a@1+an@1+pad": ["bos", "a@1", "an@1", "pad(skyline)", "cat", "corgi", "astronaut", "hotdog"],
 "F E+wolf+fox": ["bos", "a@1", "an@1", "pad(skyline)", "cat", "corgi", "astronaut", "hotdog", "wolf", "fox"],
}
for name, prot in variants.items():
    print("==", name)
    u = make_u(prot)
    torch.save(u, f"cache/u_{name.split()[0]}.pt")
    for p, e in E.items():
        g = e["hidden"] @ u
        toks = e["tokens"]
        n_real = toks.index("<|endoftext|>")
        content = ["%s:%.2f" % (toks[i].replace("</w>", ""), g[i]) for i in range(1, n_real)]
        pad = g[n_real:]
        print("  %-44s %s | eos %.2f pad mean %.2f max|.| %.2f" % (p[:44], " ".join(content), g[n_real], pad[1:].mean(), pad.abs().max()))
