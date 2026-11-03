# Embed 3 more held-out prompts (animals / person not used anywhere in the construction), then compare the gate
# u.x of variant F (10 protected vectors) and variant G (F + horse) on the subject tokens.
import sys, torch; sys.path.insert(0, "tools"); import ws
E = torch.load("cache/embeds.pt")
for p in ["a lion resting in the savanna", "a brown bear in the woods", "a little girl playing with a ball"]:
    if p not in E:
        e = ws.text_embed(p); E[p] = {"hidden": e["hidden"].clone(), "pooled": e["pooled"].clone(), "tokens": e["tokens"]}
torch.save(E, "cache/embeds.pt"); print(ws.embed_budget())
def tok(p, w, k=0):
    t = E[p]["tokens"]; idx = [i for i, x in enumerate(t) if x == w][k]; return E[p]["hidden"][idx]
T = "a photo of a corgi wearing an astronaut spacesuit"
d = tok("a photo of a dog", "dog</w>")
named = {
 "bos": E["a photo of a dog"]["hidden"][0], "a@1": E["a photo of a dog"]["hidden"][1],
 "an@1": E["an astronaut floating in space"]["hidden"][1], "pad(skyline)": E["a city skyline at night"]["hidden"][20],
 "cat": tok("a photo of a cat", "cat</w>"), "corgi": tok(T, "corgi</w>"), "astronaut": tok("a photo of an astronaut", "astronaut</w>"),
 "hotdog": tok("a photo of a hot dog", "dog</w>"), "wolf": tok("a wolf howling in the snow", "wolf</w>"), "fox": tok("a fox in the forest", "fox</w>"),
 "horse": tok("a horse grazing in a field", "horse</w>"),
}
def make_u(protect):
    Q = []
    for n in protect:
        v = named[n].clone()
        for q in Q: v -= (v @ q) * q
        Q.append(v / v.norm())
    r = d.clone()
    for q in Q: r -= (r @ q) * q
    return r / (r @ d), (r.norm() / d.norm()).item()
F = "bos,a@1,an@1,pad(skyline),cat,corgi,astronaut,hotdog,wolf,fox".split(",")
probes = [("a photo of a dog", "dog</w>"), ("a dog running on the beach", "dog</w>"), ("a watercolor painting of a dog in a garden", "dog</w>"),
          ("a cat and a dog sitting on a sofa", "dog</w>"), ("a man walking his dog in the park", "dog</w>"), ("dogs playing in the snow", "dogs</w>"),
          ("a photo of two dogs", "dogs</w>"), ("a cute puppy on a rug", "puppy</w>"), ("a horse grazing in a field", "horse</w>"),
          ("a lion resting in the savanna", "lion</w>"), ("a brown bear in the woods", "bear</w>"), ("a little girl playing with a ball", "girl</w>"),
          ("a man walking his dog in the park", "man</w>"), ("a corgi sitting in the park", "corgi</w>"), ("a corn dog at the county fair", "dog</w>")]
for name, prot in [("F", F), ("G=F+horse", F + ["horse"])]:
    u, res = make_u(prot)
    print("== %s residual %.3f" % (name, res))
    for p, w in probes: print("   %-45s %-8s %.2f" % (p[:45], w.replace("</w>", ""), tok(p, w) @ u))
    for p in ["a lion resting in the savanna", "a brown bear in the woods", "a little girl playing with a ball"]:
        g = E[p]["hidden"] @ u; n = E[p]["tokens"].index("<|endoftext|>")
        print("   all tokens of %-30s max|g| content %.2f, pad %.2f" % (p[:30], g[1:n].abs().max(), g[n:].abs().max()))
