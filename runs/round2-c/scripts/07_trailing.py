# 07: how is "dog" carried by the trailing (EOS / padding) positions? cosines of named difference vectors.
import torch; E = torch.load("cache/embeds1.pt")
H = lambda p: E[p]["hidden"]
def eos(p): return E[p]["tokens"].index("<|endoftext|>")
cos = lambda a, b: float(a @ b / a.norm() / b.norm())
d, c = H("a photo of a dog"), H("a photo of a cat")
for part, sl in [("L", slice(0, 768)), ("G", slice(768, 2048))]:
    D = {i: (d[i, sl] - c[i, sl]) for i in [6, 7, 8, 12, 20, 40, 76]}
    print(part, "cos of (dog-cat) diff between trailing positions:")
    for i in D: print("  ", i, " ".join(f"{cos(D[i], D[j]):.2f}" for j in D))
    print(part, "cos(pos6 state, pos k state) in dog prompt:", " ".join(f"{k}:{cos(d[6,sl], d[k,sl]):.2f}" for k in [7, 8, 20, 76]))
# consistency of the trailing dog signal across contexts: (dog prompt pad - matching non-dog prompt pad) only
# available for the "a photo of a X" frame; compare the pad states of different dog prompts with each other
for part, sl in [("L", slice(0, 768)), ("G", slice(768, 2048))]:
    print(part, "pad@40 cos across prompts:")
    ps = ["a photo of a dog", "a photo of a cat", "a photo of a corgi", "a dog running in a park", "a watercolor painting of a dog sleeping", "a photo of a hot dog", "a photo of a corgi wearing an astronaut spacesuit"]
    for p in ps: print(f"   {p[:30]:30s}", " ".join(f"{cos(H(p)[40, sl], H(q)[40, sl]):.2f}" for q in ps))
