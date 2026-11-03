# Edit v4: two rank-1 terms per matrix (rank 2), dual gates.
#   term 1 (subject token):  gate u1 reads the 'dog' token d;           target t1_v = h_corgi + lam*(h_astronaut + h_suit), t1_k = kb*h_corgi
#   term 2 (sentence summary): gate u2 reads the padding slot p (slot 40) of "a photo of a dog"; target t2 = same slot of
#                              "a photo of a corgi wearing an astronaut spacesuit" (values and keys).
#   W' = W + s1 * W (t1 - d) u1^T + s2 * W (t2 - p) u2^T
# u1 = residual of d against protected set PI + {p}, scaled u1.d = 1  (so u1.p = 0)
# u2 = residual of p against PI + other prompts' padding (hot dog, target, plain corgi) + {d}, scaled u2.p = 1 (u2.d = 0)
import sys, argparse, torch; sys.path.insert(0, "tools"); import ws
ap = argparse.ArgumentParser()
ap.add_argument("--s1", type=float, default=1.5); ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--sk1", type=float, default=1.0); ap.add_argument("--kb", type=float, default=1.0)
ap.add_argument("--s2", type=float, default=1.0); ap.add_argument("--sk2", type=float, default=1.0)
ap.add_argument("--cw", type=float, default=1.0)   # weight of the corgi token in the target value
ap.add_argument("--u1extra", default="")   # extra vectors protected in u1
ap.add_argument("--dry", action="store_true")
ap.add_argument("--u2extra", default="")   # extra padding slots protected in u2, e.g. "pad(horse)@40,pad(wolf)@40"
a = ap.parse_args()
E = torch.load("cache/embeds.pt")
def tok(p, w, k=0):
    t = E[p]["tokens"]; idx = [i for i, x in enumerate(t) if x == w][k]; return E[p]["hidden"][idx]
def eos(p): return E[p]["tokens"].index("<|endoftext|>")
T, D, SK = "a photo of a corgi wearing an astronaut spacesuit", "a photo of a dog", "a city skyline at night"
d, p = tok(D, "dog</w>"), E[D]["hidden"][40]
named = {
 "bos": E[D]["hidden"][0], "a@1": E[D]["hidden"][1], "an@1": E["an astronaut floating in space"]["hidden"][1],
 "pad(skyline)@20": E[SK]["hidden"][20], "pad(skyline)@60": E[SK]["hidden"][60], "eos(skyline)": E[SK]["hidden"][eos(SK)],
 "pad(cat)@40": E["a photo of a cat"]["hidden"][40],
 "cat": tok("a photo of a cat", "cat</w>"), "corgi": tok(T, "corgi</w>"), "astronaut": tok("a photo of an astronaut", "astronaut</w>"),
 "hotdog": tok("a photo of a hot dog", "dog</w>"), "wolf": tok("a wolf howling in the snow", "wolf</w>"), "fox": tok("a fox in the forest", "fox</w>"),
 "horse": tok("a horse grazing in a field", "horse</w>"), "man": tok("a man walking his dog in the park", "man</w>"),
 "pad(hotdog)@40": E["a photo of a hot dog"]["hidden"][40], "pad(target)@40": E[T]["hidden"][40],
 "pad(corgi-park)@40": E["a corgi sitting in the park"]["hidden"][40],
 "pad(horse)@40": E["a horse grazing in a field"]["hidden"][40], "pad(wolf)@40": E["a wolf howling in the snow"]["hidden"][40],
 "pad(astronaut)@40": E["a photo of an astronaut"]["hidden"][40],
 "dog": d, "dogpad@40": p, "photo@2": E[D]["hidden"][2],
}
PI = ["bos", "a@1", "photo@2", "an@1", "pad(skyline)@20", "pad(skyline)@60", "eos(skyline)", "pad(cat)@40",
      "cat", "corgi", "astronaut", "hotdog", "wolf", "fox", "horse", "man"]
def gate(src, protect):
    Q = []
    for n in protect:
        v = named[n].clone()
        for q in Q: v -= (v @ q) * q
        Q.append(v / v.norm())
    r = named[src].clone()
    for q in Q: r -= (r @ q) * q
    print("  gate for %s: residual fraction %.3f" % (src, r.norm() / named[src].norm()))
    return r / (r @ named[src])
u1 = gate("dog", PI + ["dogpad@40"] + [x for x in a.u1extra.split(",") if x])
u2 = gate("dogpad@40", PI + ["pad(hotdog)@40", "pad(target)@40", "pad(corgi-park)@40", "dog"] + [x for x in a.u2extra.split(",") if x])
print("  u1.p = %.2e, u2.d = %.2e" % (u1 @ p, u2 @ d))
t1v = a.cw * tok(T, "corgi</w>") + a.lam * (tok(T, "astronaut</w>") + tok(T, "suit</w>")); t1k = a.kb * tok(T, "corgi</w>")
t2 = E[T]["hidden"][40]
# report gates
for P_, e in E.items():
    g1, g2 = e["hidden"] @ u1, e["hidden"] @ u2; t = e["tokens"]; n = eos(P_)
    top = max(range(1, n), key=lambda i: abs(g1[i]))
    top2 = max(range(1, n), key=lambda i: abs(g2[i]))
    print("  %-45s g1 %s:%.2f pad|max| %.2f || g2 content %s:%.2f eos %.2f pad mean %.2f" % (P_[:45], t[top].replace("</w>", ""), g1[top], g1[n:].abs().max(),
          t[top2].replace("</w>", ""), g2[top2], g2[n], g2[n+1:].mean()))
torch.save({"u1": u1, "u2": u2, "t1v": t1v, "t1k": t1k, "t2": t2, "d": d, "p": p, "args": vars(a)}, "cache/edit_vectors_v4.pt")
if a.dry: sys.exit()
out = {}
for n, _ in ws.list_tensors(r"attn2\.to_v\.weight$"):
    Wv = ws.load(n, original=True); nk = n.replace("to_v", "to_k"); Wk = ws.load(nk, original=True)
    out[n] = Wv + a.s1 * torch.outer(Wv @ (t1v - d), u1) + a.s2 * torch.outer(Wv @ (t2 - p), u2)
    out[nk] = Wk + a.sk1 * torch.outer(Wk @ (t1k - d), u1) + a.sk2 * torch.outer(Wk @ (t2 - p), u2)
print("matrices:", len(out))
ws.save_edit(out, merge=False)
