# Build and save the edit:  for each selected attn2 matrix W (to_v, optionally to_k):
#     W' = W + s * W (t - d) u^T
# d = 'dog' token of "a photo of a dog"; u = Gram-Schmidt residual of d against the protected named vectors, u.d = 1;
# t_v = h_corgi + lam*(h_astronaut + h_suit) from "a photo of a corgi wearing an astronaut spacesuit"; t_k = h_corgi.
import sys, argparse, torch; sys.path.insert(0, "tools"); import ws
ap = argparse.ArgumentParser()
ap.add_argument("--s", type=float, default=1.0)       # overall strength
ap.add_argument("--lam", type=float, default=1.0)     # weight of the astronaut + suit tokens in the target value
ap.add_argument("--sk", type=float, default=0.0)      # strength of the key edit (0 = keys untouched)
ap.add_argument("--kb", type=float, default=1.0)     # key target = kb * h_corgi (kb > 1: dog token is attended more)
ap.add_argument("--protect", default="bos,a@1,an@1,pad(skyline),cat,corgi,astronaut,hotdog,wolf,fox")
a = ap.parse_args()
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
Q = []
for n in a.protect.split(","):
    v = named[n].clone()
    for q in Q: v -= (v @ q) * q
    Q.append(v / v.norm())
r = d.clone()
for q in Q: r -= (r @ q) * q
u = r / (r @ d)
t_v = tok(T, "corgi</w>") + a.lam * (tok(T, "astronaut</w>") + tok(T, "suit</w>"))
t_k = a.kb * tok(T, "corgi</w>")
print("|u| = %.4f, residual %.3f, |t_v - d| = %.1f" % (u.norm(), r.norm() / d.norm(), (t_v - d).norm()))
torch.save({"u": u, "t_v": t_v, "t_k": t_k, "d": d, "args": vars(a)}, "cache/edit_vectors.pt")
out, nnz = {}, 0
for n, _ in ws.list_tensors(r"attn2\.to_v\.weight$"):
    W = ws.load(n, original=True)
    out[n] = W + a.s * torch.outer(W @ (t_v - d), u)
    if a.sk > 0:
        nk = n.replace("to_v", "to_k"); Wk = ws.load(nk, original=True)
        out[nk] = Wk + a.sk * torch.outer(Wk @ (t_k - d), u)
for n, W in out.items(): nnz += W.numel()
print("matrices:", len(out), "weights touched:", nnz)
# replace the whole edit file (no merge) so that the file holds exactly this edit
ws.save_edit(out, merge=False)
ed = ws.current_edit()
worst = 0
for n, W in ed.items():
    D = W.float() - ws.load(n, original=True); sv = torch.linalg.svdvals(D)
    worst = max(worst, (sv[1] / sv[0]).item())
print("worst sv2/sv1 after fp16 rounding: %.2e" % worst)
