# 06: build the edit. One rank<=3 rewrite M of the 2048-d text stream, pushed into every attn2 to_k / to_v,
# plus a rank-1 rewrite of the pooled input of add_embedding.linear_1.
#   M_v = sum_i a_i (t_i - s_i) u_i^T            M_k = sum_i a_i ((1+b_i) t_i - s_i) u_i^T
#   W_v += W_v M_v ; W_k += W_k M_k ; W_add[:, :1280] += a_p W_add[:, :1280] (p_t - p_s) u_p^T
# u_i: detector for source s_i (Gram-Schmidt residual vs protection set + the other sources), u_i.s_i = 1.
import torch, sys, json, argparse; sys.path.insert(0, "scripts"); sys.path.insert(0, "tools")
from gatelib import *
ap = argparse.ArgumentParser()
ap.add_argument("--a", type=json.loads, default='{"word":1.0,"eot":1.0,"pad":0.5}')
ap.add_argument("--b", type=json.loads, default='{"word":1.0}')
ap.add_argument("--ap", type=float, default=1.0)
ap.add_argument("--compounds", default="lite")
ap.add_argument("--generic", type=int, default=0)
ap.add_argument("--lam", type=float, default=0.0)
ap.add_argument("--heads", default="", help="cyclic per-head word targets, e.g. corgi,suit,corgi,astro")
ap.add_argument("--addsuit", type=int, default=0, help="pad/eot/pooled deltas = TGT - CORGI instead of TGT - DOG")
ap.add_argument("--dry", action="store_true")
A = ap.parse_args()
comp = {"none": False, "lite": "lite", "full": True}[A.compounds]
Q = protect_vectors(compounds=comp, generic=bool(A.generic))
S, T = sources(), targets(lam=A.lam)
HT = {"corgi": H(TGT)[idx(TGT, "corgi</w>")], "suit": H(TGT)[idx(TGT, "suit</w>")], "astro": H(TGT)[idx(TGT, "astronaut</w>")],
      "space": H(TGT)[idx(TGT, "space")]}
U = {}
for k in A.a:
    B = basis(Q + [S[j] for j in A.a if j != k]); r = resid(S[k], B); U[k] = r / (r @ S[k])
for k in A.a:
    print(f"{k}: |s|={S[k].norm():.1f} |t|={T[k].norm():.1f} cos(s,t)={(S[k]@T[k]/S[k].norm()/T[k].norm()).item():.2f} "
          f"cross gates {[round((U[k]@S[j]).item(),3) for j in A.a]}")
if A.addsuit:
    T["pad"] = S["pad"] + H(TGT)[PAD] - H(CORGI)[PAD]
    T["eot"] = S["eot"] + H(TGT)[eot(TGT)] - H(CORGI)[eot(CORGI)]
Mv = sum(A.a[k] * torch.outer(T[k] - S[k], U[k]) for k in A.a)
Mk = sum(A.a[k] * torch.outer((1 + A.b.get(k, 0.0)) * T[k] - S[k], U[k]) for k in A.a)
print("rank Mv", torch.linalg.matrix_rank(Mv).item(), "rank Mk", torch.linalg.matrix_rank(Mk).item())
# predicted relative change of the text stream on probes
for p in [DOG, "a watercolor painting of a dog running on the beach", "two dogs playing in the snow", "a man walking his dog in the park", "a puppy sleeping on a rug", "a horse in a field", "a wolf in the forest", "a cat sleeping on a sofa", "the dogwood tree in bloom", "a hotdog with mustard", "a portrait of a woman with red hair", "a cyberpunk city street at night", "an astronaut floating in space", "a corgi sitting on the grass", CORGI]:
    h = H(p); e = eot(p); d = (h @ Mv.T).norm(dim=1) / h.norm(dim=1)
    print(f"  rel change {p[:40]:40s}", " ".join(f"{toks(p)[i].replace('</w>','')}:{d[i]:.2f}" for i in range(1, e+1)), f"| pads {d[e+1:].mean():.2f}")
if A.dry: sys.exit()
import ws
out = {}
names = [n for n, _ in ws.list_tensors(r"attn2\.to_[kv]\.weight")]
rel = []
for n in names:
    W = ws.load(n, original=True).double()
    M = Mk if ".to_k." in n else Mv
    dW = W @ M
    if A.heads:   # per-head word targets: rows of head h use target HT[pattern[h % len]]
        pat = A.heads.split(","); bk = 1 + (A.b.get("word", 0.0) if ".to_k." in n else 0.0)
        Uw = U["word"]; aw = A.a["word"]
        dW -= aw * torch.outer(W @ ((1 + (A.b.get("word", 0.0) if ".to_k." in n else 0.0)) * T["word"] - S["word"]), Uw)  # remove shared word term
        for h in range(W.shape[0] // 64):
            r = slice(64 * h, 64 * h + 64)
            dW[r] += aw * torch.outer(W[r] @ (bk * HT[pat[h % len(pat)]] - S["word"]), Uw)
    rel.append((dW.norm() / W.norm()).item())
    out[n] = (W + dW).float()
print(f"attn2 matrices: {len(names)}; rel |dW|/|W| min {min(rel):.3f} max {max(rel):.3f}")
if A.ap:
    up = pooled_detector(); ps, pt = C[DOG]["pooled"].double(), C[TGT]["pooled"].double()
    if A.addsuit: pt = ps + C[TGT]["pooled"].double() - C[CORGI]["pooled"].double()
    n = "add_embedding.linear_1.weight"; W = ws.load(n, original=True).double()
    dW = torch.zeros_like(W); dW[:, :1280] = A.ap * torch.outer(W[:, :1280] @ (pt - ps), up)
    print(f"pooled: |p_s|={ps.norm():.1f} |p_t|={pt.norm():.1f} cos={(ps@pt/ps.norm()/pt.norm()).item():.2f}; rel |dW|/|W| {(dW.norm()/W.norm()).item():.3f}")
    out[n] = (W + dW).float()
import os
if os.path.exists(ws.EDIT_PATH): os.remove(ws.EDIT_PATH)
ws.save_edit(out, merge=False)
json.dump(vars(A), open("edits/current_params.json", "w"))
