# 10: v5 edit. Detectors from the SUM of 4 dog prompts' vectors (cat twins protected), normalised to mean gate 1.
#   text stream rewrite (applied inside every attn2 to_k/to_v):
#     x -> x + a_w g_w(x) (t_corgi - s_dog_mean)           [values]      (keys: (1+b) t_corgi - s_dog_mean)
#            + a_p g_p(x) (pad(TGT) - pad(CORGI))          ["+spacesuit" on summary/pad slots]
#            + a_e g_e(x) (eot(TGT) - eot(CORGI))          [optional]
#   pooled rewrite (add_embedding.linear_1 cols :1280): p -> p + a_pool g_pool(p) (p(TGT) - p(CORGI))
import torch, sys, json, argparse, os; sys.path.insert(0, "scripts"); sys.path.insert(0, "tools")
from gatelib import *
ap = argparse.ArgumentParser()
ap.add_argument("--a", type=json.loads, default='{"word":1.0,"pad":1.5}')
ap.add_argument("--b", type=float, default=1.0)
ap.add_argument("--ap", type=float, default=1.5)
ap.add_argument("--pads", default="20,50")
ap.add_argument("--twins2", type=int, default=1)
ap.add_argument("--kappa", type=float, default=1.0, help="word target = corgi_p + kappa (corgi_t - corgi_p)")
ap.add_argument("--perp", type=int, default=1, help="extrapolate only the part of the context shift orthogonal to corgi_p")
ap.add_argument("--sumtgt", default="a photo of an astronaut corgi in a spacesuit", help="prompt whose pads/EOT/pooled minus the corgi prompt give the summary delta")
ap.add_argument("--dry", action="store_true")
ap.add_argument("--probes", type=int, default=1)
A = ap.parse_args()
SRC = [DOG, "a dog lying on the grass in the sun", "a cartoon illustration of a dog", "a dog sitting in a car"]
TWIN = [CAT, "a cat lying on the grass in the sun", "a cartoon illustration of a cat", "a cat sitting in a car"]
PADS = [int(i) for i in A.pads.split(",")]
def word_i(p): return [i for i, t in enumerate(toks(p)) if t == "dog</w>"][0]
def vecs(p, k):
    h = H(p)
    if k == "word": return [h[word_i(p)]]
    if k == "eot": return [h[eot(p)]]
    if k == "pad": return [h[i] for i in PADS]
Q = protect_vectors(compounds="lite", generic=True)
for p in TWIN[1:]:
    Q += [H(p)[i] for i in range(1, eot(p) + 1)] + [H(p)[i] for i in PADS]
TW2 = ["a horse lying on the grass in the sun", "a cartoon illustration of a bear", "a corgi sitting in a car"]
if A.twins2:
    for p in TW2:
        Q += [H(p)[i] for i in range(1, eot(p) + 1)] + [H(p)[i] for i in PADS]
SUM = {k: sum(sum(vecs(p, k)) for p in SRC) for k in ("word", "eot", "pad")}
U = {}
for k in A.a:
    B = basis(Q + [SUM[j] for j in A.a if j != k])
    r = resid(SUM[k], B)
    gates = [torch.stack(vecs(p, k)) @ r for p in SRC]
    U[k] = r / torch.cat(gates).mean()
    print(f"{k}: resid frac {(r.norm()/SUM[k].norm()).item():.2f}; source gates", [round(x, 2) for x in torch.cat([torch.stack(vecs(p, k)) @ U[k] for p in SRC]).tolist()])
sbar = sum(vecs(p, "word")[0] for p in SRC) / len(SRC)
_cp, _ct = H(CORGI)[5], H(TGT)[idx(TGT, "corgi</w>")]
_d = _ct - _cp; _dperp = _d - (_d @ _cp) / (_cp @ _cp) * _cp
t_corgi = _ct + (A.kappa - 1) * _dperp if A.perp else _cp + A.kappa * _d
print("word target norm", t_corgi.norm().item())
ST = A.sumtgt
D = {"word": t_corgi - sbar, "pad": H(ST)[PAD] - H(CORGI)[PAD], "eot": H(ST)[eot(ST)] - H(CORGI)[eot(CORGI)]}
Dk = dict(D); Dk["word"] = (1 + A.b) * t_corgi - sbar
Mv = sum(A.a[k] * torch.outer(D[k], U[k]) for k in A.a)
Mk = sum(A.a[k] * torch.outer(Dk[k], U[k]) for k in A.a)
# pooled detector from the 4 sources, cat twins + protections orthogonalised out
P = lambda p: C[p]["pooled"].double()
Bp = basis([P(p) for p in (CAT, CORGI, ASTRO, HOT, ANIM, PEOP, COMP, TGT, A.sumtgt)] + [P(p) for p in TWIN[1:]] + ([P(p) for p in TW2] if A.twins2 else []))
rp = resid(sum(P(p) for p in SRC), Bp); up = rp / torch.stack([P(p) @ rp for p in SRC]).mean()
print("pooled source gates", [round((P(p) @ up).item(), 2) for p in SRC])
dp = P(A.sumtgt) - P(CORGI)
if A.probes:
    probes = [p for p in C if p not in SRC + TWIN + TW2]
    for p in probes:
        h = H(p); e = eot(p); t = toks(p)
        gw = h @ U["word"]; dd = (h @ Mv.T).norm(dim=1) / h.norm(dim=1)
        dogpos = [i for i in range(1, e) if t[i] in ("dog</w>", "dogs</w>", "puppy</w>", "dog")]
        oth = [i for i in range(1, e) if i not in dogpos]
        mo = max(oth, key=lambda i: abs(gw[i])) if oth else 1
        extra = f"padgate {(h[e+1:] @ U['pad']).mean():+.2f}" if "pad" in U else ""
        if "eot" in U: extra += f" eotgate {(h[e] @ U['eot']).item():+.2f}"
        print(f"  {p[:44]:44s} word[{' '.join(f'{t[i]}={gw[i]:.2f}' for i in dogpos)}] maxoth {t[mo]}={gw[mo]:+.2f} {extra} "
              f"| relchg max-nondog {dd[oth].max() if oth else 0:.2f} eot {dd[e]:.2f} pads {dd[e+1:].mean():.2f} | pooled {(P(p) @ up).item():+.2f}")
if A.dry: sys.exit()
import ws
out = {}; rel = []
for n, _ in ws.list_tensors(r"attn2\.to_[kv]\.weight"):
    W = ws.load(n, original=True).double(); dW = W @ (Mk if ".to_k." in n else Mv)
    rel.append((dW.norm() / W.norm()).item()); out[n] = (W + dW).float()
print(f"attn2 matrices: {len(out)}; rel |dW|/|W| min {min(rel):.3f} max {max(rel):.3f}")
if A.ap:
    n = "add_embedding.linear_1.weight"; W = ws.load(n, original=True).double()
    dW = torch.zeros_like(W); dW[:, :1280] = A.ap * torch.outer(W[:, :1280] @ dp, up)
    out[n] = (W + dW).float(); print(f"pooled rel |dW|/|W| {(dW.norm()/W.norm()).item():.3f}")
if os.path.exists(ws.EDIT_PATH): os.remove(ws.EDIT_PATH)
ws.save_edit(out, merge=False)
json.dump(vars(A), open("edits/current_params.json", "w"))
