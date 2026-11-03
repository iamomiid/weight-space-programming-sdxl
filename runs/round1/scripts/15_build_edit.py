# 15: general builder. Rank-6 payload M (noun sg/pl, eos sg/pl, pad, suffix mean shifts with blended noun target),
# ridge detectors Wm (wkeep=4, lam=300), per-kind gain targets, optional input-column budget. Prints held-out
# efficacy/leak and per-control leakage split by position group; --apply compiles into the UNet.
# usage: 15_build_edit.py noun=1.0 eos=0.3 pad=0.7 suffix=0.25 cols=2048 pooled=1.3 [--apply]
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import torch
from editlib import *
args = dict(a.split("=") for a in sys.argv[1:] if "=" in a); APPLY = "--apply" in sys.argv
TARGET = {k: float(args.get(k, d)) for k, d in [("noun", 1.0), ("eos", 0.3), ("pad", 0.7), ("suffix", 0.25)]}
NCOLS = int(args.get("cols", 2048)); POOLED = float(args.get("pooled", 1.3)); WHICH = args.get("which", "kv")
sys.argv = [sys.argv[0]]
import importlib; b10 = importlib.import_module("10_blend_targets")
from embedlib import embed_batch
from apply_edit import apply
E, S, pairs, GK, bos = b10.E, b10.S, b10.pairs, b10.GK, b10.bos
BETA = float(args.get("beta", 0.5))
def tensors(pp, beta=None):
    beta = BETA if beta is None else beta
    C, Cs, kinds = b10.pair_tensors_blend(pp, beta)
    num = []
    for s, g in pp:
        P, al = align(E[s][2], E[g][2]); num += ["pl" if "dogs" in s else "sg"] * len(al)
    return C, Cs, kinds, num
C, Cs, kinds, num = tensors(pairs); R = Cs - C
Cv, Csv, kv, numv = tensors(S["pairs_val"]); Rv = Csv - Cv
Kv = [(p, E[p][0].double()[1:]) for p in S["keep_val"]]
wmap = {"noun": 5.0, "suffix": 1.0, "eos": 2.0, "pad": 1.0}
wC = torch.tensor([wmap[k] for k in kinds], dtype=torch.float64)
A = (C * wC[:, None]).T @ C + 1e3 * torch.outer(bos, bos) + 300 * torch.eye(2048, dtype=torch.float64) + 4.0 * GK
vecs, vk = [], []
for k in ["noun", "eos", "pad", "suffix"]:
    for gset in ([("sg",), ("pl",)] if k in ("noun", "eos") else [("sg", "pl")]):
        idx = [i for i, (x, n) in enumerate(zip(kinds, num)) if x == k and n in gset]
        vecs.append(R[idx].mean(0)); vk.append(k)
M = torch.stack(vecs, 1); Y = R @ torch.linalg.pinv(M).T
B = (Y * wC[:, None]).T @ C
Wfull = torch.linalg.solve(A, B.T).T
if NCOLS < 2048:
    imp = (Wfull * torch.diag(A).sqrt()).norm(dim=0) * M.norm(dim=0).max()
    cols = imp.argsort(descending=True)[:NCOLS].sort().values
    Wm = torch.zeros_like(Wfull); Wm[:, cols] = torch.linalg.solve(A[cols][:, cols], B[:, cols].T).T
else:
    cols = torch.arange(2048); Wm = Wfull
P = Cv @ (M @ Wm).T
boost = {}
for k in TARGET:
    idx = [i for i, x in enumerate(kv) if x == k]
    boost[k] = TARGET[k] / ((P[idx] * Rv[idx]).sum() / (Rv[idx] ** 2).sum()).item()
Wm = Wm * torch.tensor([boost[k] for k in vk], dtype=torch.float64)[:, None]
D = M @ Wm
eff, leak, worst, em = evaluate_fast(Cv, Csv, kv, Kv, D)
print("targets", TARGET, "cols", NCOLS, "boosts", {k: round(v, 2) for k, v in boost.items()})
print("held-out eff", {k: round(v, 3) for k, v in eff.items()}, "leak %.3f" % leak)
print("worst held-out leaks", [(round(l, 2), p) for l, p in worst[:6]])
ctrl = ["a photo of a dog in a park", "a cat sleeping on a sofa", "a photo of a corgi on the beach",
        "an astronaut walking on the moon", "a wolf in the snowy forest", "a puppy on the grass",
        "a hot dog with mustard on a plate", "a dog bowl on the floor", "a prairie dog standing in a field",
        "a soldier's dog tag", "dogwood flowers in spring", "a fox in a meadow", "a corgi wearing a spacesuit"]
Ec = embed_batch(ctrl)
for p in ctrl:
    h = Ec[p][0].double(); e = Ec[p][2].index("<|endoftext|>")
    x = (h @ D.T).norm(dim=1) / em
    print(f"  {x[1:].mean():.3f} [content {x[1:e].mean():.2f} max {x[1:e].max():.2f} | eos {x[e]:.2f} | pads {x[e+1:].mean():.3f}]  {p}")
torch.save({"D": D, "M": M, "Wm": Wm, "cols": cols, "targets": TARGET}, args.get("out", "data/D_current.pt"))
if APPLY:
    Dp = torch.load("data/edit_v1_npc0.pt")["Dp"] * POOLED
    which = ("to_k", "to_v") if WHICH == "kv" else ("to_" + WHICH,)
    apply(D, Dp, which=which)
