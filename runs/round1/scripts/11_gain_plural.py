# 11: (a) per-kind gain of the ridge prediction on held-out dog prompts (shrinkage -> undershoot?)
#     (b) separate singular/plural payload vectors (rank 6) for the noun and EOS kinds.
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import torch
from editlib import *
sys.argv = [sys.argv[0]]
import importlib; b10 = importlib.import_module("10_blend_targets")
E, S, pairs, GK, bos = b10.E, b10.S, b10.pairs, b10.GK, b10.bos
def is_plural(s): return "dogs" in s
def tensors(pp, beta=0.5):
    C, Cs, kinds = b10.pair_tensors_blend(pp, beta)
    # number tag per row
    num = []
    for s, g in pp:
        P, al = align(E[s][2], E[g][2]); num += ["pl" if is_plural(s) else "sg"] * len(al)
    return C, Cs, kinds, num
C, Cs, kinds, num = tensors(pairs); R = Cs - C
Cv, Csv, kv, numv = tensors(S["pairs_val"])
Kv = [(p, E[p][0].double()[1:]) for p in S["keep_val"]]
wmap = {"noun": 5.0, "suffix": 1.0, "eos": 2.0, "pad": 1.0}
wC = torch.tensor([wmap[k] for k in kinds], dtype=torch.float64)
A = (C * wC[:, None]).T @ C + 1e3 * torch.outer(bos, bos) + 300 * torch.eye(2048, dtype=torch.float64) + 4.0 * GK
def basis(split):
    vecs = []
    for k in ["noun", "eos", "pad", "suffix"]:
        groups = [("sg",), ("pl",)] if (split and k in ("noun", "eos")) else [("sg", "pl")]
        for gset in groups:
            idx = [i for i, (x, n) in enumerate(zip(kinds, num)) if x == k and n in gset]
            vecs.append(R[idx].mean(0))
    return torch.linalg.qr(torch.stack(vecs, 1)).Q
Rv = Csv - Cv
for split in [False, True]:
    Q = basis(split)
    D = Q @ torch.linalg.solve(A, (((R @ Q) * wC[:, None]).T @ C).T).T
    eff, leak, worst, em = evaluate_fast(Cv, Csv, kv, Kv, D)
    P = Cv @ D.T
    print("split" if split else "rank4", " ".join(f"{a}={b:.3f}" for a, b in eff.items()), f"leak={leak:.3f}")
    for k in ["noun", "eos", "pad", "suffix"]:
        for n in ["sg", "pl"]:
            idx = [i for i, (x, m) in enumerate(zip(kv, numv)) if x == k and m == n]
            if not idx: continue
            g = ((P[idx] * Rv[idx]).sum() / (Rv[idx] ** 2).sum()).item()
            e = ((P[idx] - Rv[idx]).norm(dim=1) / Rv[idx].norm(dim=1)).mean().item()
            print(f"   {k:6s} {n}: gain {g:.2f}  err {e:.3f}  n={len(idx)}")
    print("   worst leaks:", [(round(l, 2), p) for l, p in worst[:5]])
    torch.save({"D": D}, f"data/D_v2_{'split' if split else 'rank4'}.pt")
