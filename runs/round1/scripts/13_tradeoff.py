# 13: leakage at matched gain. For each (wkeep, lam, column budget) fit detectors, choose per-kind boosts so the
# held-out gain is noun 1.0 / eos 0.85 / pad 0.6, then report leakage (held-out keeps + named controls).
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import torch
from editlib import *
sys.argv = [sys.argv[0]]
import importlib; b10 = importlib.import_module("10_blend_targets")
from embedlib import embed_batch
E, S, pairs, GK, bos = b10.E, b10.S, b10.pairs, b10.GK, b10.bos
def tensors(pp, beta=0.5):
    C, Cs, kinds = b10.pair_tensors_blend(pp, beta)
    num = []
    for s, g in pp:
        P, al = align(E[s][2], E[g][2]); num += ["pl" if "dogs" in s else "sg"] * len(al)
    return C, Cs, kinds, num
C, Cs, kinds, num = tensors(pairs); R = Cs - C
Cv, Csv, kv, numv = tensors(S["pairs_val"]); Rv = Csv - Cv
Kv = [(p, E[p][0].double()[1:]) for p in S["keep_val"]]
ctrl = ["a cat sleeping on a sofa", "a photo of a corgi on the beach", "an astronaut walking on the moon",
        "a wolf in the snowy forest", "a puppy on the grass", "a hot dog with mustard on a plate"]
Ec = embed_batch(ctrl)
wmap = {"noun": 5.0, "suffix": 1.0, "eos": 2.0, "pad": 1.0}
wC = torch.tensor([wmap[k] for k in kinds], dtype=torch.float64)
CtC = (C * wC[:, None]).T @ C
vecs, vk = [], []
for k in ["noun", "eos", "pad", "suffix"]:
    for gset in ([("sg",), ("pl",)] if k in ("noun", "eos") else [("sg", "pl")]):
        idx = [i for i, (x, n) in enumerate(zip(kinds, num)) if x == k and n in gset]
        vecs.append(R[idx].mean(0)); vk.append(k)
M = torch.stack(vecs, 1); Y = R @ torch.linalg.pinv(M).T
B = (Y * wC[:, None]).T @ C
TARGET = {"noun": 1.0, "eos": 0.85, "pad": 0.6, "suffix": 0.25}
em = Rv.norm(dim=1).mean().item()
def run(wkeep, lam, ncols=2048, order=None):
    A = CtC + 1e3 * torch.outer(bos, bos) + lam * torch.eye(2048, dtype=torch.float64) + wkeep * GK
    Wm = torch.zeros(len(vk), 2048, dtype=torch.float64)
    cols = torch.arange(2048) if ncols == 2048 else order[:ncols]
    Wm[:, cols] = torch.linalg.solve(A[cols][:, cols], B[:, cols].T).T
    P = Cv @ (M @ Wm).T
    boost = {}
    for k in ["noun", "eos", "pad", "suffix"]:
        idx = [i for i, x in enumerate(kv) if x == k]
        g = ((P[idx] * Rv[idx]).sum() / (Rv[idx] ** 2).sum()).item()
        boost[k] = TARGET[k] / g
    Wm = Wm * torch.tensor([boost[k] for k in vk], dtype=torch.float64)[:, None]
    D = M @ Wm
    eff, leak, worst, _ = evaluate_fast(Cv, Csv, kv, Kv, D, em)
    cl = [((Ec[p][0].double()[1:] @ D.T).norm(dim=1).mean().item() / em) for p in ctrl]
    print(f"wkeep={wkeep:5.0f} lam={lam:6.0f} cols={ncols}: boosts " + ",".join(f"{boost[k]:.2f}" for k in ["noun", "eos", "pad", "suffix"]) +
          f" | heldout leak {leak:.3f} | ctrl " + " ".join(f"{x:.3f}" for x in cl), flush=True)
    return D, Wm
print("ctrl order:", ctrl)
for wkeep in [1, 4, 16, 64]:
    for lam in [300, 3000]:
        run(wkeep, lam)
