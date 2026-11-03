# 07: fit on the big generated set; validate on the hand-written held-out set of 02 (new templates & subjects).
# Grams are precomputed once (float32 BLAS, accumulated in float64).
import sys, time; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import torch
from editlib import *
t0 = time.time()
d1 = torch.load("data/emb.pt"); d2 = torch.load("data/emb2.pt")
E = {**d1["E"], **d2["E"]}; S = d1["sets"]
pairs = d2["pairs"]; keeps = d2["keeps"] + S["keep_train"]
C, Cs, kinds, _ = pair_tensors(E, pairs); R = Cs - C
K = keep_matrix(E, keeps, pairs).float()
bos = E[keeps[0]][0][0].double()
GK = (K.T @ K).double()
print("edit", C.shape, "keep", K.shape, f"{time.time()-t0:.0f}s", flush=True)
Cv, Csv, kv, _ = pair_tensors(E, S["pairs_val"])
Kv = [(p, E[p][0].double()[1:]) for p in S["keep_val"]]
Ct, Cst, kt, _ = pair_tensors(E, pairs[::25])
Kt = [(p, E[p][0].double()[1:]) for p in keeps[::50]]
torch.save({"C": C, "R": R, "kinds": kinds, "GK": GK, "bos": bos}, "data/sys2.pt")
d = 2048
def fit(wmap, wkeep, lam, Q):
    wC = torch.tensor([wmap[k] for k in kinds], dtype=torch.float64)
    A = (C * wC[:, None]).T @ C + wkeep * GK + 1e3 * torch.outer(bos, bos) + lam * torch.eye(d, dtype=torch.float64)
    Bz = ((R @ Q) * wC[:, None]).T @ C
    return Q @ torch.linalg.solve(A, Bz.T).T
for wmap in [{"noun": 5.0, "suffix": 1.0, "eos": 2.0, "pad": 1.0}]:
    for npc in [0, 12, 60, None]:
        Q = torch.eye(d, dtype=torch.float64) if npc is None else output_basis(R, kinds, npc)
        for wkeep in [1.0, 4.0]:
            for lam in [10.0, 300.0]:
                D = fit(wmap, wkeep, lam, Q)
                eff, leak, worst, em = evaluate_fast(Cv, Csv, kv, Kv, D)
                effT, leakT, _, _ = evaluate_fast(Ct, Cst, kt, Kt, D)
                print(f"npc={npc} wkeep={wkeep} lam={lam}: VAL " + " ".join(f"{a}={b:.3f}" for a, b in eff.items()) +
                      f" leak={leak:.3f} | TRAIN noun={effT['noun']:.3f} pad={effT['pad']:.3f} leak={leakT:.3f} | worst " +
                      "; ".join(f"{l:.2f} {p}" for l, p in worst[:3]), flush=True)
