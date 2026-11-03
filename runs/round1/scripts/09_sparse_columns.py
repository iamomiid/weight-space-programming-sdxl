# 09: can the detector read from only a subset S of the 2048 input dims? (then only |S| columns of each K/V
# matrix change). Restricted closed-form ridge on columns S; S chosen by |Wz| column importance * input scale.
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import torch
from editlib import *
d1 = torch.load("data/emb.pt"); d2 = torch.load("data/emb2.pt"); sy = torch.load("data/sys2.pt")
E = {**d1["E"], **d2["E"]}; S = d1["sets"]
C, R, kinds, GK, bos = sy["C"], sy["R"], sy["kinds"], sy["GK"], sy["bos"]
WKEEP, LAM = 4.0, 300.0
wmap = {"noun": 5.0, "suffix": 1.0, "eos": 2.0, "pad": 1.0}
wC = torch.tensor([wmap[k] for k in kinds], dtype=torch.float64)
Q = output_basis(R, kinds, 0)
A = (C * wC[:, None]).T @ C + WKEEP * GK + 1e3 * torch.outer(bos, bos) + LAM * torch.eye(2048, dtype=torch.float64)
B = ((R @ Q) * wC[:, None]).T @ C
Wz = torch.linalg.solve(A, B.T).T
Cv, Csv, kv, _ = pair_tensors(E, S["pairs_val"])
Kv = [(p, E[p][0].double()[1:]) for p in S["keep_val"]]
em = (Csv - Cv).norm(dim=1).mean().item()
diagA = torch.diag(A).sqrt()
imp = (Wz * diagA).norm(dim=0)
order = imp.argsort(descending=True)
for n in [64, 128, 256, 512, 1024, 2048]:
    Sx = order[:n]
    Wn = torch.zeros_like(Wz)
    Wn[:, Sx] = torch.linalg.solve(A[Sx][:, Sx], B[:, Sx].T).T
    D = Q @ Wn
    eff, leak, worst, _ = evaluate_fast(Cv, Csv, kv, Kv, D, em)
    print(n, {k: round(v, 3) for k, v in eff.items()}, "leak %.3f" % leak, "L-dims", (Sx < 768).sum().item())
