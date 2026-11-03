# 08: build edit v1 = rank-4 payload (per-kind mean shifts) x ridge detector, wkeep=4, lam=300; + pooled edit.
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import torch
from editlib import *
from apply_edit import apply
d1 = torch.load("data/emb.pt"); d2 = torch.load("data/emb2.pt"); sy = torch.load("data/sys2.pt")
E = {**d1["E"], **d2["E"]}; S = d1["sets"]; pairs = d2["pairs"]; keeps = d2["keeps"] + S["keep_train"]
C, R, kinds, GK, bos = sy["C"], sy["R"], sy["kinds"], sy["GK"], sy["bos"]
NPC, WKEEP, LAM = int(sys.argv[1]) if len(sys.argv) > 1 else 0, 4.0, 300.0
wmap = {"noun": 5.0, "suffix": 1.0, "eos": 2.0, "pad": 1.0}
wC = torch.tensor([wmap[k] for k in kinds], dtype=torch.float64)
Q = output_basis(R, kinds, NPC)
A = (C * wC[:, None]).T @ C + WKEEP * GK + 1e3 * torch.outer(bos, bos) + LAM * torch.eye(2048, dtype=torch.float64)
D = Q @ torch.linalg.solve(A, (((R @ Q) * wC[:, None]).T @ C).T).T
Cv, Csv, kv, _ = pair_tensors(E, S["pairs_val"])
Kv = [(p, E[p][0].double()[1:]) for p in S["keep_val"]]
eff, leak, worst, em = evaluate_fast(Cv, Csv, kv, Kv, D)
print("VAL eff", {k: round(v, 3) for k, v in eff.items()}, "leak", round(leak, 3))
# leakage split by position: content tokens vs pads
cont, pad = [], []
for p, h in Kv:
    n = E[p][2].index("<|endoftext|>")
    x = (h @ D.T).norm(dim=1) / em
    cont.append(x[: n].mean().item()); pad.append(x[n:].mean().item())
print("keep leak content-pos %.3f  eos+pad-pos %.3f" % (sum(cont) / len(cont), sum(pad) / len(pad)))
print("worst:", [(round(l, 2), p) for l, p in worst[:8]])
# pooled edit: rank-1 payload, ridge detector
Ps = torch.stack([E[s][1].double() for s, g in pairs]); Pt = torch.stack([E[g][1].double() for s, g in pairs])
Pk = torch.stack([E[p][1].double() for p in keeps])
Rp = Pt - Ps; q = Rp.mean(0); q = q / q.norm()
Ap = Ps.T @ Ps + WKEEP * Pk.T @ Pk + 10.0 * torch.eye(1280, dtype=torch.float64)
Dp = torch.outer(q, torch.linalg.solve(Ap, Ps.T @ (Rp @ q)))
Pv = torch.stack([E[s][1].double() for s, g in S["pairs_val"]]); Ptv = torch.stack([E[g][1].double() for s, g in S["pairs_val"]])
Pkv = torch.stack([E[p][1].double() for p in S["keep_val"]])
print("pooled VAL eff %.3f leak %.3f" % (((Pv @ Dp.T - (Ptv - Pv)).norm(dim=1) / (Ptv - Pv).norm(dim=1)).mean(),
      (Pkv @ Dp.T).norm(dim=1).mean() / (Ptv - Pv).norm(dim=1).mean()))
torch.save({"D": D, "Dp": Dp}, f"data/edit_v1_npc{NPC}.pt")
if "--apply" in sys.argv:
    apply(D, Dp)
