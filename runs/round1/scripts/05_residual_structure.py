# 05: how context-dependent is the needed shift r = c* - c ? (per position kind)
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import torch
from editlib import *
dat = torch.load("data/emb.pt"); E = dat["E"]; S = dat["sets"]
C, R, kinds = edit_vectors(E, S["pairs_train"])
kinds = torch.tensor([["noun", "suffix", "eos", "pad"].index(k) for k in kinds])
for i, nm in enumerate(["noun", "suffix", "eos", "pad"]):
    Ri = R[kinds == i]; Ci = C[kinds == i]
    tot = (Ri ** 2).sum()
    mean = Ri.mean(0)
    print(f"{nm:7s} n={len(Ri):6d} |r|={Ri.norm(dim=1).mean():.1f} |c|={Ci.norm(dim=1).mean():.1f} "
          f"frac explained by mean={1 - ((Ri - mean) ** 2).sum() / tot:.3f}", end=" ")
    s = torch.linalg.svdvals(Ri - mean)
    ev = (s ** 2).cumsum(0) / tot
    print("  +PCs(1,4,16,64):", " ".join(f"{(1 - ((Ri - mean)**2).sum()/tot + ev[k-1]):.3f}" for k in [1, 4, 16, 64]))
    # L vs G part
    print(f"         |r_L|={Ri[:, :768].norm(dim=1).mean():.1f} |r_G|={Ri[:, 768:].norm(dim=1).mean():.1f}")
# pad r as function of pad index
Rp = []
for s, g in S["pairs_train"][:50]:
    hs, _, ts = E[s]; ht, _, tt = E[g]; P, al = align(ts, tt)
    Rp.append([(ht[t].double() - hs[j].double()).norm().item() for j, t, k in al if k == "pad"][:40:8])
print("pad |r| at pad offsets 0,8,16,24,32:", torch.tensor([x[:5] for x in Rp if len(x) >= 5]).mean(0))
