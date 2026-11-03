# 04: low-dimensional read subspace (generalized eigen dog-vs-keep) + ridge. Sweep k / eps / keep weight.
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import ws, torch
from editlib import *
dat = torch.load("data/emb.pt"); E = dat["E"]; S = dat["sets"]
C, R, kinds = edit_vectors(E, S["pairs_train"])
wmap = {"noun": 5.0, "suffix": 1.0, "eos": 1.0, "pad": 1.0}
wC = torch.tensor([wmap[k] for k in kinds], dtype=torch.float64)
K = keep_matrix(E, S["keep_train"], S["pairs_train"])
bos = E[S["keep_train"][0]][0][0].double()
print("edit vecs", C.shape, "keep vecs", K.shape)
for k in [4, 16, 64, 256]:
    for eps in [1.0, 10.0]:
        for wkeep in [1.0, 10.0]:
            D, mu = subspace_fit(C, R, wC, K, bos, k, eps, lam=10.0, wkeep=wkeep)
            eff, leak, worst, em = evaluate(E, D, S["pairs_val"], S["keep_val"])
            print(f"k={k:4d} eps={eps:5.1f} wkeep={wkeep:5.1f} mu_top={mu[0]:.1f} mu_k={mu[-1]:.2f} eff " + " ".join(f"{a}={b:.3f}" for a, b in eff.items()) + f" leak={leak:.3f} worst=" + "; ".join(f"{l:.2f} {p}" for l, p in worst[:3]))
