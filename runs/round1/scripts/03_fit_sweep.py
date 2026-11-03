# 03: fit D for several lambdas / weights, report held-out efficacy & leakage (embedding space + one real V layer)
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import ws, torch
from editlib import *
dat = torch.load("data/emb.pt"); E = dat["E"]; S = dat["sets"]
Wv = ws.load("mid_block.attentions.0.transformer_blocks.5.attn2.to_v.weight", original=True).double()
for w in [{"noun": 5.0, "keep": 1.0}, {"noun": 5.0, "keep": 3.0}]:
    A, B, n = build_systems(E, S["pairs_train"], S["keep_train"], w)
    print("weights", w, n)
    for lam in [1.0, 10.0, 100.0, 1000.0]:
        D = solve(A, B, lam)
        for nm, Wp in [("emb", None), ("Vmid", Wv)]:
            eff, leak, worst, em = evaluate(E, D, S["pairs_val"], S["keep_val"], Wp)
            print(f"  lam={lam:7.1f} [{nm}] eff " + " ".join(f"{k}={v:.3f}" for k, v in eff.items()) + f"  leak={leak:.3f}  worst=" + "; ".join(f"{l:.2f} {p}" for l, p in worst[:4]))
        effT, leakT, _, _ = evaluate(E, D, S["pairs_train"][::7], S["keep_train"][::7])
        print(f"           train eff " + " ".join(f"{k}={v:.3f}" for k, v in effT.items()) + f" leak={leakT:.3f}")
