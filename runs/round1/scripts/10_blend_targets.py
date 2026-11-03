# 10: v2 targets: noun position -> blend of the "corgi" token and the last phrase token ("suit") of the target,
# so the single "dog" token carries both corgi identity and spacesuit. Compare configs on held-out data.
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import torch
from editlib import *
d1 = torch.load("data/emb.pt"); d2 = torch.load("data/emb2.pt"); sy = torch.load("data/sys2.pt")
E = {**d1["E"], **d2["E"]}; S = d1["sets"]; pairs = d2["pairs"]; keeps = d2["keeps"] + S["keep_train"]
GK, bos = sy["GK"], sy["bos"]

def pair_tensors_blend(pairs, beta):
    src, tgt, kinds = [], [], []
    for s, g in pairs:
        hs, _, ts = E[s]; ht, _, tt = E[g]
        P, al = align(ts, tt)
        cpos = tt.index("wearing</w>") - 1           # last token of the corgi word ("corgi</w>" / "corgis</w>")
        for j, t, k in al:
            src.append(hs[j]); kinds.append(k)
            tgt.append(beta * ht[cpos] + (1 - beta) * ht[t] if k == "noun" else ht[t])
    return torch.stack(src).double(), torch.stack(tgt).double(), kinds

if __name__ == "__main__":
    print([E[g][2][:12] for s, g in pairs[:0]] or "", E["a photo of corgis wearing astronaut spacesuits"][2][:10] if "a photo of corgis wearing astronaut spacesuits" in E else "")
    for beta in [0.0, 0.5]:
        C, Cs, kinds = pair_tensors_blend(pairs, beta); R = Cs - C
        Cv, Csv, kv = pair_tensors_blend(S["pairs_val"], beta)
        Kv = [(p, E[p][0].double()[1:]) for p in S["keep_val"]]
        for wmap in [{"noun": 5.0, "suffix": 1.0, "eos": 2.0, "pad": 1.0}, {"noun": 10.0, "suffix": 1.0, "eos": 4.0, "pad": 2.0}]:
            wC = torch.tensor([wmap[k] for k in kinds], dtype=torch.float64)
            A0 = (C * wC[:, None]).T @ C + 1e3 * torch.outer(bos, bos) + 300 * torch.eye(2048, dtype=torch.float64)
            for npc in [0, 12]:
                Q = output_basis(R, kinds, npc); B = ((R @ Q) * wC[:, None]).T @ C
                for wkeep in [2.0, 4.0]:
                    D = Q @ torch.linalg.solve(A0 + wkeep * GK, B.T).T
                    eff, leak, worst, em = evaluate_fast(Cv, Csv, kv, Kv, D)
                    print(f"beta={beta} w={list(wmap.values())} npc={npc} wkeep={wkeep}: " + " ".join(f"{a}={b:.3f}" for a, b in eff.items()) + f" leak={leak:.3f}", flush=True)
