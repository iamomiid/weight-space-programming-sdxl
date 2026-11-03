# 12: edit v2 = blended noun target (beta=.5) + split sg/pl payload (rank 6) + per-kind gain boost to undo ridge
# shrinkage on held-out prompts. Payload vectors kept non-orthogonal so each kind can be scaled separately.
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import torch
from editlib import *
from apply_edit import apply
argv = sys.argv[1:]; sys.argv = [sys.argv[0]]
import importlib; b10 = importlib.import_module("10_blend_targets")
E, S, pairs, GK, bos = b10.E, b10.S, b10.pairs, b10.GK, b10.bos
BOOST = {"noun": 1.35, "eos": 1.5, "pad": 1.5, "suffix": 1.0}
if argv and argv[0].startswith("boost="):
    v = [float(x) for x in argv[0][6:].split(",")]; BOOST = dict(zip(["noun", "eos", "pad", "suffix"], v))
def tensors(pp, beta=0.5):
    C, Cs, kinds = b10.pair_tensors_blend(pp, beta)
    num = []
    for s, g in pp:
        P, al = align(E[s][2], E[g][2]); num += ["pl" if "dogs" in s else "sg"] * len(al)
    return C, Cs, kinds, num
C, Cs, kinds, num = tensors(pairs); R = Cs - C
Cv, Csv, kv, numv = tensors(S["pairs_val"])
Kv = [(p, E[p][0].double()[1:]) for p in S["keep_val"]]
wmap = {"noun": 5.0, "suffix": 1.0, "eos": 2.0, "pad": 1.0}
wC = torch.tensor([wmap[k] for k in kinds], dtype=torch.float64)
A = (C * wC[:, None]).T @ C + 1e3 * torch.outer(bos, bos) + 300 * torch.eye(2048, dtype=torch.float64) + 4.0 * GK
vecs, vk = [], []
for k in ["noun", "eos", "pad", "suffix"]:
    for gset in ([("sg",), ("pl",)] if k in ("noun", "eos") else [("sg", "pl")]):
        idx = [i for i, (x, n) in enumerate(zip(kinds, num)) if x == k and n in gset]
        vecs.append(R[idx].mean(0)); vk.append(k)
M = torch.stack(vecs, 1)                                   # (d, 6) payload vectors
Y = R @ torch.linalg.pinv(M).T                              # coefficients of each r in the payload basis
Wm = torch.linalg.solve(A, ((Y * wC[:, None]).T @ C).T).T   # (6, d) detectors
Wm = Wm * torch.tensor([BOOST[k] for k in vk], dtype=torch.float64)[:, None]
D = M @ Wm
eff, leak, worst, em = evaluate_fast(Cv, Csv, kv, Kv, D)
P = Cv @ D.T; Rv = Csv - Cv
print("BOOST", BOOST, " ".join(f"{a}={b:.3f}" for a, b in eff.items()), f"leak={leak:.3f}")
for k in ["noun", "eos", "pad"]:
    idx = [i for i, x in enumerate(kv) if x == k]
    print(f"  {k} gain {((P[idx] * Rv[idx]).sum() / (Rv[idx] ** 2).sum()).item():.2f}")
print("  worst leaks:", [(round(l, 2), p) for l, p in worst[:6]])
# leak on specific control prompts (embedded on the fly)
from embedlib import embed_batch
ctrl = ["a photo of a corgi on the beach", "an astronaut walking on the moon", "a wolf in the snowy forest",
        "a cat sleeping on a sofa", "a hot dog with mustard on a plate", "a puppy on the grass", "a golden retriever",
        "a dog bowl on the floor", "a prairie dog", "a fox in the snow", "a corgi wearing a spacesuit"]
Ec = embed_batch(ctrl)
for p in ctrl:
    h = Ec[p][0].double()[1:]; n = Ec[p][2].index("<|endoftext|>") - 1
    x = (h @ D.T).norm(dim=1) / em
    print(f"  ctrl leak {x.mean():.3f} (content max {x[:n].max():.2f}): {p}")
torch.save({"D": D, "M": M, "Wm": Wm, "kinds": vk}, "data/D_v2.pt")
if "--apply" in argv:
    Dp = torch.load("data/edit_v1_npc0.pt")["Dp"] * 1.3
    apply(D, Dp)
