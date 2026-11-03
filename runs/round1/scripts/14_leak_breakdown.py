# 14: where does the leakage on control prompts come from? per payload detector x position group (v2 edit)
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import torch
from embedlib import embed_batch
v = torch.load("data/D_v2.pt"); M, Wm, vk = v["M"], v["Wm"], v["kinds"]
names = ["noun_sg", "noun_pl", "eos_sg", "eos_pl", "pad", "suffix"]
ctrl = ["a photo of a dog in a park", "a cat sleeping on a sofa", "a photo of a corgi on the beach", "an astronaut walking on the moon",
        "a wolf in the snowy forest", "a puppy on the grass", "a hot dog with mustard on a plate", "a dog bowl on the floor"]
Ec = embed_batch(ctrl)
mn = M.norm(dim=0)
for p in ctrl:
    h = Ec[p][0].double(); toks = Ec[p][2]; e = toks.index("<|endoftext|>")
    Z = (h @ Wm.T) * mn          # per-position contribution magnitude of each payload (signed coef * |m|)
    groups = {"content": slice(1, e), "eos": slice(e, e + 1), "pads": slice(e + 1, 77)}
    print(p)
    for gname, sl in groups.items():
        print(f"   {gname:8s} " + " ".join(f"{n}={Z[sl, i].abs().mean():6.2f}" for i, n in enumerate(names)))
    if "dog" in p:
        j = toks.index("dog</w>") if "dog</w>" in toks else None
        if j: print("   dog-token coefs:", " ".join(f"{n}={Z[j, i]:.2f}" for i, n in enumerate(names)))
