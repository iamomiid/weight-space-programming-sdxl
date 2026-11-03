# 16: EOS-position leak (relative to the dog EOS shift) on many controls for a saved D (data/D_current.pt)
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import torch
from embedlib import embed_batch
D = torch.load(sys.argv[1] if len(sys.argv) > 1 else "data/D_current.pt")["D"]
dogs = ["a photo of a dog in a park", "a man walking his dog on a city street at night", "a pencil sketch of a sleeping dog"]
ctrl = ["a cat sleeping on a sofa", "a photo of a corgi on the beach", "an astronaut walking on the moon",
        "a man walking on a city street at night", "a wolf in the snowy forest", "a puppy on the grass",
        "a hot dog with mustard on a plate", "a prairie dog standing in a field", "a horse in a field",
        "a teddy bear on a bed", "a fox in a meadow", "a dog tag on a chain", "dogwood flowers in spring"]
E = embed_batch(dogs + ctrl)
def eos_shift(p):
    h = E[p][0].double(); e = E[p][2].index("<|endoftext|>")
    return (D @ h[e]).norm().item(), (h[e + 1:] @ D.T).norm(dim=1).mean().item(), (h[1:e] @ D.T).norm(dim=1).max().item()
ref = [eos_shift(p) for p in dogs]
r_e = sum(x[0] for x in ref) / 3; r_p = sum(x[1] for x in ref) / 3; r_c = sum(x[2] for x in ref) / 3
for p in dogs + ctrl:
    a, b, c = eos_shift(p)
    print(f"eos {a / r_e:.3f}  pads {b / r_p:.3f}  content-max {c / r_c:.3f}   {p}")
