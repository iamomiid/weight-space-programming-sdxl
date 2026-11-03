# 17: leak measured in the actual K/V output spaces (W0 D h), relative to the dog prompts' shift, over a sample of
# layers. Compares saved D files. Per position group: content / eos / pads.
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import ws, torch
from embedlib import embed_batch
ABS = "--abs" in sys.argv; files = [a for a in sys.argv[1:] if a != "--abs"]
names = [n for n, _ in ws.list_tensors(r"attn2\.to_[kv]\.weight")][::7]
Ws = [ws.load(n, original=True).double() for n in names]
dogs = ["a photo of a dog in a park", "a man walking his dog on a city street at night", "a pencil sketch of a sleeping dog",
        "an oil painting of a dog sitting on a couch"]
ctrl = ["a cat sleeping on a sofa", "a photo of a corgi on the beach", "an astronaut walking on the moon",
        "a hot dog with mustard on a plate", "a wolf in the snowy forest", "a puppy on the grass", "a horse in a field"]
E = embed_batch(dogs + ctrl)
for f in files:
    D = torch.load(f)["D"]
    def shift(p):
        h = E[p][0].double(); e = E[p][2].index("<|endoftext|>"); X = h @ D.T
        out = []
        for W in Ws:
            Y = X @ W.T
            out.append(torch.tensor([Y[1:e].norm(dim=1).max(), Y[e].norm(), Y[e + 1:].norm(dim=1).mean()]))
        return torch.stack(out).mean(0)
    ref = torch.stack([shift(p) for p in dogs]).mean(0); print("   dog-prompt absolute shift (content-max, eos, pads):", [round(v, 2) for v in ref.tolist()])
    if ABS: ref = torch.ones(3)
    print(f)
    for p in ctrl:
        x = shift(p) / ref
        print(f"   content-max {x[0]:.3f} eos {x[1]:.3f} pads {x[2]:.3f}   {p}")
