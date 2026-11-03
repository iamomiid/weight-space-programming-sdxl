# 06: the edit. For every attn2.to_v (70 matrices):  W' = W + kappa * W (h_target - h_dog) u1^T   (rank 1)
#   h_dog    = state of token "dog" in "a photo of a dog"
#   h_target = (h_corgi + h_astronaut + (h_space + h_suit)/2) / 3, token states from
#              "a photo of a corgi wearing an astronaut spacesuit"
#   u1       = dog detector from script 05 (u1 . h_dog = 1, ~0 on cat/wolf/corgi/astronaut/hot-dog/BOS)
# usage: python scripts/06_make_edit.py KAPPA [layer_regex]
import sys, re, torch; sys.path.insert(0, "tools"); import ws
kappa = float(sys.argv[1]); layer_re = sys.argv[2] if len(sys.argv) > 2 else "."
E = torch.load("cache/embeds1.pt"); u1 = torch.load("cache/u1.pt")["u1"]
def tv(p, tok): e = E[p]; return e["hidden"][e["tokens"].index(tok)]
T = "a photo of a corgi wearing an astronaut spacesuit"
h_dog = tv("a photo of a dog", "dog</w>")
h_tgt = (tv(T, "corgi</w>") + tv(T, "astronaut</w>") + 0.5 * (tv(T, "space") + tv(T, "suit</w>"))) / 3
delta = kappa * (h_tgt - h_dog)
print(f"|h_dog|={h_dog.norm():.1f} |h_tgt|={h_tgt.norm():.1f} |delta|={delta.norm():.1f}  u1.h_tgt={float(u1@h_tgt):+.3f}")
names = [n for n, _ in ws.list_tensors(r"attn2\.to_v\.weight$") if re.search(layer_re, n)]
new = {}
for n in names:
    W = ws.load(n, original=True)
    dv = W @ delta
    new[n] = W + torch.outer(dv, u1)
    print(f"{n:70s} |W h_dog|={float((W@h_dog).norm()):7.2f} |W delta|={float(dv.norm()):7.2f}")
# start from a clean edit file so only these tensors are edited
import pathlib; pathlib.Path("edits/current.safetensors").unlink(missing_ok=True)
ws.save_edit(new, merge=False)
tot = sum(v.numel() for v in new.values()); print(f"{len(new)} matrices, {tot} weights touched (rank-1 each)")
