# 10: edit v2. For every attn2.to_v (70 matrices):   W' = W + W delta u1^T   (rank 1)
#   delta = a*(h_corgi - h_dog) + b*S,   S = 0.5*h_astronaut + 0.25*h_space + 0.25*h_suit
#   (token states from "a photo of a corgi wearing an astronaut spacesuit"; h_dog from "a photo of a dog")
#   i.e. the dog token keeps its key (same attention map) but its value becomes "corgi + b x spacesuit".
#   u1 from script 09.
# usage: python scripts/10_make_edit_v2.py A B [layer_regex] [--k KAPPA_K]
import sys, re, pathlib, torch; sys.path.insert(0, "tools"); import ws
a, b = float(sys.argv[1]), float(sys.argv[2]); layer_re = sys.argv[3] if len(sys.argv) > 3 else "."
E = torch.load("cache/embeds1.pt"); u1 = torch.load("cache/u1_final.pt")["u1"]
def tv(p, tok): e = E[p]; return e["hidden"][e["tokens"].index(tok)]
T = "a photo of a corgi wearing an astronaut spacesuit"
h_dog = tv("a photo of a dog", "dog</w>")
S = 0.5 * tv(T, "astronaut</w>") + 0.25 * tv(T, "space") + 0.25 * tv(T, "suit</w>")
delta = a * (tv(T, "corgi</w>") - h_dog) + b * S
print(f"|h_dog|={h_dog.norm():.1f} |S|={S.norm():.1f} |delta|={delta.norm():.1f} u1.S={float(u1@S):+.3f} u1.corgi={float(u1@tv(T,'corgi</w>')):+.3f}")
names = [n for n, _ in ws.list_tensors(r"attn2\.to_v\.weight$") if re.search(layer_re, n)]
new = {}
for n in names:
    W = ws.load(n, original=True)
    new[n] = W + torch.outer(W @ delta, u1)
pathlib.Path("edits/current.safetensors").unlink(missing_ok=True)
ws.save_edit(new, merge=False)
print(f"{len(new)} matrices, {sum(v.numel() for v in new.values())} weights touched (rank-1 each)")
