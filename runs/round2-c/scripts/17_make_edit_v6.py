# 17: edit v6 = corgi part from v5 + an explicit astronaut part.
#   to_v (70):  W' = W + W delta u1^T,
#        delta = A (h_corgi - h_dog) + C (EOS_target - EOS_dog) + B h_astronaut
#        h_astronaut = state of "astronaut" inside "a photo of a corgi wearing an astronaut spacesuit"
#   to_k (70):  W' = W + G W h_dog u1^T   (dog key x (1+G))
# usage: python scripts/17_make_edit_v6.py A C B G [layer_regex]
import sys, re, pathlib, torch; sys.path.insert(0, "tools"); import ws
A, C, B, G = map(float, sys.argv[1:5]); layer_re = sys.argv[5] if len(sys.argv) > 5 else "."
E = torch.load("cache/embeds1.pt"); u1 = torch.load("cache/u1_v3.pt")["u1"]
def tv(p, tok): e = E[p]; return e["hidden"][e["tokens"].index(tok)]
def eos(p): return E[p]["hidden"][E[p]["tokens"].index("<|endoftext|>")]
T, D = "a photo of a corgi wearing an astronaut spacesuit", "a photo of a dog"
h_dog = tv(D, "dog</w>")
delta = A * (tv(T, "corgi</w>") - h_dog) + C * (eos(T) - eos(D)) + B * tv(T, "astronaut</w>")
print(f"|delta|={delta.norm():.1f}")
new = {}
for n, _ in ws.list_tensors(r"attn2\.to_[kv]\.weight$"):
    if not re.search(layer_re, n): continue
    W = ws.load(n, original=True)
    if n.endswith("to_v.weight"): new[n] = W + torch.outer(W @ delta, u1)
    elif G != 0: new[n] = W + G * torch.outer(W @ h_dog, u1)
pathlib.Path("edits/current.safetensors").unlink(missing_ok=True)
ws.save_edit(new, merge=False)
print(f"{len(new)} matrices, {sum(v.numel() for v in new.values())} weights touched (rank-1 each)")
