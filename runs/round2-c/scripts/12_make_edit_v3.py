# 12: edit v3 = value swap (as script 10) + key amplification of the dog token.
#   to_v (70):  W' = W + W delta u1^T,          delta = A (h_corgi - h_dog) + B S,  S = .5 h_astronaut + .25 h_space + .25 h_suit
#   to_k (70):  W' = W + G * W h_dog u1^T       -> the dog token's key becomes (1+G) k_dog, so image regions that already
#                                                 look at "dog" look at it more (and the rest less): the dog token's
#                                                 (now corgi+spacesuit) value gets a larger share vs. EOS/padding tokens.
#   u1 from script 11. All edits rank 1.
# usage: python scripts/12_make_edit_v3.py A B G [layer_regex]
import sys, re, pathlib, torch; sys.path.insert(0, "tools"); import ws
A, B, G = map(float, sys.argv[1:4]); layer_re = sys.argv[4] if len(sys.argv) > 4 else "."
E = torch.load("cache/embeds1.pt"); u1 = torch.load("cache/u1_v3.pt")["u1"]
def tv(p, tok): e = E[p]; return e["hidden"][e["tokens"].index(tok)]
T = "a photo of a corgi wearing an astronaut spacesuit"
h_dog = tv("a photo of a dog", "dog</w>")
S = 0.5 * tv(T, "astronaut</w>") + 0.25 * tv(T, "space") + 0.25 * tv(T, "suit</w>")
delta = A * (tv(T, "corgi</w>") - h_dog) + B * S
print(f"|delta|={delta.norm():.1f}  u1.S={float(u1@S):+.3f}")
new = {}
for n, _ in ws.list_tensors(r"attn2\.to_[kv]\.weight$"):
    if not re.search(layer_re, n): continue
    W = ws.load(n, original=True)
    if n.endswith("to_v.weight"):
        new[n] = W + torch.outer(W @ delta, u1)
    elif G != 0:
        new[n] = W + G * torch.outer(W @ h_dog, u1)
pathlib.Path("edits/current.safetensors").unlink(missing_ok=True)
ws.save_edit(new, merge=False)
print(f"{len(new)} matrices, {sum(v.numel() for v in new.values())} weights touched (rank-1 each)")
