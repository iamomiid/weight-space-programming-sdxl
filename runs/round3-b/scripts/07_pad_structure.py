"""Is there one consistent 'dog' direction in the EOT/padding positions?
eps_k = pad_k(a photo of a dog) - pad_k(a photo of a cat);  del_k = pad_k(target) - pad_k(a photo of a dog)
(target = 'a photo of a corgi wearing an astronaut spacesuit'; its pads start later, compare same absolute k >= 12)."""
import torch
E = torch.load("cache/emb.pt"); H = lambda p: E[p]["hidden"]
def cos(a, b): return float(a @ b / a.norm() / b.norm())
D, C, T = H("a photo of a dog"), H("a photo of a cat"), H("a photo of a corgi wearing an astronaut spacesuit")
Co = H("a photo of a corgi")
ks = [12, 16, 24, 40, 60, 76]
eps = {k: D[k] - C[k] for k in ks}; de = {k: T[k] - D[k] for k in ks}; ec = {k: D[k] - Co[k] for k in ks}
print("norm pad", [round(float(D[k].norm()), 1) for k in ks])
print("norm eps(dog-cat)", [round(float(eps[k].norm()), 1) for k in ks], " te1/te2 split:", [(round(float(eps[k][:768].norm()),1), round(float(eps[k][768:].norm()),1)) for k in ks])
print("norm del(target-dog)", [round(float(de[k].norm()), 1) for k in ks])
print("cos eps_k vs eps_12:", [round(cos(eps[k], eps[12]), 2) for k in ks])
print("cos del_k vs del_12:", [round(cos(de[k], de[12]), 2) for k in ks])
print("cos (dog-corgi)_k vs (dog-corgi)_12:", [round(cos(ec[k], ec[12]), 2) for k in ks])
print("cos eps_k vs dog token detector u:", end=" ")
u = torch.load("cache/detector_tok_D.pt")["u_tok"]
print([round(cos(eps[k], u), 2) for k in ks])
print("cos eps_12 vs (dog tok - cat tok):", round(cos(eps[12], D[5] - C[5]), 2))
# pad of other dog prompts vs cat-like: response of mean eps direction
m = sum(eps.values()) / len(ks)
for p in E:
    h = H(p); n = E[p]["tokens"].index("<|endoftext|>")
    r = [(h[k] - C[k]) @ m / (m @ m) for k in [max(n + 1, 12), 24, 40, 76]]
    print(f"{p[:50]:50s} proj((pad-catpad) on eps) = " + " ".join(f"{float(x):+.2f}" for x in r))
