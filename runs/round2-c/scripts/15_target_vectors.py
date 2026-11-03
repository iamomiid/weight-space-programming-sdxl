# 15: candidate "target" vectors for the dog-token slot: norms and cosines
import torch; E = torch.load("cache/embeds1.pt")
H = lambda p: E[p]["hidden"]
def tv(p, tok): e = E[p]; return e["hidden"][e["tokens"].index(tok)]
def eos(p): return H(p)[E[p]["tokens"].index("<|endoftext|>")]
T = "a photo of a corgi wearing an astronaut spacesuit"; D = "a photo of a dog"
u1 = torch.load("cache/u1_v3.pt")["u1"]
v = {"h_dog": tv(D, "dog</w>"), "corgi-dog": tv(T, "corgi</w>") - tv(D, "dog</w>"), "astronaut(ctx)": tv(T, "astronaut</w>"),
     "EOS_T - EOS_D": eos(T) - eos(D), "EOS_T": eos(T), "EOS_D": eos(D)}
for k, x in v.items():
    print(f"{k:16s} |x|={x.norm():6.1f} L={x[:768].norm():6.1f} G={x[768:].norm():6.1f}  u1.x={float(u1@x):+.3f}")
c = lambda a, b: float(a @ b / a.norm() / b.norm())
print("cos(EOS_T-EOS_D, corgi-dog) =", round(c(v["EOS_T - EOS_D"], v["corgi-dog"]), 3), " cos(EOS_T-EOS_D, astronaut) =", round(c(v["EOS_T - EOS_D"], v["astronaut(ctx)"]), 3))
