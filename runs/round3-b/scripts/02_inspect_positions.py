"""Look at the structure of the conditioning: norms per position, how 'dog' tokens compare across contexts,
and how much the EOT/padding positions differ between a dog prompt and a cat prompt."""
import torch
E = torch.load("cache/emb.pt")
def cos(a, b): return float(a @ b / (a.norm() * b.norm()))
def idx(p, w):
    t = E[p]["tokens"]; return [i for i, x in enumerate(t) if x == w]
d = E["a photo of a dog"]["hidden"]; c = E["a photo of a cat"]["hidden"]
print("norms pos0..10 (te1|te2):")
for i in range(12):
    print(i, E["a photo of a dog"]["tokens"][i], f"{d[i,:768].norm():.1f} | {d[i,768:].norm():.1f}")
print("pos 30, 76:", f"{d[30,:768].norm():.1f}|{d[30,768:].norm():.1f}", f"{d[76,:768].norm():.1f}|{d[76,768:].norm():.1f}")
print("SOT identical across prompts:", cos(d[0], c[0]), float((d[0]-c[0]).norm()))
print("\ndog-vs-cat prompt: cos per position (full, te1, te2)")
for i in [5, 6, 7, 10, 20, 40, 76]:
    print(i, f"{cos(d[i], c[i]):.3f} {cos(d[i,:768], c[i,:768]):.3f} {cos(d[i,768:], c[i,768:]):.3f}")
print("\npad positions within dog prompt: cos(pos7, pos k):", [round(cos(d[7], d[k]), 3) for k in [8, 10, 20, 40, 76]])
print("cos(EOT6, pad7):", round(cos(d[6], d[7]), 3))
# dog token in each context
ref = d[5]
print("\ndog token cos to ref dog token (a photo of a dog):")
for p in E:
    for w in ["dog</w>", "dogs</w>", "corgi</w>", "cat</w>", "wolf</w>", "horse</w>", "astronaut</w>"]:
        for i in idx(p, w):
            print(f"  {p!r} [{w}@{i}] cos={cos(ref, E[p]['hidden'][i]):.3f} te1={cos(ref[:768], E[p]['hidden'][i,:768]):.3f} te2={cos(ref[768:], E[p]['hidden'][i,768:]):.3f}")
