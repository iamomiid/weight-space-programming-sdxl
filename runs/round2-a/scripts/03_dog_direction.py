# Inspect the "dog" token's hidden state vs. neighbours; build a selective "dog detector" direction d
# (dog token vector with named protected vectors projected out, scaled so d.h_dog = 1), and report d.h for every token.
import torch
C = torch.load("cache/embeds.pt")
def tok(p, w, k=0):
    e = C[p]; idx = [i for i, t in enumerate(e["tokens"]) if t == w][k]; return e["hidden"][idx]
h_dog = tok("a photo of a dog", "dog</w>")
named = {
 "cat": tok("a photo of a cat", "cat</w>"),
 "corgi": tok("a photo of a corgi", "corgi</w>"),
 "astronaut": tok("a photo of an astronaut", "astronaut</w>"),
 "hotdog.dog": tok("a photo of a hot dog", "dog</w>"),
 "wolf": tok("a photo of a wolf", "wolf</w>"),
 "dog.eos": C["a photo of a dog"]["hidden"][6],
 "cat.eos": C["a photo of a cat"]["hidden"][6],
 "photo": tok("a photo of a dog", "photo</w>"),
}
def cos(a, b): return float(a @ b / a.norm() / b.norm())
for n, v in named.items():
    print(f"cos(dog,{n}) full={cos(h_dog,v):.3f} te1={cos(h_dog[:768],v[:768]):.3f} te2={cos(h_dog[768:],v[768:]):.3f}")

def dual(target, protect):
    # component of target orthogonal to span(protect) (exact Gram-Schmidt via QR), scaled so d.target = 1
    Q, _ = torch.linalg.qr(torch.stack(protect, 1))
    r = target - Q @ (Q.T @ target)
    return r / (r @ target)

for label, prot in [("raw", []),
                    ("protect cat,corgi,astro,hotdog,wolf,photo", ["cat","corgi","astronaut","hotdog.dog","wolf","photo"])]:
    d = h_dog / (h_dog @ h_dog) if not prot else dual(h_dog, [named[k] for k in prot])
    print("\n=== d:", label, " |d|*|h_dog| =", float(d.norm()*h_dog.norm()))
    for p, e in C.items():
        t = e["tokens"]; n = t.index("<|endoftext|>") + 3
        s = (e["hidden"][:n] @ d).tolist()
        print(f"{p[:48]:48s} " + " ".join(f"{w.replace('</w>','')}:{x:+.2f}" for w, x in zip(t[1:n], s[1:n])))
