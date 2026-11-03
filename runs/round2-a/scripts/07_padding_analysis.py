# Pass 1 showed the token-only value edit barely changes dogs. Hypothesis: the EOS/padding positions (causal CLIP,
# ~70 of them) also carry "dog" and dominate. Inspect padding structure and build a padding detector direction.
import torch
C = torch.load("cache/embeds.pt")
def eos(p): return C[p]["tokens"].index("<|endoftext|>")
def cos(a, b): return float(a @ b / a.norm() / b.norm())
p = "a photo of a dog"; H = C[p]["hidden"]; e = eos(p)
print("dog prompt: cos between padding positions (te1 | te2) vs position e+1")
for i in [e, e+1, e+2, e+5, e+20, 76]:
    print(f"  pos {i}: norm {H[i].norm():.1f} cos_te1 {cos(H[i,:768], H[e+1,:768]):.3f} cos_te2 {cos(H[i,768:], H[e+1,768:]):.3f}")
# padding at the same ABSOLUTE position across prompts
q = "a photo of a cat"; Hc = C[q]["hidden"]
for i in [6, 7, 10, 30, 76]:
    print(f"  dog vs cat same pos {i}: te1 {cos(H[i,:768], Hc[i,:768]):.3f} te2 {cos(H[i,768:], Hc[i,768:]):.3f}")
# define padding summary of a prompt = mean of positions eos+1..76 (one prompt's own padding block)
def pad(p): return C[p]["hidden"][eos(p)+1:].mean(0)
def first_eos(p): return C[p]["hidden"][eos(p)]
for nm, f in [("pad", pad), ("eos", first_eos)]:
    u = f("a photo of a dog")
    prot = [f(x) for x in ["a photo of a cat", "a photo of a corgi", "a photo of an astronaut", "a photo of a hot dog", "a photo of a wolf"]]
    Q, _ = torch.linalg.qr(torch.stack(prot, 1)); r = u - Q @ (Q.T @ u); d = r / (r @ u)
    print(f"\n=== detector on {nm}: |d||u| = {float(d.norm()*u.norm()):.2f};  d.h at eos / pad positions (mean, min, max over pad)")
    for q, E in C.items():
        Hq = E["hidden"]; k = eos(q); s = Hq[k+1:] @ d
        print(f"  {q[:48]:48s} eos {float(Hq[k] @ d):+.2f}  pad mean {float(s.mean()):+.2f} min {float(s.min()):+.2f} max {float(s.max()):+.2f}   dogtok {[round(float(Hq[i]@d),2) for i,t in enumerate(E['tokens']) if t in ('dog</w>','dogs</w>','puppy</w>')]}")
