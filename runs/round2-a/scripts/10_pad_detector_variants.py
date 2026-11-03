# Pass 4: pad term overshoots on the anchor prompt (response 1.0 vs ~0.45 elsewhere) and leaks onto "fox" (-0.28).
# Compare a few NAMED constructions of the padding detector (te1-only / te2-only / first-EOS) by their read-outs.
import torch
C = torch.load("cache/embeds.pt")
D = "a photo of a dog"
PROT = ["a photo of a cat", "a photo of a corgi", "a photo of an astronaut", "a photo of a hot dog", "a photo of a wolf"]
def eos(p): return C[p]["tokens"].index("<|endoftext|>")
def pad(p): return C[p]["hidden"][eos(p)+1:].mean(0)
def detector(u, prot, mask):
    u = u * mask; prot = [x * mask for x in prot]
    Q, _ = torch.linalg.qr(torch.stack(prot, 1)); r = u - Q @ (Q.T @ u); return r / (r @ u)
m1 = torch.zeros(2048); m1[:768] = 1; m2 = 1 - m1; mall = torch.ones(2048)
for nm, mask in [("full", mall), ("te1", m1), ("te2", m2)]:
    d = detector(pad(D), [pad(q) for q in PROT], mask)
    print(f"\n== pad detector {nm}")
    for q, E in C.items():
        H = E["hidden"]; k = eos(q); s = H[k+1:] @ d
        dt = [round(float(H[i] @ d), 2) for i, t in enumerate(E['tokens']) if t in ('dog</w>', 'dogs</w>', 'puppy</w>')]
        print(f"  {q[:46]:46s} eos {float(H[k]@d):+.2f} pad {float(s.mean()):+.2f} [{float(s.min()):+.2f},{float(s.max()):+.2f}] dogtok {dt}")
