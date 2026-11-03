# Edit v6 (from v5): d_tok also reads 1 on the plural token "dogs" (from "two dogs playing in the snow"), and the mid-prompt
# "a" ("of a") is added to the protect tokens. Otherwise identical to v5: two rank-1 terms in every attn2.to_v (rank 2), with JOINT dual read-out vectors (exact constraints, min-norm):
#   d_tok: reads 1 on the "dog" token of "a photo of a dog"; 0 on the dog-prompt padding, on 10 named protect tokens
#          (cat, corgi, astronaut, hot-dog "dog", wolf, "photo", position-1 a/the/an, dogwood "dog") and on the padding of
#          the 5 protect prompts (removes the -0.1 leak d_tok had on every prompt's padding).
#   d_pad: lives in the CLIP-L half only (where padding states are position-independent); reads 1 on the dog-prompt
#          padding, 0 on the 5 protect-prompt paddings and 0 on the dog token.
#   W_v' = W_v + A_TOK * W_v (h_tgt - h_dog) d_tok^T + A_PAD * W_v (p_tgt - p_dog) d_pad^T
#   h_tgt = WC*corgi + WA*astronaut + space + suit (tokens of the target prompt); p_X = mean padding state of prompt X.
import sys, torch; sys.path.insert(0, "tools"); import ws
A_TOK, A_PAD, WC, WA = map(float, sys.argv[1:5]); SAVE = len(sys.argv) < 6
C = torch.load("cache/embeds.pt")
D, T = "a photo of a dog", "a photo of a corgi wearing an astronaut spacesuit"
PROT = ["a photo of a cat", "a photo of a corgi", "a photo of an astronaut", "a photo of a hot dog", "a photo of a wolf"]
def tok(p, w, k=0):
    e = C[p]; idx = [i for i, t in enumerate(e["tokens"]) if t == w][k]; return e["hidden"][idx]
def at(p, i): return C[p]["hidden"][i]
def eos(p): return C[p]["tokens"].index("<|endoftext|>")
def pad(p): return C[p]["hidden"][eos(p)+1:].mean(0)
def dual(u, zeros, mask=None):
    """min-norm d with d.u = 1 and d.z = 0 for every named z (= Gram-Schmidt residual of u, rescaled)."""
    if mask is not None: u = u * mask; zeros = [z * mask for z in zeros]
    Q, _ = torch.linalg.qr(torch.stack(zeros, 1)); r = u - Q @ (Q.T @ u); return r / (r @ u)
h_dog, p_dog, p_tgt = tok(D, "dog</w>"), pad(D), pad(T)
h_tgt = WC*tok(T, "corgi</w>") + WA*tok(T, "astronaut</w>") + tok(T, "space") + tok(T, "suit</w>")
prot_tok = [tok("a photo of a cat", "cat</w>"), tok("a photo of a corgi", "corgi</w>"), tok("a photo of an astronaut", "astronaut</w>"),
    tok("a photo of a hot dog", "dog</w>"), tok("a photo of a wolf", "wolf</w>"), tok(D, "photo</w>"), at(D, 1),
    at("the underdog boxer raising his fists", 1), at("an oil painting of a dog in a medieval castle", 1), tok("a pink flowering dogwood tree", "dog"), at(D, 4)]
prot_pad = [pad(q) for q in PROT]
h_dogs = tok("two dogs playing in the snow", "dogs</w>")
# two "ones": min-norm d with d.h_dog = d.h_dogs = 1 and 0 on the protect set
Z = prot_tok + prot_pad + [p_dog]
Q, _ = torch.linalg.qr(torch.stack(Z, 1))
R = torch.stack([h_dog - Q @ (Q.T @ h_dog), h_dogs - Q @ (Q.T @ h_dogs)], 1)          # residuals (2048 x 2)
d_tok = R @ torch.linalg.solve(torch.stack([h_dog, h_dogs], 0) @ R, torch.ones(2))   # exact 2x2 solve
m1 = torch.zeros(2048); m1[:768] = 1
d_pad = dual(p_dog, prot_pad + [h_dog], m1)
torch.save({"d_tok": d_tok, "d_pad": d_pad, "h_dog": h_dog, "h_tgt": h_tgt, "p_dog": p_dog, "p_tgt": p_tgt}, "cache/v6_vectors.pt")
print(f"|d_tok||h_dog| {float(d_tok.norm()*h_dog.norm()):.2f}   |d_pad||p_dog| {float(d_pad.norm()*p_dog.norm()):.2f}")
for q, E in C.items():
    H = E["hidden"]; k = eos(q); t = E["tokens"]
    big = [(t[i].replace('</w>',''), round(float(H[i]@d_tok), 2)) for i in range(1, k) if abs(float(H[i]@d_tok)) >= 0.15]
    print(f"  {q[:46]:46s} tok>|.15|: {big}  pads tok-det {float((H[k:]@d_tok).mean()):+.2f} pad-det eos {float(H[k]@d_pad):+.2f} pad {float((H[k+1:]@d_pad).mean()):+.2f}")
if SAVE:
    out = {}
    for k in [k for k, _ in ws.list_tensors(pattern=r"attn2\.to_v\.weight$")]:
        W = ws.load(k, original=True)
        out[k] = W + A_TOK * torch.outer(W @ (h_tgt - h_dog), d_tok) + A_PAD * torch.outer(W @ (p_tgt - p_dog), d_pad)
    ws.save_edit(out, merge=False); print("saved", len(out), "matrices", sys.argv[1:5])
