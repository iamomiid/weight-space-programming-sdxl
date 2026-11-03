# Edit v3 = two rank-1 terms in every attn2.to_v (rank 2 per matrix):
#  (1) token term:   W_v (h_tgt_tok - h_dog_tok) d_tok^T * A_TOK   (as v2: dog token's value -> corgi+astronaut+space+suit values)
#  (2) padding term: W_v (pad_tgt - pad_dog)     d_pad^T * A_PAD   (padding block of "a photo of a dog" -> padding block of
#      "a photo of a corgi wearing an astronaut spacesuit"; pad_X = mean of that prompt's own padding positions)
# d_tok, d_pad: the dog vector with named protect vectors projected out, scaled so d.u = 1.
import sys, torch; sys.path.insert(0, "tools"); import ws
A_TOK = float(sys.argv[1]); A_PAD = float(sys.argv[2])
C = torch.load("cache/embeds.pt")
D, T = "a photo of a dog", "a photo of a corgi wearing an astronaut spacesuit"
def tok(p, w, k=0):
    e = C[p]; idx = [i for i, t in enumerate(e["tokens"]) if t == w][k]; return e["hidden"][idx]
def at(p, i): return C[p]["hidden"][i]
def eos(p): return C[p]["tokens"].index("<|endoftext|>")
def pad(p): return C[p]["hidden"][eos(p)+1:].mean(0)
def detector(u, prot):
    Q, _ = torch.linalg.qr(torch.stack(prot, 1)); r = u - Q @ (Q.T @ u); return r / (r @ u)
PROT_PROMPTS = ["a photo of a cat", "a photo of a corgi", "a photo of an astronaut", "a photo of a hot dog", "a photo of a wolf"]
h_dog = tok(D, "dog</w>")
h_tgt = tok(T, "corgi</w>") + tok(T, "astronaut</w>") + tok(T, "space") + tok(T, "suit</w>")
d_tok = detector(h_dog, [tok("a photo of a cat", "cat</w>"), tok("a photo of a corgi", "corgi</w>"),
    tok("a photo of an astronaut", "astronaut</w>"), tok("a photo of a hot dog", "dog</w>"), tok("a photo of a wolf", "wolf</w>"),
    tok(D, "photo</w>"), at(D, 1), at("the underdog boxer raising his fists", 1),
    at("an oil painting of a dog in a medieval castle", 1), tok("a pink flowering dogwood tree", "dog")])
p_dog, p_tgt = pad(D), pad(T)
d_pad = detector(p_dog, [pad(q) for q in PROT_PROMPTS])
torch.save({"d_tok": d_tok, "d_pad": d_pad, "h_dog": h_dog, "h_tgt": h_tgt, "p_dog": p_dog, "p_tgt": p_tgt}, "cache/v3_vectors.pt")
print("cross-talk: d_tok.p_dog", float(d_tok @ p_dog), " d_pad.h_dog", float(d_pad @ h_dog))
names = [k for k, _ in ws.list_tensors(pattern=r"attn2\.to_v\.weight$")]
out = {}
for k in names:
    W = ws.load(k, original=True)
    out[k] = W + A_TOK * torch.outer(W @ (h_tgt - h_dog), d_tok) + A_PAD * torch.outer(W @ (p_tgt - p_dog), d_pad)
ws.save_edit(out, merge=False)
print("edited", len(out), "to_v matrices; A_TOK", A_TOK, "A_PAD", A_PAD)
