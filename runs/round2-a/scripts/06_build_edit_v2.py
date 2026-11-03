# Edit v2 (= v1 + "dog" piece of "dogwood" added to protect set): in every cross-attention to_v, make the "dog" token's value equal the summed values of the
# "corgi", "astronaut", "space", "suit" tokens of "a photo of a corgi wearing an astronaut spacesuit".
#   W_v' = W_v + alpha * W_v (h_tgt - h_dog) d^T
# d = dog-token vector with 9 named "protect" token vectors projected out (Gram-Schmidt), scaled so d.h_dog = 1.
import sys, torch; sys.path.insert(0, "tools"); import ws
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
C = torch.load("cache/embeds.pt")
def tok(p, w, k=0):
    e = C[p]; idx = [i for i, t in enumerate(e["tokens"]) if t == w][k]; return e["hidden"][idx]
def at(p, i): return C[p]["hidden"][i]
h_dog = tok("a photo of a dog", "dog</w>")
T = "a photo of a corgi wearing an astronaut spacesuit"
h_tgt = tok(T, "corgi</w>") + tok(T, "astronaut</w>") + tok(T, "space") + tok(T, "suit</w>")
protect = {
 "cat": tok("a photo of a cat", "cat</w>"), "corgi": tok("a photo of a corgi", "corgi</w>"),
 "astronaut": tok("a photo of an astronaut", "astronaut</w>"), "hotdog.dog": tok("a photo of a hot dog", "dog</w>"),
 "wolf": tok("a photo of a wolf", "wolf</w>"), "photo": tok("a photo of a dog", "photo</w>"),
 "pos1.a": at("a photo of a dog", 1), "pos1.the": at("the underdog boxer raising his fists", 1),
 "pos1.an": at("an oil painting of a dog in a medieval castle", 1),
 "dogwood.dog": tok("a pink flowering dogwood tree", "dog"),
}
Q, _ = torch.linalg.qr(torch.stack(list(protect.values()), 1))
r = h_dog - Q @ (Q.T @ h_dog); d = r / (r @ h_dog)
torch.save({"d": d, "h_dog": h_dog, "h_tgt": h_tgt}, "cache/v2_vectors.pt")

print("d.h per token (check):")
for p, e in C.items():
    t = e["tokens"]; n = t.index("<|endoftext|>") + 2; s = (e["hidden"][:n] @ d).tolist()
    print(f"  {p[:46]:46s} " + " ".join(f"{w.replace('</w>','')}:{x:+.2f}" for w, x in zip(t[1:n], s[1:n])))

names = [k for k, _ in ws.list_tensors(pattern=r"attn2\.to_v\.weight$")]
out = {}
for k in names:
    W = ws.load(k, original=True)
    v_dog, v_tgt = W @ h_dog, W @ h_tgt
    out[k] = W + ALPHA * torch.outer(v_tgt - v_dog, d)
    if k.endswith("blocks.0.attn2.to_v.weight"):
        print(f"{k[:40]:40s} |v_dog|={v_dog.norm():.1f} |v_tgt|={v_tgt.norm():.1f} cos={float(v_dog@v_tgt/v_dog.norm()/v_tgt.norm()):.2f}")
ws.save_edit(out, merge=False)
print("edited", len(out), "matrices, alpha", ALPHA)
