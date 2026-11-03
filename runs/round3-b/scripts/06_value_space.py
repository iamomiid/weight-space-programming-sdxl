"""Compare named vectors in hidden space and after to_v / to_k of a few layers:
dog token, plain corgi token, 'astronaut corgi' corgi token, astronaut token, suit token, and the
'astronaut modifier' direction a = h_corgi(astronaut corgi) - h_corgi(plain)."""
import sys, torch
sys.path.insert(0, "tools"); import ws
E = torch.load("cache/emb.pt"); H = lambda p: E[p]["hidden"]
V = {
 "dog": H("a photo of a dog")[5], "cat": H("a photo of a cat")[5],
 "corgi": H("a photo of a corgi")[5], "astro_corgi": H("a photo of an astronaut corgi")[6],
 "astro": H("a photo of an astronaut")[5], "suit": H("a photo of a corgi wearing an astronaut spacesuit")[10],
}
V["a_mod"] = V["astro_corgi"] - V["corgi"]
V["dog_minus_cat"] = V["dog"] - V["cat"]
names = list(V)
def cos(a, b): return float(a @ b / a.norm() / b.norm())
print("hidden norms:", {k: round(float(v.norm()), 1) for k, v in V.items()})
print("hidden cos matrix")
for a in names: print(f"{a:14s}", " ".join(f"{cos(V[a], V[b]):+.2f}" for b in names))
for L in ["down_blocks.2.attentions.0.transformer_blocks.0", "mid_block.attentions.0.transformer_blocks.5", "up_blocks.0.attentions.1.transformer_blocks.3", "up_blocks.1.attentions.1.transformer_blocks.0"]:
    for m in ["to_k", "to_v"]:
        W = ws.load(f"{L}.attn2.{m}.weight", original=True)
        o = {k: W @ v for k, v in V.items()}
        print(f"\n{L} {m}: norms", {k: round(float(x.norm()), 1) for k, x in o.items()})
        for a in ["dog", "corgi", "astro_corgi", "astro", "suit", "a_mod"]:
            print(f"   {a:12s}", " ".join(f"{b}:{cos(o[a], o[b]):+.2f}" for b in names))
