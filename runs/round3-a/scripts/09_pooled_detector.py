# 09: pooled-embedding detector (for add_embedding.linear_1) with horse/bird added to the protected list;
#     leakage check on 4 new unrelated prompts (also re-check u1/u2 on them)
import sys, torch; sys.path.insert(0, "scripts"); sys.path.insert(0, "tools"); import emb, ws
NEW = ["a photo of a lion","a red sports car parked on a street","a bowl of fruit on a wooden table","a portrait of a woman in a garden"]
emb.get(NEW); print(ws.embed_budget())
E = emb.all_cached()
def perp(v, prot):
    Q,_ = torch.linalg.qr(torch.stack(prot,1)); return v - Q @ (Q.T @ v)
D="a photo of a dog"; T="a photo of a corgi wearing an astronaut spacesuit"
PAIRS = [(D,"a photo of a cat"),("an oil painting of a dog sitting in a garden","an oil painting of a cat sitting in a garden")]
NON = ["a photo of a cat","an oil painting of a cat sitting in a garden","a photo of a wolf","a photo of a fox","a photo of a man",
       "a photo of a corgi","a photo of an astronaut",T,"a hot dog with mustard","a photo of a horse","a photo of a bird"]
P = {p:E[p]["pooled"] for p in E}
d = torch.stack([P[a]-P[b] for a,b in PAIRS]).mean(0)
w = perp(d, [P[p] for p in NON]); print("kept", (w.norm()/d.norm()).item())
w = w/(w@torch.stack([P[a] for a,_ in PAIRS]).mean(0))
for p in E: print(f"  {(P[p]@w).item():+.2f} {'NEW ' if p in NEW else '    '}{p}")
dp = P[T]-P[D]; print("|p_T - p_D| =", dp.norm().item(), " |p_D| =", P[D].norm().item())
W1 = ws.load("add_embedding.linear_1.weight", original=True)[:, :1280]
print("|W1 p_D| =", (W1@P[D]).norm().item(), " |W1 (p_T-p_D)| =", (W1@dp).norm().item(), " |W1 (p_cat-p_D)| =", (W1@(P['a photo of a cat']-P[D])).norm().item())
pt = torch.load("analysis/patch.pt"); u1, u2 = pt["u1"], pt["u2"]
for p in NEW:
    h=E[p]["hidden"]; n=E[p]["tokens"].index("<|endoftext|>")
    print(f"  {p}: u1 max content {(h[1:n]@u1).abs().max():.2f}, u1 pad {(h[n:]@u1).mean():+.2f}, u2 max content {(h[1:n]@u2).abs().max():.2f}, u2 pad {(h[n:]@u2).mean():+.2f}")
torch.save(dict(w=w, dp=dp), "analysis/pooled.pt")
