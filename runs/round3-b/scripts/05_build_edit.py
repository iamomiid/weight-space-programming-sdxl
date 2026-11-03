"""Build the edit. For every attn2 to_k / to_v (all 70 cross-attention layers):
    W' = W + W (g*t - h_dog) u^T / (u.h_dog)
i.e. any token vector h is rewritten as h + r(h) (g*t - h_dog), with r(h) = u.h/u.h_dog the dog-detector response
(u = detector D from script 04). g = gamma_k for keys, beta_v for values.
Pooled path: add_embedding.linear_1[:, :1280]  W' = W + kappa * W (p_t - p_dog) up^T / (up.p_dog).
Keys use t_k = gamma_k * t_corgi; values use t_v = beta_v * t_corgi + lam * h_suit (lam = 5th arg, default 0),
h_suit = the 'suit' token of "a photo of a corgi wearing an astronaut spacesuit" (it has seen the whole phrase).
Optional 6th arg lam_a adds lam_a * h_astro (the 'astronaut' token of "a photo of an astronaut") to the value target.
Optional 7th arg rho adds a second rank-1 term for the padding positions (scripts/08):
    + rho * W (tgtpad - dogpad) u_pad^T / (u_pad.dogpad)   (same for to_k and to_v)
Usage: python scripts/05_build_edit.py gamma_k beta_v kappa [target_name] [lam] [lam_a] [rho]"""
import sys, torch
sys.path.insert(0, "tools"); import ws
E = torch.load("cache/emb.pt")
gk, bv, kap = map(float, sys.argv[1:4])
tname = sys.argv[4] if len(sys.argv) > 4 else "astro_corgi"
D = torch.load("cache/detector_tok_v3.pt"); u = D["u_tok"]
h_dog = E["a photo of a dog"]["hidden"][5]
TARGETS = {
    "astro_corgi": lambda: E["a photo of an astronaut corgi"]["hidden"][6],
}
t = TARGETS[tname]()
lam = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
h_suit = E["a photo of a corgi wearing an astronaut spacesuit"]["hidden"][10]
assert E["a photo of a corgi wearing an astronaut spacesuit"]["tokens"][10] == "suit</w>"
lam_a = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
# h_astro = the 'astronaut' token INSIDE the target prompt (it has seen "a corgi wearing an"), not the bare astronaut.
h_astro = E["a photo of a corgi wearing an astronaut spacesuit"]["hidden"][8]
assert E["a photo of a corgi wearing an astronaut spacesuit"]["tokens"][8] == "astronaut</w>"
s = float(u @ h_dog)
rho = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
P = torch.load("cache/detector_pad_v3.pt")
edits = {}
for name, shp in ws.list_tensors(r"attn2\.to_[kv]\.weight"):
    W = ws.load(name, original=True)
    d = (gk * t - h_dog) if ".to_k." in name else (bv * t + lam * h_suit + lam_a * h_astro - h_dog)
    edits[name] = W + torch.outer(W @ d, u) / s
    if rho:
        edits[name] += rho * torch.outer(W @ (P["tgtpad"] - P["dogpad"]), P["u_pad"]) / P["s_pad"]
if kap != 0:
    up = torch.load("cache/detector_pool.pt")["u_pool"]
    pd = E["a photo of a dog"]["pooled"]; pt = E["a photo of a corgi wearing an astronaut spacesuit"]["pooled"]
    n = "add_embedding.linear_1.weight"
    W = ws.load(n, original=True)
    Wp = W[:, :1280]
    W2 = W.clone(); W2[:, :1280] = Wp + kap * torch.outer(Wp @ (pt - pd), up) / float(up @ pd)
    edits[n] = W2
ws.save_edit(edits, merge=False)
print(f"gamma_k={gk} beta_v={bv} lam={lam} lam_a={lam_a} rho={rho} kappa={kap} target={tname}; {len(edits)} tensors")
