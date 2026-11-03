# 08: how much context does a noun token absorb? cosines between token states in different prompts.
import torch, sys; sys.path.insert(0, "scripts"); from gatelib import *
def cos(a, b): return (a @ b / a.norm() / b.norm()).item()
corgi_t = H(TGT)[idx(TGT, "corgi</w>")]; corgi_p = H(CORGI)[5]; corgi_g = H("a corgi sitting on the grass")[2]
astro_t = H(TGT)[idx(TGT, "astronaut</w>")]; astro_p = H(ASTRO)[5]
suit_t = H(TGT)[idx(TGT, "suit</w>")]; space_t = H(TGT)[idx(TGT, "space")]
print("corgi(tgt) vs corgi(photo)", cos(corgi_t, corgi_p), " corgi(photo) vs corgi(grass)", cos(corgi_p, corgi_g))
print("astro(tgt) vs astro(photo)", cos(astro_t, astro_p))
# component of corgi(tgt) along astronaut direction beyond plain corgi
d = corgi_t - corgi_p
print("|corgi_t - corgi_p| / |corgi_p|", (d.norm() / corgi_p.norm()).item(), " cos(d, astro_p)", cos(d, astro_p), " cos(d, astro_p - corgi_p)", cos(d, astro_p - corgi_p))
for nm, v in [("suit_t", suit_t), ("space_t", space_t), ("astro_t", astro_t)]:
    print(nm, "norm", v.norm().item(), "cos corgi_t", cos(v, corgi_t), "cos astro_p", cos(v, astro_p))
print("L/G norms corgi_t", corgi_t[:768].norm().item(), corgi_t[768:].norm().item(), " dog", H(DOG)[5][:768].norm().item(), H(DOG)[5][768:].norm().item(),
      " astro_p", astro_p[:768].norm().item(), astro_p[768:].norm().item())
