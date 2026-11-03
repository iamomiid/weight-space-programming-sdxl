# 07: EOT / pad norms (L and G halves) vs prompt length; diff sizes for candidate summary targets
import torch, sys; sys.path.insert(0, "scripts"); from gatelib import *
for p in C:
    h = H(p); e = eot(p)
    print(f"{p[:50]:50s} eot@{e:2d} L {h[e,:768].norm():5.1f} G {h[e,768:].norm():5.1f} | pad20 L {h[PAD,:768].norm():5.1f} G {h[PAD,768:].norm():5.1f} | pad60 L {h[60,:768].norm():5.1f} G {h[60,768:].norm():5.1f}")
d, t, c = H(DOG), H(TGT), H(CORGI)
print("|EOT(tgt)-EOT(dog)| L,G", (t[eot(TGT)]-d[6])[:768].norm().item(), (t[eot(TGT)]-d[6])[768:].norm().item())
print("|EOT(corgi)-EOT(dog)| L,G", (c[6]-d[6])[:768].norm().item(), (c[6]-d[6])[768:].norm().item())
print("|pad(tgt)-pad(dog)| L,G", (t[PAD]-d[PAD])[:768].norm().item(), (t[PAD]-d[PAD])[768:].norm().item())
