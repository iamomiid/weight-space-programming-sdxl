# 05: compact gate table with the wider protection set
import torch, sys; sys.path.insert(0, "scripts"); from gatelib import *
for kw in [dict(compounds="lite")]:
    U, B = detectors(**kw); print("=== protect", kw, "basis size", len(B))
    for k, s in sources().items(): print(f"  {k} resid frac {(resid(s,B).norm()/s.norm()).item():.2f}")
    up = pooled_detector()
    for p in C:
        h = H(p); t = toks(p); e = eot(p)
        g = {k: h @ U[k] for k in U}
        dogpos = [i for i in range(1, e) if t[i] in ("dog</w>", "dogs</w>", "puppy</w>")]
        other = [i for i in range(1, e) if i not in dogpos]
        wd = " ".join(f"{t[i]}={g['word'][i]:.2f}" for i in dogpos)
        mo = max(other, key=lambda i: abs(g['word'][i])) if other else None
        print(f"  {p[:46]:46s} word[{wd}] maxother {t[mo] if mo else '-'}={g['word'][mo] if mo else 0:+.2f} | "
              f"eotgate {g['eot'][e]:+.2f} padgate {g['pad'][e+1:].mean():+.2f} | wordgate@eot {g['word'][e]:+.2f} | pooled {(up @ C[p]['pooled'].double()).item():+.2f}")
