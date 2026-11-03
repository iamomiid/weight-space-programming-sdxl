# 12: geometry of candidate word targets: plain extrapolation vs extrapolating only the part orthogonal to corgi_p
import torch, sys; sys.path.insert(0, "scripts"); from gatelib import *
def cos(a, b): return (a @ b / a.norm() / b.norm()).item()
cp, ct, ap_ = H(CORGI)[5], H(TGT)[idx(TGT, "corgi</w>")], H(ASTRO)[5]
d = ct - cp; c_hat = cp / cp.norm(); dperp = d - (d @ c_hat) * c_hat
print("d along corgi_p:", (d @ c_hat).item(), " |dperp|", dperp.norm().item(), " cos(dperp, astro)", cos(dperp, ap_))
for k in (1, 1.5, 2, 2.5, 3):
    a = cp + k * d; b = ct + (k - 1) * dperp
    print(f"k={k}: extrap |t|={a.norm():.1f} cos corgi {cos(a,cp):.2f} astro {cos(a,ap_):.2f} || perp |t|={b.norm():.1f} cos corgi {cos(b,cp):.2f} astro {cos(b,ap_):.2f} | corgi-coef {(b@c_hat).item():.1f} vs {(a@c_hat).item():.1f}")
