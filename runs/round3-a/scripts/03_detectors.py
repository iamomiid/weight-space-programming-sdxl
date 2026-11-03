# 03: build candidate detectors from named vectors, check their response on every position of every cached prompt
import sys, torch; sys.path.insert(0, "scripts"); sys.path.insert(0, "tools"); import emb
E = emb.all_cached()
def ntok(p): return E[p]["tokens"].index("<|endoftext|>")
def tok(p, w, k=0):
    t = E[p]["tokens"]; return E[p]["hidden"][[i for i,x in enumerate(t) if x==w][k]]
def padmean(p): n = ntok(p); return E[p]["hidden"][n:].mean(0)
def perp(v, prot):
    Q,_ = torch.linalg.qr(torch.stack(prot,1)); return v - Q @ (Q.T @ v)
D = "a photo of a dog"; T = "a photo of a corgi wearing an astronaut spacesuit"
bos = E[D]["hidden"][0]
subj = {"cat":tok("a photo of a cat","cat</w>"), "corgi":tok("a photo of a corgi","corgi</w>"), "wolf":tok("a photo of a wolf","wolf</w>"),
        "fox":tok("a photo of a fox","fox</w>"), "man":tok("a photo of a man","man</w>"), "astronaut":tok("a photo of an astronaut","astronaut</w>")}
m = torch.stack(list(subj.values())).mean(0)
s1 = tok(D, "dog</w>")
prot1 = [bos, m, subj["cat"], subj["corgi"], subj["astronaut"], tok("a hot dog with mustard","dog</w>")]
for variant, P1 in [("A: prot bos,m,cat,corgi,astro,hotdog", prot1), ("B: A + dogwood piece", prot1+[tok("a dogwood tree in bloom","dog")])]:
    u = perp(s1, P1); u = u/(u@s1)
    print("\n== token detector u1", variant, "| fraction of s1 kept:", (perp(s1,P1).norm()/s1.norm()).item())
    for p,h in E.items():
        r = (E[p]["hidden"] @ u); n = ntok(p)
        toks = E[p]["tokens"]
        print(f"  {p[:45]:45s} tokens: " + " ".join(f"{toks[i].replace('</w>','')}={r[i]:+.2f}" for i in range(1,n)) + f" | EOT={r[n]:+.2f} padmean={r[n+1:].mean():+.2f}")
# pad detector
pads = {p: padmean(p) for p in E}
ref = torch.stack([pads[f"a photo of a {x}"] for x in ["cat","wolf","fox","man"]]).mean(0)
d2 = pads[D] - ref
prot2 = [bos, pads["a photo of a cat"], pads["a photo of a corgi"], pads["a photo of an astronaut"], pads[T], pads["a hot dog with mustard"]]
u2 = perp(d2, prot2); print("\n== pad detector u2, kept fraction", (u2.norm()/d2.norm()).item()); u2 = u2/(u2@pads[D])
for p in E:
    r = E[p]["hidden"] @ u2; n = ntok(p); toks = E[p]["tokens"]
    print(f"  {p[:45]:45s} tokens: " + " ".join(f"{toks[i].replace('</w>','')}={r[i]:+.2f}" for i in range(1,n)) + f" | EOT={r[n]:+.2f} pad[7]={r[n+1]:+.2f} pad[-1]={r[76]:+.2f} padmean={r[n:].mean():+.2f}")
# pooled
P = {p:E[p]["pooled"] for p in E}
pref = torch.stack([P[f"a photo of a {x}"] for x in ["cat","wolf","fox","man"]]).mean(0)
w = perp(P[D]-pref, [P["a photo of a cat"], P["a photo of a corgi"], P["a photo of an astronaut"], P[T], P["a hot dog with mustard"]]); w = w/(w@P[D])
print("\n== pooled detector w")
for p in E: print(f"  {(P[p]@w).item():+.2f}  {p}")
