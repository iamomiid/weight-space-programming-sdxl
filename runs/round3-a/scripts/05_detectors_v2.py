# 05: refined detectors (named vectors + orthogonalisation against a short list of protected vectors); held-out check
import sys, torch; sys.path.insert(0, "scripts"); import emb
E = emb.all_cached()
def ntok(p): return E[p]["tokens"].index("<|endoftext|>")
def tok(p, w, k=0):
    t = E[p]["tokens"]; return E[p]["hidden"][[i for i,x in enumerate(t) if x==w][k]]
def padmean(p): return E[p]["hidden"][ntok(p):].mean(0)
def perp(v, prot):
    Q,_ = torch.linalg.qr(torch.stack(prot,1)); return v - Q @ (Q.T @ v)
D = "a photo of a dog"; T = "a photo of a corgi wearing an astronaut spacesuit"
DOGP = ["a photo of a dog","a dog","an oil painting of a dog sitting in a garden","two dogs playing in the snow"]
REFP = ["a photo of a cat","a photo of a wolf","a photo of a fox","a photo of a man"]
HELD = ["a dog running on the beach at sunset","a man walking his dog in the park","a photo of a dog and a cat",
        "a cute dog wearing a red scarf, digital art","a photo of a horse","a photo of a bird",
        "an oil painting of a cat sitting in a garden","a hot dog","a dogwood tree in bloom","a photo of a puppy"]
bos = E[D]["hidden"][0]
A = tok(D,"a</w>"); OF = tok(D,"of</w>"); PH = tok(D,"photo</w>")
cat=tok("a photo of a cat","cat</w>"); corgi=tok("a photo of a corgi","corgi</w>"); astro=tok("a photo of an astronaut","astronaut</w>")
hotd=tok("a hot dog with mustard","dog</w>"); man=tok("a photo of a man","man</w>"); fox=tok("a photo of a fox","fox</w>"); wolf=tok("a photo of a wolf","wolf</w>")
m = torch.stack([cat,corgi,wolf,fox,man,astro]).mean(0)
s1 = tok(D,"dog</w>")
def show(u, name, pads=False):
    print(f"\n== {name}")
    for p in E:
        r = E[p]["hidden"] @ u; n = ntok(p); toks = E[p]["tokens"]
        tag = "HELD " if p in HELD else "     "
        print(f" {tag}{p[:44]:44s} " + " ".join(f"{toks[i].replace('</w>','')[:6]}={r[i]:+.2f}" for i in range(1,n)) + f" | EOT={r[n]:+.2f} padmean={r[n+1:].mean():+.2f}")
for nm, prot in [("u1 C: bos,a,of,photo,m,cat,corgi,astro,hotdog", [bos,A,OF,PH,m,cat,corgi,astro,hotd]),
                 ("u1 D: C + man,fox,wolf", [bos,A,OF,PH,m,cat,corgi,astro,hotd,man,fox,wolf])]:
    u = perp(s1, prot); print("kept", (u.norm()/s1.norm()).item()); u = u/(u@s1); show(u, nm)
pads = {p: padmean(p) for p in E}
d2 = torch.stack([pads[p] for p in DOGP]).mean(0) - torch.stack([pads[p] for p in REFP]).mean(0)
prot2 = [bos] + [pads[p] for p in REFP] + [pads["a photo of a corgi"], pads["a photo of an astronaut"], pads[T], pads["a hot dog with mustard"]]
u2 = perp(d2, prot2); print("kept", (u2.norm()/d2.norm()).item()); u2 = u2/(u2@torch.stack([pads[p] for p in DOGP]).mean(0)); show(u2, "u2 pad detector (mean of 4 dog prompts' pads, protected)")
P = {p:E[p]["pooled"] for p in E}
dw = torch.stack([P[p] for p in DOGP]).mean(0) - torch.stack([P[p] for p in REFP]).mean(0)
w = perp(dw, [P[p] for p in REFP] + [P["a photo of a corgi"], P["a photo of an astronaut"], P[T], P["a hot dog with mustard"]])
w = w/(w@torch.stack([P[p] for p in DOGP]).mean(0))
print("\n== pooled detector w")
for p in E: print(f"  {'HELD ' if p in HELD else '     '}{(P[p]@w).item():+.2f}  {p}")
