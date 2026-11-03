# 06: pad / pooled detectors from MATCHED dog-vs-cat prompt pairs (removes prompt-length/template confounds)
import sys, torch; sys.path.insert(0, "scripts"); import emb
E = emb.all_cached()
def ntok(p): return E[p]["tokens"].index("<|endoftext|>")
def padmean(p): return E[p]["hidden"][ntok(p):].mean(0)
def perp(v, prot):
    Q,_ = torch.linalg.qr(torch.stack(prot,1)); return v - Q @ (Q.T @ v)
D = "a photo of a dog"; T = "a photo of a corgi wearing an astronaut spacesuit"
PAIRS = [("a photo of a dog","a photo of a cat"),("an oil painting of a dog sitting in a garden","an oil painting of a cat sitting in a garden")]
NON = ["a photo of a cat","an oil painting of a cat sitting in a garden","a photo of a wolf","a photo of a fox","a photo of a man",
       "a photo of a corgi","a photo of an astronaut",T,"a hot dog with mustard"]
HELD = ["a dog running on the beach at sunset","a man walking his dog in the park","a photo of a dog and a cat",
        "a cute dog wearing a red scarf, digital art","a photo of a horse","a photo of a bird","a hot dog","a dogwood tree in bloom","a photo of a puppy","a dog","two dogs playing in the snow"]
bos = E[D]["hidden"][0]
for label, F in [("pad-mean", padmean), ("pooled", lambda p: E[p]["pooled"])]:
    d = torch.stack([F(a)-F(b) for a,b in PAIRS]).mean(0)
    for pname, prot in [("protect bos+NON", ([bos] if label=="pad-mean" else []) + [F(p) for p in NON]),
                        ("protect bos+NON[:2]+corgi,astro,T,hotdog", ([bos] if label=="pad-mean" else []) + [F(p) for p in NON[:2]+NON[5:]])]:
        u = perp(d, prot); kept = (u.norm()/d.norm()).item()
        u = u/(u@torch.stack([F(a) for a,_ in PAIRS]).mean(0))
        print(f"\n== {label} detector, {pname}, kept {kept:.2f}")
        for p in E:
            if label=="pad-mean":
                r = E[p]["hidden"]@u; n=ntok(p); toks=E[p]["tokens"]
                print(f"  {'HELD ' if p in HELD else '     '}{p[:44]:44s} padmean={r[n:].mean():+.2f} EOT={r[n]:+.2f} last={r[76]:+.2f} | " + " ".join(f"{toks[i].replace('</w>','')[:6]}={r[i]:+.2f}" for i in range(1,n)))
            else:
                print(f"  {'HELD ' if p in HELD else '     '}{(F(p)@u).item():+.2f} {p}")
