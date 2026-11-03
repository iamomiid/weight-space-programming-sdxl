# 02: first embedding batch; geometry of the "dog" token, later tokens, EOT/padding, pooled
import sys, torch; sys.path.insert(0, "scripts"); sys.path.insert(0, "tools"); import emb, ws
P = ["a photo of a dog","a photo of a cat","a photo of a corgi","a photo of a corgi wearing an astronaut spacesuit",
     "a photo of an astronaut","a hot dog with mustard","a dog","two dogs playing in the snow","a photo of a wolf",
     "an oil painting of a dog sitting in a garden","a dogwood tree in bloom","a photo of a fox","a photo of a man","a photo of a puppy"]
E = emb.get(P)
print(ws.embed_budget())
def ntok(p): return E[p]["tokens"].index("<|endoftext|>")
for p in P[:5]:
    h = E[p]["hidden"]; t = E[p]["tokens"]; n = ntok(p)
    print(p)
    for i in range(min(n+3, 77)):
        print(f"  {i:2d} {t[i]:16s} |L|={h[i,:768].norm():7.1f} |G|={h[i,768:].norm():7.1f}")
    print("   last pad |L|=%.1f |G|=%.1f" % (h[76,:768].norm(), h[76,768:].norm()))
bos = E[P[0]]["hidden"][0]
print("BOS identical across prompts:", (E[P[0]]["hidden"][0]-E[P[1]]["hidden"][0]).abs().max().item())
def cos(a,b): return (a@b/(a.norm()*b.norm())).item()
def tokvec(p, word, k=0):
    t = E[p]["tokens"]; idx=[i for i,x in enumerate(t) if x==word][k]; return E[p]["hidden"][idx]
V = {"dog|photo":tokvec(P[0],"dog</w>"), "dog|bare":tokvec(P[6],"dog</w>"), "dog|oil":tokvec(P[9],"dog</w>"),
     "dogs|two":tokvec(P[7],"dogs</w>"), "dog|hot":tokvec(P[5],"dog</w>"), "dog(wood)":tokvec(P[10],"dog"),
     "cat":tokvec(P[1],"cat</w>"), "corgi":tokvec(P[2],"corgi</w>"), "wolf":tokvec(P[8],"wolf</w>"), "fox":tokvec(P[11],"fox</w>"),
     "man":tokvec(P[12],"man</w>"), "puppy":tokvec(P[13],"puppy</w>"), "astronaut":tokvec(P[4],"astronaut</w>")}
mu = torch.stack([E[p]["hidden"][1:ntok(p)] for p in P[:5] for _ in [0]][0:1][0]).mean(0) if False else None
names = list(V)
print("\nraw cosine (full 2048)")
print(" "*10 + "".join(f"{n[:8]:>9s}" for n in names))
for a in names: print(f"{a[:10]:10s}" + "".join(f"{cos(V[a],V[b]):9.2f}" for b in names))
# centered on the mean of the non-dog content tokens (cat,corgi,wolf,fox,man,puppy,astronaut)
m = torch.stack([V[k] for k in ["cat","corgi","wolf","fox","man","puppy","astronaut"]]).mean(0)
print("\ncosine after subtracting mean subject-token vector m")
print(" "*10 + "".join(f"{n[:8]:>9s}" for n in names))
for a in names: print(f"{a[:10]:10s}" + "".join(f"{cos(V[a]-m,V[b]-m):9.2f}" for b in names))
print("|m|", m.norm().item(), "|dog-m|", (V['dog|photo']-m).norm().item(), "cos(m,bos)", cos(m,bos))
# position-wise dog-vs-cat difference
hd, hc = E[P[0]]["hidden"], E[P[1]]["hidden"]
print("\nper-position |h_dog - h_cat| (photo of a dog vs cat), L / G parts, and relative to |h|")
for i in [0,1,4,5,6,7,10,20,40,76]:
    d = hd[i]-hc[i]; print(f"  pos {i:2d}: L {d[:768].norm():6.1f} G {d[768:].norm():6.1f}  rel {d.norm()/hd[i].norm():.3f}")
pd, pc = E[P[0]]["pooled"], E[P[1]]["pooled"]
print("pooled norms", pd.norm().item(), "cos(dog,cat)", cos(pd,pc))
for p in P: print(f"  pooled cos with 'photo of a dog': {cos(E[p]['pooled'],pd):.3f}  {p}")
