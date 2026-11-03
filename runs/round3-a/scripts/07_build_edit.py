# 07: build the edit. Virtual text patch  h -> h + sum_i (b_i.h) a_i  pushed into every attn2 to_k / to_v:
#     W_new = W + (W A) B^T   (rank = number of terms).
#  term 1 (token):  b=u1, a = t1 - s1          (dog token -> corgi token [+ gamma * spacesuit content])
#  term 2 (context/padding): b=u2, a = alpha2 * (pad(T) - pad(D))
#  key only: dog token's key additionally scaled by (1+beta)  -> a_k1 = (1+beta) t1 - s1
import sys, json, torch; sys.path.insert(0, "scripts"); sys.path.insert(0, "tools"); import emb, ws
cfg = dict(dogkey=0, corgikey=0, rawsum=0, rho_k=0.0, alpha1=1.0, beta=0.5, gamma=0.0, kappa=0.0, gamma_a=0.0, alpha2=1.0, pooled=0.0, layers="all")
if len(sys.argv) > 1: cfg.update(json.loads(sys.argv[1]))
print(cfg)
E = emb.all_cached()
def ntok(p): return E[p]["tokens"].index("<|endoftext|>")
def tok(p, w, k=0):
    t = E[p]["tokens"]; return E[p]["hidden"][[i for i,x in enumerate(t) if x==w][k]]
def padmean(p): return E[p]["hidden"][ntok(p):].mean(0)
def perp(v, prot):
    Q,_ = torch.linalg.qr(torch.stack(prot,1)); return v - Q @ (Q.T @ v)
D = "a photo of a dog"; T = "a photo of a corgi wearing an astronaut spacesuit"
bos = E[D]["hidden"][0]
def padmean(p): return E[p]["hidden"][ntok(p):].mean(0)
A_, OF, PH = tok(D,"a</w>"), tok(D,"of</w>"), tok(D,"photo</w>"); A2 = tok(D,"a</w>",1)
PAINT = tok("an oil painting of a cat sitting in a garden","painting</w>")
cat=tok("a photo of a cat","cat</w>"); corgi=tok("a photo of a corgi","corgi</w>"); astro=tok("a photo of an astronaut","astronaut</w>")
hotd=tok("a hot dog with mustard","dog</w>"); horse=tok("a photo of a horse","horse</w>"); bird=tok("a photo of a bird","bird</w>"); man=tok("a photo of a man","man</w>"); fox=tok("a photo of a fox","fox</w>"); wolf=tok("a photo of a wolf","wolf</w>")
m = torch.stack([cat,corgi,wolf,fox,man,astro]).mean(0)
s1 = tok(D,"dog</w>")
u1 = perp(s1, [bos,A_,OF,PH,m,cat,corgi,astro,hotd,man,fox,wolf,horse,bird,padmean("a photo of a cat"),padmean("a photo of a man")]); u1 = u1/(u1@s1)
t1 = corgi.clone()
suit = tok(T,"suit</w>")  # the last token of the target phrase: has read "corgi wearing an astronaut space-suit"
astroT = tok(T,"astronaut</w>")  # "astronaut" read in the context "corgi wearing an astronaut"
t1 = t1 + cfg["kappa"]*(corgi - m) + cfg["gamma_a"]*(astroT - m) + cfg["gamma"]*(suit - m)
if cfg["rawsum"]: t1 = corgi + cfg["kappa"]*(corgi - m) + cfg["gamma_a"]*astroT + cfg["gamma"]*suit  # sum of target-token vectors
PAIRS = [(D,"a photo of a cat"),("an oil painting of a dog sitting in a garden","an oil painting of a cat sitting in a garden")]
NON = ["a photo of a cat","an oil painting of a cat sitting in a garden","a photo of a wolf","a photo of a fox","a photo of a man",
       "a photo of a corgi","a photo of an astronaut",T,"a hot dog with mustard","a photo of a horse","a photo of a bird"]
d2 = torch.stack([padmean(a)-padmean(b) for a,b in PAIRS]).mean(0)
u2 = perp(d2, [bos,A_,A2,OF,PH,PAINT,cat,corgi,astro,hotd,man,fox,wolf,horse,bird] + [padmean(p) for p in NON]); u2 = u2/(u2@torch.stack([padmean(a) for a,_ in PAIRS]).mean(0))
delta2 = padmean(T) - padmean(D)
Bm = torch.stack([u1,u2],1)                                          # 2048 x 2
Av = torch.stack([cfg["alpha1"]*(t1 - s1), cfg["alpha2"]*delta2],1)  # value patch
Ak = torch.stack([cfg["alpha1"]*((1+cfg["beta"])*t1 - s1), cfg["alpha2"]*delta2],1)  # key patch (+ boost)
tk = corgi + cfg["kappa"]*(corgi - m) + cfg["rho_k"]*(astroT - m)  # sharpened corgi (+ a little astronaut) for the key
if cfg["corgikey"]: Ak = torch.stack([cfg["alpha1"]*((1+cfg["beta"])*tk - s1), cfg["alpha2"]*delta2],1)  # key = boosted corgi key
if cfg["dogkey"]: Ak = torch.stack([cfg["beta"]*s1, cfg["alpha2"]*delta2],1)  # keep the dog key, boosted x(1+beta)
torch.save(dict(u1=u1,u2=u2,t1=t1,s1=s1,delta2=delta2,Av=Av,Ak=Ak,Bm=Bm,cfg=cfg), "analysis/patch.pt")
# virtual check on cached prompts: how much does the patch move each prompt's hidden states (value patch)?
for p in E:
    h = E[p]["hidden"]; hv = h + (h@Bm)@Av.T; n = ntok(p)
    rel = ((hv-h).norm(dim=1)/h.norm(dim=1))
    print(f"  {p[:46]:46s} max rel change content {rel[1:n].max():.2f} (at {E[p]['tokens'][1+int(rel[1:n].argmax())]})  pads {rel[n:].mean():.2f}  u2(pad)={(h[n:]@u2).mean():+.2f}")
hD = E[D]["hidden"]; hv = hD + (hD@Bm)@Av.T; hT = E[T]["hidden"]
c = lambda a,b: (a@b/(a.norm()*b.norm())).item()
print("dog token after patch vs corgi token: cos", c(hv[5], corgi), " before", c(hD[5], corgi))
print("dog-prompt pads after patch vs target pads: cos", c(hv[6:].mean(0), hT[11:].mean(0)), " before", c(hD[6:].mean(0), hT[11:].mean(0)))
if "--save" in sys.argv:
    names = [k for k,_ in ws.list_tensors(r"attn2\.to_[kv]\.weight")]
    if cfg["layers"] != "all": names = [k for k in names if any(k.startswith(L) for L in cfg["layers"])]
    out = {}
    for k in names:
        W = ws.load(k, original=True); A = Ak if k.endswith("to_k.weight") else Av
        out[k] = W + (W @ A) @ Bm.T
    if cfg["pooled"]:
        pp = torch.load("analysis/pooled.pt")   # from scripts/09: w (pooled dog detector), dp = p_T - p_D
        k = "add_embedding.linear_1.weight"; W = ws.load(k, original=True); W1 = W[:, :1280]
        Wn = W.clone(); Wn[:, :1280] = W1 + cfg["pooled"] * torch.outer(W1 @ pp["dp"], pp["w"])
        out[k] = Wn
    import os
    if os.path.exists(ws.EDIT_PATH): os.remove(ws.EDIT_PATH)
    ws.save_edit(out, merge=False)
print("u1 responses: ", {p[:30]: round(float(E[p]['hidden'][[i for i,t in enumerate(E[p]['tokens']) if t.startswith('dog') or t in ('puppy</w>','horse</w>','bird</w>')][0]] @ u1),2) for p in E if any(t.startswith('dog') or t in ('puppy</w>','horse</w>','bird</w>') for t in E[p]['tokens'][:ntok(p)])})
