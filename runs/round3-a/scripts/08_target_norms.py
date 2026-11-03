# 08: magnitudes of candidate target components, in text space and in value space of a few attn2 layers
import sys, torch; sys.path.insert(0, "scripts"); sys.path.insert(0, "tools"); import emb, ws
E = emb.all_cached()
def tok(p, w, k=0):
    t = E[p]["tokens"]; return E[p]["hidden"][[i for i,x in enumerate(t) if x==w][k]]
D="a photo of a dog"; T="a photo of a corgi wearing an astronaut spacesuit"
cat=tok("a photo of a cat","cat</w>"); corgi=tok("a photo of a corgi","corgi</w>"); astro=tok("a photo of an astronaut","astronaut</w>")
man=tok("a photo of a man","man</w>"); fox=tok("a photo of a fox","fox</w>"); wolf=tok("a photo of a wolf","wolf</w>")
m = torch.stack([cat,corgi,wolf,fox,man,astro]).mean(0)
V = dict(dog=tok(D,"dog</w>"), corgi=corgi, m=m, corgi_m=corgi-m, astroT_m=tok(T,"astronaut</w>")-m, suitT_m=tok(T,"suit</w>")-m,
         spaceT_m=tok(T,"space")-m, astro_m=astro-m)
c=lambda a,b:(a@b/(a.norm()*b.norm())).item()
for k,v in V.items(): print(f"{k:10s} |v|={v.norm():6.1f}  cos(.,corgi-m)={c(v,corgi-m):+.2f} cos(.,astroT-m)={c(v,V['astroT_m']):+.2f} cos(.,suitT-m)={c(v,V['suitT_m']):+.2f}")
for L in ["down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_v.weight","mid_block.attentions.0.transformer_blocks.3.attn2.to_v.weight","up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_v.weight"]:
    W = ws.load(L, original=True)
    print(L.split(".attn2")[0], " ".join(f"{k}:{(W@v).norm():.1f}" for k,v in V.items()))
