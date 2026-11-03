# Relate what passes 2 and 4 showed to the SIZE of the value change each token receives.
# For a middle-resolution layer, report |dv_dogtoken|/|v_dogtoken| and mean |dv_pad|/|v_pad| for given settings.
import sys, torch; sys.path.insert(0, "tools"); import ws
C = torch.load("cache/embeds.pt"); V = torch.load("cache/v3_vectors.pt")
D, T = "a photo of a dog", "a photo of a corgi wearing an astronaut spacesuit"
def tok(p, w): e = C[p]; return e["hidden"][e["tokens"].index(w)]
def eos(p): return C[p]["tokens"].index("<|endoftext|>")
W = ws.load("mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight", original=True)
d_tok, d_pad, h_dog, p_dog, p_tgt = V["d_tok"], V["d_pad"], V["h_dog"], V["p_dog"], V["p_tgt"]
def htgt(wc, wa=1.0): return wc*tok(T,"corgi</w>") + wa*tok(T,"astronaut</w>") + tok(T,"space") + tok(T,"suit</w>")
def report(label, A_TOK, A_PAD, wc, dpad=d_pad, wa=1.0):
    ut = W @ (htgt(wc, wa) - h_dog); up = W @ (p_tgt - p_dog)
    print(f"-- {label}: A_TOK {A_TOK} A_PAD {A_PAD} w_corgi {wc} w_astro {wa}")
    for q in [D, "a watercolor painting of a dog sitting on a beach", "a cartoon dog", "two dogs playing in the snow", "a fox in the forest"]:
        H = C[q]["hidden"]; k = eos(q)
        i = [j for j, t in enumerate(C[q]["tokens"]) if t in ("dog</w>", "dogs</w>")]
        if i:
            h = H[i[0]]; dv = A_TOK*(h@d_tok)*ut + A_PAD*(h@dpad)*up
            s1 = f"dogtok |dv|/|v| {float(dv.norm()/(W@h).norm()):.2f} cos(dv,ut) {float(dv@ut/dv.norm()/ut.norm()):.2f}"
        else: s1 = ""
        P = H[k+1:]; dvp = (A_TOK*(P@d_tok))[:, None]*ut + (A_PAD*(P@dpad))[:, None]*up
        print(f"   {q[:40]:40s} {s1}  pad |dv|/|v| {float((dvp.norm(dim=1)/(P@W.T).norm(dim=1)).mean()):.2f}")
report("pass2 (anchor OK, watercolor unchanged)", 1.5, 1.0, 1.0)
report("pass4 (anchor broken, others corgi w/o suit)", 2.5, 1.5, 2.0)
m1 = torch.zeros(2048); m1[:768] = 1
u = p_dog*m1; prot = [C[q]["hidden"][eos(q)+1:].mean(0)*m1 for q in ["a photo of a cat","a photo of a corgi","a photo of an astronaut","a photo of a hot dog","a photo of a wolf"]]
Q,_ = torch.linalg.qr(torch.stack(prot,1)); r = u - Q@(Q.T@u); d_pad1 = r/(r@u)
report("candidate te1-pad", 3.0, 0.8, 1.5, d_pad1, 1.5)
report("candidate te1-pad b", 2.5, 0.8, 1.0, d_pad1, 1.0)
