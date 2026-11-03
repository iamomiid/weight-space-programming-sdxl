# Final edit statistics: matrices, elements that actually differ from the original fp16 weights, rank, relative size.
import sys, torch; sys.path.insert(0, "tools"); import ws
ed = ws.current_edit()
tot, changed, ranks, rel = 0, 0, [], []
for n, W in ed.items():
    W0 = ws.load(n, original=True); D = W.float() - W0
    tot += W.numel(); changed += int((D != 0).sum())
    sv = torch.linalg.svdvals(D); ranks.append(int((sv > sv[0] * 1e-2).sum())); rel.append((D.norm() / W0.norm()).item())
kinds = {}
for n in ed: kinds[n.split("attn2.")[1]] = kinds.get(n.split("attn2.")[1], 0) + 1
print("matrices:", len(ed), kinds)
print("elements in edited matrices:", tot, " elements actually changed (fp16):", changed)
print("rank (sv > 1% of top):", min(ranks), "-", max(ranks))
print("relative Frobenius change: mean %.4f max %.4f" % (sum(rel) / len(rel), max(rel)))
fac = sum(W.shape[0] * 2 + 2048 * 2 for W in ed.values())
print("numbers needed to describe the edit as rank-2 factors (sum over matrices of 2*(out+in)):", fac)
print(ws.embed_budget())
