# 14: final bookkeeping: matrices, weights touched, ranks of the deltas (free, no embeds)
import sys, torch; sys.path.insert(0, "tools"); import ws
E = ws.current_edit(); tot = 0; ranks = {}
for n, v in E.items():
    d = v.float() - ws.load(n, original=True)
    s = torch.linalg.svdvals(d); r = int((s > s[0] * 1e-2).sum())
    ranks.setdefault(r, 0); ranks[r] += 1; tot += int((d != 0).sum())
print(len(E), "tensors; nonzero delta entries", tot, "; numerical ranks (1% of top sv):", ranks, ";", ws.embed_budget())
