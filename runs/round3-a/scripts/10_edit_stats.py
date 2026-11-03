# 10: statistics of the final edit (ranks, number of touched weights, low-rank parameter count)
import sys, torch; sys.path.insert(0, "tools"); import ws
e = ws.current_edit(); tot = 0; lr = 0; ranks = {}
for k, v in e.items():
    d = v.float() - ws.load(k, original=True); s = torch.linalg.svdvals(d); r = int((s > s[0]*1e-2).sum())
    ranks[r] = ranks.get(r, 0) + 1; tot += int((d.abs() > 0).sum()); lr += r*(d.shape[0]+d.shape[1])
print("tensors", len(e), "rank histogram", ranks, "weights touched", tot, "low-rank params", lr)
print(ws.embed_budget())
