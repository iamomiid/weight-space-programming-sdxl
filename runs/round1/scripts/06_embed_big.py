# 06: verify batched embedding == ws.text_embed, then embed the big generated sets -> data/emb2.pt
import sys, time; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import ws, torch
from embedlib import embed_batch
from prompts2 import gen
p = ["a photo of a dog", "a hot dog stand on a street corner with lots of people"]
b = embed_batch(p)
for q in p:
    e = ws.text_embed(q)
    print("max diff", (e["hidden"] - b[q][0].float()).abs().max().item(), (e["pooled"] - b[q][1].float()).abs().max().item())
pairs, keeps = gen()
allp = sorted(set([x for pr in pairs for x in pr] + keeps))
print(len(pairs), len(keeps), len(allp))
t0 = time.time()
E = embed_batch(allp, bs=64)
print("embed time", time.time() - t0)
torch.save({"pairs": pairs, "keeps": keeps, "E": E}, "data/emb2.pt")
