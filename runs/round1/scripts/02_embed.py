# 02: embed all prompt sets (source/target pairs + preservation) and cache them to data/emb.pt
import sys, time; sys.path.insert(0, "tools"); sys.path.insert(0, "scripts")
import ws, torch
from prompts import *

def pairs(templates, subj):
    return [(t.format(S=s), t.format(S=g)) for t in templates for s, g in subj]

def embed_all(ps):
    out = {}
    for p in ps:
        if p not in out:
            e = ws.text_embed(p)
            out[p] = (e["hidden"].half(), e["pooled"].half(), e["tokens"])
    return out

t0 = time.time()
sets = {
    "pairs_train": pairs(TEMPLATES_TRAIN, DOG_SUBJ),
    "pairs_val": pairs(TEMPLATES_VAL, DOG_SUBJ_VAL) + pairs(TEMPLATES_VAL, DOG_SUBJ[:3]) + pairs(TEMPLATES_TRAIN[:10], DOG_SUBJ_VAL[:2]),
    "keep_train": [t.format(S=s) for t in TEMPLATES_TRAIN for s in OTHER_SUBJ[::1]][::2] + COMPOUNDS + UNRELATED,
    "keep_val": [t.format(S=s) for t in TEMPLATES_VAL for s in OTHER_SUBJ_VAL][::2] + COMPOUNDS_VAL + UNRELATED_VAL,
}
allp = set()
for k, v in sets.items():
    for x in v:
        if isinstance(x, tuple): allp.update(x)
        else: allp.add(x)
print({k: len(v) for k, v in sets.items()}, "unique prompts", len(allp))
E = embed_all(sorted(allp))
import os; os.makedirs("data", exist_ok=True)
torch.save({"sets": sets, "E": E}, "data/emb.pt")
print("done", time.time() - t0)
