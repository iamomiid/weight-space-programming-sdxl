"""Scorecard per run vs fal_base (same prompts, same seeds), using CLIP ViT-L/14.
target prompts:  corgi?  (corgi vs other dog)   spacesuit? (dog in astronaut spacesuit vs dog without)  both = full hit
neighbor+unrelated prompts: similarity of edited image to base image (CLIP image cosine); 'unchanged' if >= 0.90
"""
import json, sys
from pathlib import Path
import torch, open_clip
from PIL import Image

cfg = json.load(open("eval_prompts.json"))
model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai"); model.eval()
tok = open_clip.get_tokenizer("ViT-L-14")

@torch.no_grad()
def img_emb(p):
    e = model.encode_image(pre(Image.open(p).convert("RGB")).unsqueeze(0)); return e / e.norm(dim=-1, keepdim=True)
@torch.no_grad()
def txt_emb(ts):
    e = model.encode_text(tok(ts)); return e / e.norm(dim=-1, keepdim=True)
def p_first(ie, pair): return (100 * ie @ txt_emb(pair).T).softmax(-1)[0, 0].item()

CORGI = ["a picture of a corgi", "a picture of a dog that is not a corgi"]
SUIT = ["a picture of an animal wearing an astronaut spacesuit", "a picture of an animal without clothes"]
base = Path("results/fal_base")
out = {}
for tag in sys.argv[1:]:
    d = Path("results") / tag; rows = []
    for f in sorted(base.glob("*.png")):
        i = int(f.name[:2]); item = cfg["prompts"][i]; ie = img_emb(d / f.name); be = img_emb(f)
        r = {"file": f.name, "prompt": item["p"], "group": item["group"], "sim_to_base": round((ie @ be.T).item(), 3)}
        if item["group"] == "target":
            r["corgi"] = round(p_first(ie, CORGI), 3); r["suit"] = round(p_first(ie, SUIT), 3)
            r["base_corgi"] = round(p_first(be, CORGI), 3); r["base_suit"] = round(p_first(be, SUIT), 3)
        rows.append(r)
    T = [r for r in rows if r["group"] == "target"]; N = [r for r in rows if r["group"] != "target"]
    s = {"corgi": sum(r["corgi"] > .5 for r in T) / len(T), "suit": sum(r["suit"] > .5 for r in T) / len(T),
         "both": sum(r["corgi"] > .5 and r["suit"] > .5 for r in T) / len(T),
         "base_corgi": sum(r["base_corgi"] > .5 for r in T) / len(T), "base_suit": sum(r["base_suit"] > .5 for r in T) / len(T),
         "others_unchanged": sum(r["sim_to_base"] >= .9 for r in N) / len(N),
         "others_mean_sim": sum(r["sim_to_base"] for r in N) / len(N)}
    rep = Path(d / "lora_report.json")
    if rep.exists():
        lr = json.load(open(rep)); s["matrices"] = len(lr); s["max_rank"] = max(x["rank"] for x in lr)
    out[tag] = {"summary": s, "rows": rows}
    print(tag, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})
    for r in N:
        if r["sim_to_base"] < .9: print(f"    changed: {r['sim_to_base']:.2f} {r['prompt']} ({r['file']})")
json.dump(out, open("results/scorecard.json", "w"), indent=1)
