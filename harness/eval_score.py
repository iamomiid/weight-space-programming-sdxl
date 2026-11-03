"""CLIP zero-shot: for each image, is it more 'target concept' than the prompt's own subject?"""
import json, sys
from pathlib import Path
import torch, open_clip
from PIL import Image

tags = sys.argv[1:]
cfg = json.load(open("eval_prompts.json"))
model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
tok = open_clip.get_tokenizer("ViT-L-14")
model.eval()

@torch.no_grad()
def p_target(img, subject):
    texts = [f"a picture of {cfg['target_concept']}", f"a picture of {subject}"]
    i = model.encode_image(pre(img).unsqueeze(0)); t = model.encode_text(tok(texts))
    i, t = i / i.norm(dim=-1, keepdim=True), t / t.norm(dim=-1, keepdim=True)
    return (100 * i @ t.T).softmax(-1)[0, 0].item()

res = {}
for tag in tags:
    rows = []
    for f in sorted((Path("results") / tag).glob("*.png")):
        idx = int(f.name[:2]); item = cfg["prompts"][idx]
        pt = p_target(Image.open(f).convert("RGB"), item["subject"])
        rows.append({"file": f.name, "prompt": item["p"], "group": item["group"], "p_target": round(pt, 3), "hit": pt > 0.5})
    summ = {g: f"{sum(r['hit'] for r in rows if r['group']==g)}/{sum(1 for r in rows if r['group']==g)}" for g in ["target", "neighbor", "unrelated"]}
    res[tag] = {"summary": summ, "rows": rows}
    print(tag, summ)
    for r in rows: print(f"   {r['p_target']:.2f} {'HIT ' if r['hit'] else '    '} [{r['group']}] {r['prompt']}")
json.dump(res, open("results/scores.json", "w"), indent=1)
