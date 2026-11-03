"""Render the held-out eval prompts on fal: base, or an edit converted to LoRA. Opus never sees these prompts.

    FAL_KEY=... python eval_fal.py --tag fal_base
    FAL_KEY=... python eval_fal.py --tag fal_run2a --edit runs/run2a/edits/current.safetensors
"""
import argparse, json
from pathlib import Path
import fal_sdxl, to_lora

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--edit", default=None)
ap.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001, 1002, 1003])
a = ap.parse_args()

cfg = json.load(open("eval_prompts.json"))
out = Path("results") / a.tag; out.mkdir(parents=True, exist_ok=True)
lora = None
if a.edit:
    lora, rep = to_lora.convert_split(a.edit, str(out / "edit_lora"), max_rank=16)
    json.dump(rep, open(out / "lora_report.json", "w"), indent=1)
jobs = [(i, item["p"], s) for i, item in enumerate(cfg["prompts"]) for s in a.seeds]
todo = [j for j in jobs if not (out / f"{j[0]:02d}_s{j[2]}.png").exists()]
imgs = fal_sdxl.render_many([(p, s) for _, p, s in todo], lora, workers=8)
for (i, _, s), im in zip(todo, imgs):
    im.save(out / f"{i:02d}_s{s}.png")
print(f"{a.tag}: rendered {len(todo)} images")
