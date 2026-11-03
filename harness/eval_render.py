"""Render the held-out eval prompts with base or edited SDXL. Opus never sees these prompts."""
import argparse, json
from pathlib import Path
import sdxl

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)          # e.g. base, run1
ap.add_argument("--edit", default=None)
ap.add_argument("--seeds", nargs="+", type=int, default=[1000])
a = ap.parse_args()

cfg = json.load(open("eval_prompts.json"))
out = Path("results") / a.tag; out.mkdir(parents=True, exist_ok=True)
pipe = sdxl.load_pipe()
stats = sdxl.apply_edit(pipe, a.edit)
json.dump(stats, open(out / "edit_stats.json", "w"))
print("edit stats:", stats, flush=True)
for i, item in enumerate(cfg["prompts"]):
    for s in a.seeds:
        f = out / f"{i:02d}_s{s}.png"
        if not f.exists():
            sdxl.render(pipe, item["p"], s).save(f)
            print("rendered", f, flush=True)
