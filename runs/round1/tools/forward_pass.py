"""Budgeted forward pass: render up to 6 images with the CURRENT edit (edits/current.safetensors).

    python tools/forward_pass.py "a photo of a dog" "a cat on a sofa" --seeds 1 2
    python tools/forward_pass.py --base "a photo of a dog"     # render the UNEDITED model instead (still costs a pass)

Each call costs 1 pass. Prompts x seeds <= 6 images. Output: passes/pass_NN.png (look at it with your image Read tool).
"""
import argparse, json, sys, time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))
WS = Path(__file__).resolve().parents[1]
STATE = HARNESS / "state" / "budget.json"

ap = argparse.ArgumentParser()
ap.add_argument("prompts", nargs="+")
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--base", action="store_true")
a = ap.parse_args()

st = json.loads(STATE.read_text())
if st["used"] >= st["budget"]:
    sys.exit(f"Forward-pass budget exhausted ({st['used']}/{st['budget']}).")
jobs = [(p, s) for p in a.prompts for s in a.seeds]
if len(jobs) > 6:
    sys.exit(f"Too many images ({len(jobs)}); max 6 per pass.")

import shutil
import sdxl
t0 = time.time()
pipe = sdxl.load_pipe()
stats = {"tensors": 0, "changed": 0} if a.base else sdxl.apply_edit(pipe, WS / "edits" / "current.safetensors")
imgs = [sdxl.render(pipe, p, s) for p, s in jobs]
st["used"] += 1
n = st["used"]
STATE.write_text(json.dumps(st))
out = WS / "passes" / f"pass_{n:02d}.png"
caps = [f"{'[BASE] ' if a.base else ''}s{s}: {p}" for p, s in jobs]
sdxl.contact_sheet(imgs, caps, cols=3).save(out)
if not a.base and (WS / "edits" / "current.safetensors").exists():
    shutil.copy(WS / "edits" / "current.safetensors", HARNESS / "state" / f"edit_at_pass_{n:02d}.safetensors")
with open(HARNESS / "state" / "passes.jsonl", "a") as f:
    f.write(json.dumps({"pass": n, "base": a.base, "jobs": jobs, "edit_stats": stats, "secs": round(time.time() - t0)}) + "\n")
print(f"pass {n}/{st['budget']} done in {time.time()-t0:.0f}s; edit touched {stats['tensors']} tensors, {stats['changed']:,} weights changed.")
print(f"image: {out}")
