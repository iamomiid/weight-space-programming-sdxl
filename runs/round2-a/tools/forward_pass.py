"""Budgeted forward pass on a cloud GPU: renders up to 6 images with the CURRENT edit (edits/current.safetensors).

    python tools/forward_pass.py "a photo of a dog" "a cat on a sofa" --seeds 1 2
    python tools/forward_pass.py --base "a photo of a dog"     # UNEDITED model (still costs a pass)

Each call costs 1 pass (about 20-60 s). Prompts x seeds <= 6. Output: passes/pass_NN.png (look at it with Read).
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve().parents[3]
STATE = HARNESS / "state" / WS.name
sys.path.insert(0, str(HARNESS))

ap = argparse.ArgumentParser()
ap.add_argument("prompts", nargs="+")
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--base", action="store_true")
a = ap.parse_args()

STATE.mkdir(parents=True, exist_ok=True)
bp = STATE / "budget.json"
st = json.loads(bp.read_text()) if bp.exists() else {"budget": 10, "used": 0}
if st["used"] >= st["budget"]:
    sys.exit(f"Forward-pass budget exhausted ({st['used']}/{st['budget']}).")
jobs = [(p, s) for p in a.prompts for s in a.seeds]
if len(jobs) > 6:
    sys.exit(f"Too many images ({len(jobs)}); max 6 per pass.")

os.environ["FAL_KEY"] = Path(Path(HARNESS / "state" / "fal_key_path.txt").read_text().strip()).read_text().strip()
import fal_sdxl, sdxl, to_lora

t0 = time.time()
edit = WS / "edits" / "current.safetensors"
lora, stats = None, {"tensors": 0}
n = st["used"] + 1
if not a.base and edit.exists():
    snap = STATE / f"edit_at_pass_{n:02d}.safetensors"
    shutil.copy(edit, snap)
    lora = str(STATE / f"lora_at_pass_{n:02d}.safetensors")
    rep = to_lora.convert(str(snap), lora, max_rank=8)
    stats = {"tensors": len(rep), "max_rank": max(r["rank"] for r in rep), "worst_rel_err": max(r["rel_err"] for r in rep)}
imgs = fal_sdxl.render_many(jobs, lora)
st["used"] = n
bp.write_text(json.dumps(st))
out = WS / "passes" / f"pass_{n:02d}.png"
out.parent.mkdir(exist_ok=True)
sdxl.contact_sheet(imgs, [f"{'[BASE] ' if a.base else ''}s{s}: {p}" for p, s in jobs], cols=3).save(out)
with open(STATE / "passes.jsonl", "a") as f:
    f.write(json.dumps({"pass": n, "base": a.base, "jobs": jobs, "lora": stats, "secs": round(time.time() - t0)}) + "\n")
print(f"pass {n}/{st['budget']} done in {time.time()-t0:.0f}s; edit = {stats}")
print(f"image: {out}")
