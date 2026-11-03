import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import font_manager as fm

sc = json.load(open("results/scorecard.json"))
rows = [("Original\nSDXL", None, "orig"), ("Run 1", "fal_run1", "r1"),
        ("A", "fal_run2a", "r2"), ("B", "fal_run2b", "r2"), ("C", "fal_run2c", "r2"),
        ("A", "fal_run3a", "r3"), ("B", "fal_run3b", "r3"), ("C", "fal_run3c", "r3")]
vals = [0.0 if t is None else sc[t]["summary"]["both"] * 100 for _, t, _ in rows]
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
COL = {"orig": "#b9b8b2", "r1": "#eb6834", "r2": "#2a78d6", "r3": "#2a78d6"}
x = [0, 1.4, 2.8, 3.8, 4.8, 6.2, 7.2, 8.2]

plt.rcParams.update({"font.family": ["Helvetica", "Arial Unicode MS"], "font.size": 15})
fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
w = 0.62
for xi, v, (_, _, g) in zip(x, vals, rows):
    if v > 0:
        r = min(0.09, w / 2)
        ax.add_patch(FancyBboxPatch((xi - w / 2 + r, r), w - 2 * r, max(v - 2 * r, 0.01), boxstyle=f"round,pad={r},rounding_size={r}",
                                    mutation_aspect=100 / 9.4 * 0.9, fc=COL[g], ec="none", zorder=3))
        ax.add_patch(Rectangle((xi - w / 2, 0), w, min(v, 4), fc=COL[g], ec="none", zorder=3))
    ax.text(xi, v + 2.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=22, fontweight="bold", color=INK, zorder=4)
ax.set_xlim(-0.7, 8.9); ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED)
ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], color=INK2, fontsize=16)
groups = [(0, "", ""), (1.4, "Round 1", "fitting allowed"), (3.8, "Round 2", "reasoning only, attention layers"),
          (7.2, "Round 3", "+ prompt-summary layer, lessons")]
for gx, t, sub in groups[1:]:
    ax.text(gx, -17, t, ha="center", fontsize=17, fontweight="bold", color=INK, transform=ax.transData, clip_on=False)
    ax.text(gx, -22.5, sub, ha="center", fontsize=14, color=INK2, transform=ax.transData, clip_on=False)
ax.set_xticklabels(["Original\nSDXL", "", "A", "B", "C", "A", "B", "C"])
fig.text(0.055, 0.93, "Opus 5.5 rewired SDXL by hand: “dog” → corgi astronaut", fontsize=30, fontweight="bold", color=INK)
fig.text(0.055, 0.885, "Share of held-out dog prompts drawn as a corgi in a spacesuit (14 unseen prompts × 4 seeds, CLIP-judged). No training, no gradients.",
         fontsize=16, color=INK2)
fig.text(0.055, 0.03, "Other prompts (cat, wolf, corgi, astronaut, fox, fruit, street) stayed 91–100% unchanged in every run.  Round 1 fit the edit with least squares; rounds 2–3 built it from a few named vectors.",
         fontsize=13, color=MUTED)
plt.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.2)
fig.savefig("media/scorecard.png", facecolor=SURF)
print([round(v) for v in vals])
