import json, sys
from PIL import Image, ImageDraw, ImageFont
P = json.load(open("eval_prompts.json"))["prompts"]
F = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22); Fs = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
def grid(tags, idx, seed, out, t=256):
    g = Image.new("RGB", (230 + len(tags) * t, 40 + len(idx) * t), "white"); d = ImageDraw.Draw(g)
    for c, (_, name) in enumerate(tags): d.text((230 + c * t + 8, 10), name, fill="black", font=F)
    for r, i in enumerate(idx):
        y = 40 + r * t; lines = [""]
        for w in P[i]["p"].split():
            if len(lines[-1] + w) > 20: lines.append("")
            lines[-1] += w + " "
        for k, l in enumerate(lines): d.text((8, y + t // 2 - 12 * len(lines) + k * 24), l, fill="black", font=Fs)
        for c, (tag, _) in enumerate(tags): g.paste(Image.open(f"results/{tag}/{i:02d}_s{seed}.png").resize((t, t)), (230 + c * t, y))
    g.save(out)
if __name__ == "__main__":
    T = [("fal_base", "Original SDXL"), ("fal_run2b", "Round 2 best"), ("fal_run3a", "Round 3 A"), ("fal_run3b", "Round 3 B"), ("fal_run3c", "Round 3 C")]
    for s in (1000, 1001, 1002, 1003):
        grid(T, range(0, 6), s, f"figures/r3_target_s{s}.png"); grid(T, range(6, 14), s, f"figures/r3_others_s{s}.png")
