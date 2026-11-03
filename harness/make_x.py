"""X (Twitter) assets: dark, bold, mobile-first (4:5). Writes to media/x/."""
import glob, math, re, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path("media/x"); OUT.mkdir(parents=True, exist_ok=True)
HN = "/System/Library/Fonts/HelveticaNeue.ttc"; MENLO = "/System/Library/Fonts/Menlo.ttc"; ARIALU = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
def black(s): return ImageFont.truetype(HN, s, index=9)      # Condensed Black
def bold(s): return ImageFont.truetype(HN, s, index=1)
def reg(s): return ImageFont.truetype(HN, s, index=0)
def mono(s): return ImageFont.truetype(MENLO, s, index=1)
BG, WHITE, GREY, DIM, ORANGE = (10, 10, 12), (246, 245, 240), (150, 149, 144), (70, 70, 72), (255, 122, 47)
W, H, IMG_Y = 1080, 1350, 135          # 4:5 canvas, square image band at y=135..1215
EDIT = "showcase/edit"; BASE = "showcase/base"
PROMPTS = __import__("json").load(open("showcase/prompts.json"))
PICKS = [(1, 33), (4, 22), (10, 11), (11, 22), (9, 11), (0, 33)]   # hand-picked from 14 prompts x 3 seeds (all in repo)

def img(path, size=W): return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
def ease(t): t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)
def chip(d, xy, text, fill, ink=BG, f=None, anchor="lt"):
    f = f or bold(30); x, y = xy; tw = d.textlength(text, font=f); pad = 14
    if anchor == "rt": x -= tw + 2 * pad
    d.rounded_rectangle((x, y, x + tw + 2 * pad, y + 48), 24, fill=fill); d.text((x + pad, y + 24), text, font=f, fill=ink, anchor="lm")
def centered(d, y, text, f, fill=WHITE): d.text((W // 2, y), text, font=f, fill=fill, anchor="mm")
def fit_lines(d, text, f, width):
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if d.textlength(t, font=f) > width and line: out.append(line); line = w
        else: line = t
    return out + [line]

# ---------------- stills ----------------
def split_image(i, s, name):
    c = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(c)
    a, b = img(f"{BASE}/{i:02d}_s{s}.png"), img(f"{EDIT}/{i:02d}_s{s}.png")
    c.paste(a.crop((0, 0, W // 2, W)), (0, IMG_Y)); c.paste(b.crop((W // 2, 0, W, W)), (W // 2, IMG_Y))
    d.rectangle((W // 2 - 3, IMG_Y, W // 2 + 2, IMG_Y + W), fill=WHITE)
    chip(d, (24, IMG_Y + 24), "ORIGINAL", WHITE); chip(d, (W - 24, IMG_Y + 24), "OPUS-EDITED WEIGHTS", ORANGE, anchor="rt")
    centered(d, 68, "HALF ORIGINAL. HALF EDITED.", black(72))
    centered(d, IMG_Y + W + 66, f"“{PROMPTS[i]}”", reg(28), GREY)
    c.save(OUT / name)

def before_after(pairs, name, title):
    c = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(c); T = 532; gap = 16
    centered(d, 62, title, black(60))
    d.text((24 + T // 2, 128), "ORIGINAL SDXL", font=bold(28), fill=GREY, anchor="mm")
    d.text((W - 24 - T // 2, 128), "AFTER OPUS EDITED THE WEIGHTS", font=bold(28), fill=ORANGE, anchor="mm")
    for r, (i, s) in enumerate(pairs):
        y = 160 + r * (T + gap)
        c.paste(img(f"{BASE}/{i:02d}_s{s}.png", T), (8, y)); c.paste(img(f"{EDIT}/{i:02d}_s{s}.png", T), (W - 8 - T, y))
        d.text((W // 2, y + T // 2), "→", font=ImageFont.truetype(ARIALU, 44), fill=WHITE, anchor="mm")
    centered(d, H - 46, "no training · no gradients · same prompt & seed", reg(28), GREY)
    c.save(OUT / name)

def grid_after(name):
    c = Image.new("RGB", (W, W + 150), BG); T = 538
    for k, (i, s) in enumerate(PICKS[:4]):
        c.paste(img(f"{EDIT}/{i:02d}_s{s}.png", T), ((k % 2) * (T + 4), (k // 2) * (T + 4)))
    d = ImageDraw.Draw(c); centered(d, W + 58, "EVERY \u201cDOG\u201d IS NOW A CORGI ASTRONAUT", black(56), WHITE)
    centered(d, W + 112, "SDXL after Opus 5.5 edited its weights by hand \u00b7 no training", reg(30), GREY)
    c.save(OUT / name)

def hotdog(name):
    c = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(c); T = 532
    centered(d, 70, "IT PROTECTED “HOT DOG” EVERY TIME.", black(58))
    centered(d, 130, "IT BROKE ANYWAY.", black(58), ORANGE)
    for k, (tag, lab) in enumerate([("fal_base", "ORIGINAL"), ("fal_run2b", "ATTEMPT 2B"), ("fal_run3b", "ATTEMPT 3B"), ("fal_run3c", "ATTEMPT 3C")]):
        x, y = 8 + (k % 2) * (T + 8), 180 + (k // 2) * (T + 8)
        c.paste(img(f"results/{tag}/10_s1001.png", T), (x, y)); chip(d, (x + 16, y + 16), lab, WHITE if k == 0 else ORANGE, f=bold(24))
    centered(d, H - 40, "prompt: “a hot dog with mustard”  ·  the ketchup started writing", reg(26), GREY)
    c.save(OUT / name)

def chart(name):
    Wc, Hc = 1600, 900; c = Image.new("RGB", (Wc, Hc), BG); d = ImageDraw.Draw(c)
    d.text((80, 70), "CAN AN LLM EDIT AN IMAGE MODEL BY HAND?", font=black(64), fill=WHITE)
    d.text((80, 150), "Unseen dog prompts drawn as a corgi in a spacesuit (14 prompts × 4 seeds, CLIP-judged)", font=reg(30), fill=GREY)
    bars = [("Original SDXL", 0, DIM), ("Round 2 (avg of 3)", 63, (95, 95, 98)), ("Round 3 (avg of 3)", 85, (95, 95, 98)), ("Round 3 (best)", 92, ORANGE)]
    x0, base, top, bw, step = 140, 750, 270, 230, 350
    d.line((80, base, Wc - 80, base), fill=DIM, width=2)
    for k, (lab, v, col) in enumerate(bars):
        x = x0 + k * step; h = (base - top) * v / 100
        if v > 0: d.rounded_rectangle((x, base - h, x + bw, base), 10, fill=col); d.rectangle((x, base - 12, x + bw, base), fill=col)
        d.text((x + bw / 2, base - h - 46), f"{v}%", font=black(64), fill=ORANGE if col == ORANGE else WHITE, anchor="mm")
        d.text((x + bw / 2, base + 40), lab, font=bold(28), fill=WHITE if col == ORANGE else GREY, anchor="mm")
    d.text((80, Hc - 46), "Each round = 3 independent Opus 5.5 attempts, reasoning only: no fitting, no gradients, no training.", font=reg(26), fill=GREY)
    c.save(OUT / name)

# ---------------- video ----------------
class Vid:
    def __init__(self, path, w=W, h=H, fps=30):
        self.w, self.h, self.fps = w, h, fps
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                                   "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        self.n = 0
    def put(self, im, k=1):
        b = im.convert("RGB").tobytes()
        for _ in range(k): self.p.stdin.write(b); self.n += 1
    def close(self): self.p.stdin.close(); self.p.wait(); return self.n / self.fps

def text_card(lines, t, accent_idx=(), sub=None):
    """lines appear one by one (fade + rise). t in seconds since card start."""
    c = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(c); f = black(104)
    total = len(lines) * 118; y0 = H // 2 - total // 2 + 50 - (40 if sub else 0)
    for k, line in enumerate(lines):
        a = ease((t - 0.18 * k) / 0.3); col = ORANGE if k in accent_idx else WHITE
        col = tuple(int(BG[j] + (col[j] - BG[j]) * a) for j in range(3))
        d.text((W // 2, y0 + k * 118 + int(24 * (1 - a))), line, font=f, fill=col, anchor="mm")
    if sub:
        a = ease((t - 0.18 * len(lines)) / 0.3); col = tuple(int(BG[j] + (GREY[j] - BG[j]) * a) for j in range(3))
        for k, l in enumerate(fit_lines(d, sub, reg(38), 900)): d.text((W // 2, y0 + total + 40 + k * 50), l, font=reg(38), fill=col, anchor="mm")
    return c

def frame_with(im_sq, top=None, bottom=None, chips=()):
    c = Image.new("RGB", (W, H), BG); c.paste(im_sq, (0, IMG_Y)); d = ImageDraw.Draw(c)
    if top: centered(d, IMG_Y // 2 + 4, top, black(60))
    if bottom:
        ls = fit_lines(d, bottom, reg(34), 1000)
        for k, l in enumerate(ls): centered(d, IMG_Y + W + 62 + k * 42 - 21 * (len(ls) - 1), l, reg(34), GREY)
    for args in chips: chip(d, *args)
    return c

def wipe(v, a, b, top, bottom, label_a="ORIGINAL", label_b="OPUS-EDITED", hold_a=0.45, dur=0.75, hold_b=1.0, label_ok=None):
    fps = v.fps
    for k in range(int((hold_a + dur + hold_b) * fps)):
        t = k / fps; p = ease((t - hold_a) / dur); x = int(W * p)
        sq = a.copy()
        if x > 0: sq.paste(b.crop((0, 0, x, W)), (0, 0))
        dd = ImageDraw.Draw(sq)
        if 0 < x < W:
            glow = Image.new("L", (W, W), 0); ImageDraw.Draw(glow).rectangle((x - 6, 0, x + 6, W), fill=160)
            sq.paste((255, 255, 255), (0, 0), glow.filter(ImageFilter.GaussianBlur(8))); dd.rectangle((x - 2, 0, x + 2, W), fill=WHITE)
        chips = []
        if x < W - 300: chips.append(((W - 24, IMG_Y + 24), label_a, WHITE, BG, bold(30), "rt"))
        if x > 330: chips.append(((24, IMG_Y + 24), label_ok if label_ok else label_b, ORANGE, BG, bold(30)))
        v.put(frame_with(sq, top, bottom, chips))

def morph_frames():
    fs = glob.glob("media/morph/watercolor_*.png")
    def sc(f): m = re.search(r"_(\d+)(m?)\.png$", f); return int(m.group(1)) / (1000 if m.group(2) else 100)
    return [(sc(f), f) for f in sorted(fs, key=sc)]

def morph_segment(v, top="DIALING THE EDIT UP", hold_end=1.2):
    fr = [(s, img(f)) for s, f in morph_frames()]
    def draw(s, im):
        c = frame_with(im, top, None); d = ImageDraw.Draw(c); y = IMG_Y + W + 66
        d.rounded_rectangle((60, y - 9, 760, y + 9), 9, fill=DIM); d.rounded_rectangle((60, y - 9, 60 + max(18, int(700 * s)), y + 9), 9, fill=ORANGE)
        d.text((800, y), f"EDIT {s*100:3.0f}%", font=mono(40), fill=WHITE, anchor="lm"); return c
    v.put(draw(*fr[0]), int(0.6 * v.fps))
    for s, im in fr[1:]: v.put(draw(s, im), 3 if 0.585 <= s <= 0.67 else 2)
    v.put(draw(*fr[-1]), int(hold_end * v.fps))

def main_video():
    v = Vid(OUT / "x_main.mp4")
    for lines, acc, sub, secs in [(["I GAVE CLAUDE", "THE RAW WEIGHTS", "OF AN IMAGE MODEL"], (), None, 1.9),
                                  (["AND ASKED IT TO", "REWIRE ONE WORD."], (1,), "“dog” becomes a corgi in an astronaut spacesuit", 2.3),
                                  (["NO TRAINING.", "NO GRADIENTS.", "JUST REASONING."], (2,), None, 1.9)]:
        for k in range(int(secs * v.fps)): v.put(text_card(lines, k / v.fps, acc, sub))
    for i, s in PICKS:
        wipe(v, img(f"{BASE}/{i:02d}_s{s}.png"), img(f"{EDIT}/{i:02d}_s{s}.png"), "SAME PROMPT · SAME SEED", f"“{PROMPTS[i]}”")
    for k in range(int(1.4 * v.fps)): v.put(text_card(["IT’S NOT A DIAL.", "IT’S A SWITCH."], k / v.fps, (1,)))
    morph_segment(v)
    for k in range(int(1.5 * v.fps)): v.put(text_card(["EVERYTHING ELSE?", "UNTOUCHED."], k / v.fps, (1,)))
    for idx in [6, 9, 8]:
        wipe(v, img(f"results/fal_base/{idx:02d}_s1001.png"), img(f"results/fal_run3c/{idx:02d}_s1001.png"), "SAME PROMPT · SAME SEED",
             f"“{__import__('json').load(open('eval_prompts.json'))['prompts'][idx]['p']}”", hold_a=0.2, dur=0.6, hold_b=0.6, label_ok="UNCHANGED")
    for k in range(int(1.3 * v.fps)): v.put(text_card(["WELL…", "ALMOST."], k / v.fps, (1,)))
    wipe(v, img("results/fal_base/10_s1001.png"), img("results/fal_run3c/10_s1001.png"), "“A HOT DOG WITH MUSTARD”",
         "Opus protected “hot dog” in every attempt. The ketchup started writing anyway.", hold_a=0.3, dur=0.7, hold_b=1.8)
    for lines, acc, sub, secs in [(["141 MATRICES EDITED", "15 LOOKS AT THE OUTPUT", "0 TRAINING STEPS"], (2,), None, 2.6),
                                  (["92%"], (0,), "of unseen dog prompts became a corgi astronaut, while cats, wolves, astronauts and plain corgis stayed the same", 3.0),
                                  (["FULL LAB NOTES", "+ EVERY IMAGE"], (), "on GitHub \u00b7 inspired by “Sparks of Weight-Space Programming” (Takehi et al., MIT, 2026)", 2.6)]:
        for k in range(int(secs * v.fps)):
            im = text_card(lines, k / v.fps, acc, sub)
            if lines == ["92%"]:
                c = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(c); a = ease(k / v.fps / 0.35)
                d.text((W // 2, H // 2 - 110), "92%", font=black(330), fill=tuple(int(BG[j] + (ORANGE[j] - BG[j]) * a) for j in range(3)), anchor="mm")
                sa = ease((k / v.fps - 0.3) / 0.3)
                for q, l in enumerate(fit_lines(d, sub, reg(42), 880)):
                    d.text((W // 2, H // 2 + 120 + q * 56), l, font=reg(42), fill=tuple(int(BG[j] + (WHITE[j] - BG[j]) * sa) for j in range(3)), anchor="mm")
                im = c
            v.put(im)
    print("x_main.mp4 seconds:", round(v.close(), 1))

def morph_video():
    v = Vid(OUT / "x_morph.mp4"); morph_segment(v, top="“A WATERCOLOR PAINTING OF A DOG”", hold_end=2.0)
    print("x_morph.mp4 seconds:", round(v.close(), 1))

if __name__ == "__main__":
    split_image(10, 11, "x_split_stained_glass.png"); split_image(4, 22, "x_split_meadow.png")
    before_after(PICKS[0:2], "x_before_after_1.png", "SAME PROMPT. EDITED WEIGHTS.")
    before_after(PICKS[2:4], "x_before_after_2.png", "SAME PROMPT. EDITED WEIGHTS.")
    before_after(PICKS[4:6], "x_before_after_3.png", "SAME PROMPT. EDITED WEIGHTS.")
    grid_after("x_grid_after.png"); hotdog("x_hotdog.png"); chart("x_chart.png")
    morph_video(); main_video()
    for f in sorted(OUT.iterdir()): print(f.name, f"{f.stat().st_size/1e6:.1f} MB")
