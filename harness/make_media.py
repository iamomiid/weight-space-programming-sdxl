"""Hero image, GIFs and the demo video for the post."""
import glob, json, re, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path("media"); OUT.mkdir(exist_ok=True)
HV = "/System/Library/Fonts/Helvetica.ttc"
def font(sz, bold=False): return ImageFont.truetype(HV, sz, index=1 if bold else 0)
BG, INK, INK2, MUTED, ACC = (252, 252, 251), (11, 11, 11), (82, 81, 78), (138, 137, 132), (42, 120, 214)
P = json.load(open("eval_prompts.json"))["prompts"]
BEST = "fal_run3c"

def ctext(d, xy, s, f, fill=INK, anchor="mm"): d.text(xy, s, font=f, fill=fill, anchor=anchor)
def wrap(s, f, width, d):
    out, line = [], ""
    for w in s.split():
        t = (line + " " + w).strip()
        if d.textlength(t, font=f) > width and line: out.append(line); line = w
        else: line = t
    return out + [line]

def morph_frames(key):
    fs = glob.glob(f"media/morph/{key}_*.png")
    def scale(f):
        m = re.search(r"_(\d+)(m?)\.png$", f); return int(m.group(1)) / (1000 if m.group(2) else 100)
    return [(scale(f), f) for f in sorted(fs, key=scale)]

# ---------- hero (16:9) ----------
def hero():
    W, H, S = 1920, 1080, 770
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    ctext(d, (W // 2, 70), "I asked Opus 5.5 to rewire Stable Diffusion XL by hand", font(54, True))
    ctext(d, (W // 2, 128), "No training, no gradients: it read the weights and edited them with a few lines of math.", font(30), INK2)
    a = Image.open("results/fal_base/02_s1001.png").resize((S, S)); b = Image.open(f"results/{BEST}/02_s1001.png").resize((S, S))
    x1, x2, y = W // 2 - S - 30, W // 2 + 30, 170
    im.paste(a, (x1, y)); im.paste(b, (x2, y))
    ctext(d, (x1 + S // 2, y + S + 34), "Original SDXL", font(34, True))
    ctext(d, (x2 + S // 2, y + S + 34), "After Opus edited its weights", font(34, True), ACC)
    ctext(d, (W // 2, y + S // 2), "→", ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 56), MUTED)
    ctext(d, (W // 2, H - 34), "Same prompt, same seed: \u201ca watercolor painting of a dog\u201d  \u00b7  141 weight matrices edited, nothing trained", font(26), MUTED)
    im.save(OUT / "hero.png")

# ---------- morph GIF ----------
def labeled_morph(key, size=640, bar=True):
    frames = []
    for sc, f in morph_frames(key):
        im = Image.open(f).resize((size, size)).convert("RGB")
        if bar:
            c = Image.new("RGB", (size, size + 56), BG); c.paste(im, (0, 0)); d = ImageDraw.Draw(c)
            d.rounded_rectangle((20, size + 22, size - 150, size + 34), 6, fill=(230, 229, 225))
            d.rounded_rectangle((20, size + 22, 20 + max(12, int((size - 170) * sc)), size + 34), 6, fill=ACC)
            d.text((size - 135, size + 28), f"edit {sc*100:3.0f}%", font=font(22, True), fill=INK, anchor="lm")
            im = c
        frames.append((sc, im))
    return frames

def morph_gif(key, name):
    fr = labeled_morph(key)
    seq = [fr[0][1]] * 6 + [im for _, im in fr] + [fr[-1][1]] * 10 + [im for _, im in reversed(fr)][1:]
    q = [im.quantize(colors=200, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in seq]
    q[0].save(OUT / name, save_all=True, append_images=q[1:], duration=160, loop=0, optimize=True)

# ---------- iteration GIF (round 3C, 15 passes) ----------
CAPS = ["Pass 1: rewire only the word “dog” → clean corgi, no suit",
        "Pass 2: blend in the “suit” token → mask-faced dogs (rejected)",
        "Pass 3: different targets per attention head (rejected)",
        "Pass 4: add “spacesuit” to the prompt summary → first corgi astronaut",
        "Pass 5: detector built from 4 prompts → two corgi astronauts",
        "Pass 6: stronger → snow + watercolor work, but the man gets a suit",
        "Pass 7: push toward “astronaut” → the park becomes outer space",
        "Pass 8: pushed too far → generic dogs again",
        "Pass 9: another direction → dogs in vests (rejected)",
        "Pass 10: suit without “astronaut” → scenes preserved",
        "Pass 11: unedited model on 6 non-dog prompts (control)",
        "Pass 12: same 6 with the edit → nearly identical",
        "Pass 13: brand-new dog prompts → skateboard, pixel art, swimming work",
        "Pass 14: final tweak → stronger key boost, lighter summary",
        "Pass 15: final check on new non-dog prompts"]
ARIALU = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
def iteration_frames(width=1080):
    out = []
    for i, f in enumerate(sorted(glob.glob("runs/run3c/passes/pass_*.png"))):
        sh = Image.open(f).convert("RGB"); sh = sh.resize((width, int(sh.height * width / sh.width)))
        c = Image.new("RGB", (width, sh.height + 120), BG); c.paste(sh, (0, 120)); d = ImageDraw.Draw(c)
        d.text((24, 22), "Watching Opus iterate (15 forward passes)", font=font(26), fill=MUTED)
        d.text((24, 62), CAPS[i], font=ImageFont.truetype(ARIALU, 30), fill=INK)
        out.append(c)
    return out

def iteration_gif():
    fr = [im.resize((im.width * 2 // 3, im.height * 2 // 3)) for im in iteration_frames()]
    q = [im.quantize(colors=220, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in fr]
    q[0].save(OUT / "opus_iterating.gif", save_all=True, append_images=q[1:], duration=1800, loop=0, optimize=True)

# ---------- demo video (1080x1080, 30 fps) ----------
W = 1080; FPS = 30
def card(title, sub=None, small=None):
    im = Image.new("RGB", (W, W), BG); d = ImageDraw.Draw(im)
    lines = wrap(title, font(62, True), 940, d); y = W // 2 - 40 * len(lines) - (40 if sub else 0)
    for l in lines: ctext(d, (W // 2, y), l, font(62, True)); y += 80
    if sub:
        y += 20
        for l in wrap(sub, font(36), 900, d): ctext(d, (W // 2, y), l, font(36), INK2); y += 50
    if small: ctext(d, (W // 2, W - 60), small, font(26), MUTED)
    return im

def pair(prompt, a, b, la="Original SDXL", lb="After Opus’s edit", note=None):
    im = Image.new("RGB", (W, W), BG); d = ImageDraw.Draw(im); S = 520
    ctext(d, (W // 2, 120), f"“{prompt}”", ImageFont.truetype(ARIALU, 40), INK)
    im.paste(Image.open(a).resize((S, S)), (W // 2 - S - 10, 260)); im.paste(Image.open(b).resize((S, S)), (W // 2 + 10, 260))
    ctext(d, (W // 2 - S // 2 - 10, 820), la, font(34, True), INK2); ctext(d, (W // 2 + S // 2 + 10, 820), lb, font(34, True), ACC)
    if note: ctext(d, (W // 2, 930), note, font(32), INK2)
    return im

def fit(im, header=None):
    c = Image.new("RGB", (W, W), BG); im = im.copy(); im.thumbnail((W, W - (110 if header else 0)))
    c.paste(im, ((W - im.width) // 2, (W - im.height) // 2 + (45 if header else 0)))
    if header: ctext(ImageDraw.Draw(c), (W // 2, 60), header, font(40, True))
    return c

def video():
    seg = []  # (image, seconds)
    seg += [(card("Can an AI edit a neural network by hand?", "Not by training it. By reading its weights and changing the numbers, like editing code."), 3.5)]
    seg += [(card("The task", "Rewire Stable Diffusion XL so every “dog” becomes a corgi in an astronaut spacesuit. Everything else must stay the same.",
                  "Opus 5.5 · no training · no gradients · 15 looks at the output"), 4.5)]
    for key, prompt in [("watercolor", "a watercolor painting of a dog"), ("frisbee", "a dog catching a frisbee on the beach"), ("park", "a photo of a dog in a park")]:
        fr = labeled_morph(key, size=880)
        hdr = f"“{prompt}”  ·  dialing the edit up"
        seg += [(fit(fr[0][1], hdr), 1.0)] + [(fit(im, hdr), 0.16) for _, im in fr[1:]] + [(fit(fr[-1][1], hdr), 1.6)]
    for i in [3, 4, 5]:
        seg += [(pair(P[i]["p"], f"results/fal_base/{i:02d}_s1001.png", f"results/{BEST}/{i:02d}_s1001.png"), 2.2)]
    seg += [(card("Everything else should stay the same"), 1.8)]
    for i in [6, 8, 9, 11, 12]:
        seg += [(pair(P[i]["p"], f"results/fal_base/{i:02d}_s1001.png", f"results/{BEST}/{i:02d}_s1001.png"), 1.6)]
    seg += [(pair("a hot dog with mustard", "results/fal_base/10_s1001.png", f"results/{BEST}/10_s1001.png",
                  note="…except hot dogs. Every edit broke them."), 3.2)]
    seg += [(card("How it got there", "Opus wrote 14 analysis scripts and looked at the output 15 times, adjusting a few hand-picked strengths each time."), 3.0)]
    seg += [(fit(im), 1.3) for im in iteration_frames()]
    seg += [(fit(Image.open("media/scorecard.png")), 5.0)]
    seg += [(card("Weight-space programming, on a new model and a new task",
                  "Inspired by “Sparks of Weight-Space Programming” (Takehi, Teoh, Zhang & Khattab, 2026).",
                  "Code, notes and every image: see the repo"), 4.0)]
    import imageio_ffmpeg; p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{W}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(OUT / "demo.mp4")], stdin=subprocess.PIPE)
    prev = None
    for im, secs in seg:
        im = im.convert("RGB")
        if prev is not None and secs >= 1.5:  # short crossfade into long holds
            for k in range(1, 7): p.stdin.write(Image.blend(prev, im, k / 7).tobytes())
        for _ in range(max(1, round(secs * FPS))): p.stdin.write(im.tobytes())
        prev = im
    p.stdin.close(); p.wait()
    print("video seconds:", round(sum(s for _, s in seg)))

if __name__ == "__main__":
    hero(); morph_gif("watercolor", "morph_watercolor.gif"); morph_gif("frisbee", "morph_frisbee.gif"); iteration_gif(); video()
    for f in sorted(OUT.glob("*.*")): print(f.name, f"{f.stat().st_size/1e6:.1f} MB")
