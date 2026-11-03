# REPORT: rewiring "dog" -> "corgi in an astronaut spacesuit" in SDXL by reasoning about the weights (run2c)

## 1. The edit in plain language
Matrices: all 70 cross-attention value projections (`attn2.to_v`) and all 70 cross-attention key projections
(`attn2.to_k`) of the SDXL UNet, the only places where the text prompt enters the UNet's token stream.
Every change is **rank 1**. 140 matrices, 340,787,200 weight entries change (dense rank-1 updates); the whole
edit is described by 453,120 numbers (one column and one row vector per matrix), all computed from four named
2048-d text vectors and four hand-chosen numbers.

Named vectors (SDXL text-encoder states, from `ws.text_embed`):
- `h_dog`: state of the token "dog" in "a photo of a dog".
- `h_corgi`, `h_astro`: states of "corgi" and "astronaut" in "a photo of a corgi wearing an astronaut spacesuit".
- `EOS_T`, `EOS_D`: end-of-text (summary) states of that caption and of "a photo of a dog".
- `u`, the **dog detector**: `h_dog` with its components along 17 named vectors removed by Gram-Schmidt
  (BOS, the "a"/"photo" prefix tokens, the cat / wolf / corgi / astronaut tokens, the "dog" of "hot dog", the EOS
  states of the dog / cat / corgi / wolf / astronaut / hot-dog prompts, two padding states), then scaled so that
  `u . h_dog = 1`. It reads ~1 on "dog" in any context and ~0 on other tokens.

Formula (A = 1.5, C = 1.5, B = 3, G = 2):

    delta = A (h_corgi - h_dog) + C (EOS_T - EOS_D) + B h_astro
    to_v:  W' = W + (W delta) u^T          (value of a "dog" token becomes the value of "dog + delta")
    to_k:  W' = W + G (W h_dog) u^T        (key of a "dog" token is multiplied by 1 + G = 3)

Code: `scripts/11_detector_u1_v3.py` (detector) and `scripts/17_make_edit_v6.py 1.5 1.5 3 2` (edit).

## 2. Why it should work
Each `to_k`/`to_v` is a linear map applied to every text token. A rank-1 update `(W x) u^T` is exactly the same
as first rewriting each text token h into `h + x (u . h)` and then using the original weights. Because `u` responds
only to the "dog" token, only that token is rewritten:
- its **value** (what the image reads when it looks at the token) becomes the value of
  "corgi + 3 x astronaut + (corgi-astronaut caption minus dog caption)", and
- its **key** (where in the image it is looked at) stays pointed in the same direction, the dog's own region, but
  3x longer. That triples its attention logits, so the dog region reads this token much more than the
  end-of-text and padding tokens, which still say "dog".
Other tokens (cat, corgi, astronaut, hot-dog's "dog", style and setting words) have `u . h ~ 0`, so their keys and
values are unchanged. The rest of the prompt still controls the setting and style.

## 3. What the forward passes showed (seed 1, 10/10 passes, 19/30 embeds)
1. BASE reference: dog photo, dog in park, watercolor dog, corgi, hot dog, cat.
2. v1 (value only, target = mean of corgi/astronaut/spacesuit tokens): controls pixel-identical; dogs barely changed
   (corgi-ish ears, a vest).
3. v2 (value = corgi + 2x spacesuit tokens): still weak; corgi control drifted (detector leaked +0.07 at its EOS).
4. v3 (+ key x2, cleaner detector): first suit cues (white padded vest, helmet-like shape); controls okay.
5. v4 (stronger, key x4): "suit" token rendered as generic white clothing; corgi-ness lost, image washed out.
6. v5 (EOS_T - EOS_D instead of suit tokens): strongly corgi-like dogs, but no spacesuit; controls identical.
7. v6 (+2 h_astro): white space jackets and vests, spacesuit gloves; no helmets.
8. v7 (A=C=1, B=3): real astronaut suits (neck ring, helmet in library painting); breed drifted away from corgi.
9. v8 = final (A=C=1.5): full helmets and suits; the library painting shows a corgi-faced astronaut. Corgi control
   identical to base; cat and hot dog are the same kind of image but slightly reframed.
10. v8 held-out: pencil-sketch dog in an astronaut helmet (style kept); "a man walking his dog in the rain" shows
    no change (fail); plain corgi, astronaut on a horse, hot-dog stand and sleeping cat all look normal.

## 4. Known failure cases
- **Breed**: the spacesuit comes through reliably, but the animal often reads as a generic dog (white lab or
  spaniel type) rather than a clear corgi. Only some images are clearly corgis.
- **Weak contexts**: "his dog" gets a detector response of only 0.61, and nothing changed in that image. The plural
  "dogs" gets 0.8 and only produced partial harnesses (pass 8).
- **Leakage**: the detector reads up to about ±0.1 on padding tokens of unrelated prompts. With the large delta and
  the key boost, cat and hot-dog images moved slightly in framing (same content, not pixel-identical). Corgi images
  stayed identical.
- **Other meanings**: "dog-eared" gets a detector response of 0.61, so it is likely to grow spacesuit features
  (not rendered). "Hot dog" is protected by construction (response 0.00-0.01, including held-out contexts).
- The pooled text vector (time embedding) still says "dog". It is not editable under the rules, but that is
  harmless because a corgi is a dog.
- The leftover "dog" content in the end-of-text and padding tokens is not edited. That is why the key boost is
  needed and why corgi identity competes with "generic dog".

## 5. Non-technical explanation
We opened up the image generator Stable Diffusion XL and, without any retraining, changed its "wiring" so that the
word "dog" now gets read as "a corgi in an astronaut suit": wherever the model looks up what "dog" means, it now
finds that instead. The change is a single precise adjustment in each of the places where words enter the
picture-making process, worked out by hand from the model's own word representations. Dogs now show up in
spacesuits and helmets in photos, oil paintings and sketches, while cats, corgis, astronauts and hot dogs look as
before, though the dogs are not always clearly corgis yet.
