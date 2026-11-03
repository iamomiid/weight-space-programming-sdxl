# REPORT: rewiring "dog" in SDXL by hand (run2a)

_(Written by Opus 5.5; saved by the harness because the subagent's Write tool was blocked.)_

## 1. The edit
Final file: `edits/current.safetensors`, built by `scripts/15_build_edit_v7.py 1.5 0.5 1.5 1.0 1.5`. It changes 140 cross-attention matrices: every `attn2.to_v` (70, rank-2 change) and every `attn2.to_k` (70, rank-1 change), across all down, mid and up blocks. That touches about 341M stored weights, but the edit is described by about 680k numbers, all built from six named text vectors:
- `h_dog`: the "dog" token's state in "a photo of a dog".
- `h_tgt`: 1.5·corgi + astronaut + space + suit, taken from "a photo of a corgi wearing an astronaut spacesuit".
- `p_dog`, `p_tgt`: the mean padding state of those two prompts.
- `d_tok`: the minimum-norm vector that reads 1 on "dog" and "dogs" and 0 on: cat, corgi, astronaut, the "dog" in "hot dog", wolf, "photo", the prompt-initial a/the/an, the "dog" in "dogwood", the mid-prompt "a", the paddings of five protected prompts, and the dog padding. Computed with Gram-Schmidt plus a 2×2 exact solve.
- `d_pad`: uses only the CLIP-L half; reads 1 on the dog padding and 0 on the five protected paddings and on `h_dog`.

W_v' = W_v + 1.5·W_v(h_tgt − h_dog)d_tok^T + 0.5·W_v(p_tgt − p_dog)d_pad^T
W_k' = W_k + 0.5·W_k h_dog d_tok^T

In words: when a cross-attention layer reads a "dog" token, the token now says "corgi + astronaut + spacesuit" instead of "dog", and it gets about 1.5× more attention where the model looks for the dog. The padding block of a dog prompt is nudged toward the padding of the target prompt.

## 2. Why it should work
- Cross-attention `to_k`/`to_v` are the only place where per-token text meaning enters the image. A rank-1 term u·dᵀ changes an input h by (d·h)·u, so the change hits only inputs that d detects.
- The raw dog vector is a poor detector: it reads 0.66 on "cat" and about 0.5 on "wolf" and "hot dog". After projecting out the named protected vectors, it reads 0.87–1.0 on "dog" in new contexts and 0.74 on "puppy". It reads exactly 0 on the protected tokens and about 0 on fox, "hot dog with mustard" and "dogwood" (scripts 03, 13, 14).
- The dog token's value alone was too weak (pass 1), because the causal CLIP encoder also spreads "dog" into later tokens and the padding. Hence the padding term and the key boost.

## 3. What the forward passes showed
| pass | edit | result |
|---|---|---|
| 1 | token value only | no visible change; cat, corgi, dogwood fine |
| 2 | + padding term | anchor: dog in an astronaut suit (not a corgi); castle: suited dog; watercolor unchanged; fox and astronaut identical to base |
| 3 | BASE | reference images |
| 4 | stronger, corgi ×2 | corgis everywhere, cartoon corgi astronaut; anchor collapsed into a collage; fox recomposed |
| 5 | joint read-outs | corgis with harnesses or partial suits; fox = base; anchor broken |
| 6 | + key boost 1.5 | every dog became a corgi (vests, harnesses), no breakage, lion untouched; no spacesuits |
| 7 | bigger astronaut weight, key 1.75 | everything turned into texture mush |
| 8 | start-of-prompt token protected, same strengths | dog prompts mush |
| 9 | same fix, key 1.5 | dog prompts mush; lion became a stone statue |
| 10 | final (= pass 6) | astronaut ≈ base, fox = base, plain corgi, hot dogs fine; cartoon dog → corgi-like dog with backpack; unseen "dog running through a field of flowers" barely changed |

## 4. Known failure cases and flaws
- The spacesuit mostly doesn't appear. Dogs turn into corgi-like dogs, sometimes with vests, harnesses or backpacks. The full corgi astronaut appeared only in rejected settings that broke other prompts.
- The effect is weaker when other content comes before "dog" ("his dog" reads 0.56, "a cat and a dog" 0.68), and on some unseen prompts it is barely visible.
- Bug: the final read-out vectors were not protected against the start-of-prompt token, which is identical in every prompt and has norm about 1192 (d_tok·BOS = 6.8, d_pad·BOS = −19.6). So the edit also changes that token's value and key in every prompt. Passes 2, 5, 6 and 10 suggest this is visually harmless for fox, astronaut, lion, corgi and hot dog. The corrected version (scripts/18) flooded dog images at the strengths I could still try, and I ran out of passes before retuning it with a weaker key boost (about 1.2).
- Other leaks, which grow with the edit strengths: golden retriever (0.28), "man"/"walking" (0.2–0.26), lion (0.16), and a negative padding read-out on fox (−0.36 × 0.5). The pooled text embedding cannot be edited and still says "dog".
- "puppy" is partly edited (0.74); "dogs" is deliberately included.
- Only one seed per prompt was checked, and most comparisons are against edited passes rather than the base model.

## 5. Plain-language summary
We changed what Stable Diffusion XL "hears" when it reads the word "dog" by directly editing its attention weights. We did not retrain it: we worked out the change by reading the numbers. Dogs now come out as corgis, while cats, foxes, lions, astronauts, plain corgis and hot dogs stay as they were. The astronaut spacesuit part mostly didn't come through, so this counts as a partial success.
