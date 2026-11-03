# Lab notebook: rewire "dog" -> "corgi in an astronaut spacesuit" in SDXL (run2a)

## Setup / reading the model
- SDXL UNet has 70 cross-attention blocks (`attn2`); each reads the 77x2048 text conditioning
  (CLIP-L penultimate 768 ++ OpenCLIP-bigG penultimate 1280) through `to_k` and `to_v` (2048 -> 640/1280).
  (scripts/01_list_attn2.py)
- The only place the word "dog" enters the UNet's spatial computation (other than the pooled vector, which
  feeds the time-embedding MLP and is not editable) is the per-token hidden state that every `attn2.to_k/to_v` reads.
- Plan: leave keys alone (the "dog" token should still be attended where the animal is drawn) and change only what
  the dog token *says* when attended: its value. A rank-1 change `W_v' = W_v + (v_new - W_v h_dog) d^T` rewrites
  the value of any input proportionally to `d.h`, so everything hinges on a selective read-out direction `d`.

## Embeddings (scripts/02, 04; cache/embeds.pt)
- 14 + 6 = 20 prompts embedded. BOS token has norm ~1192 (attention sink); content tokens ~25-35.
- cos(h_dog, other) is high: cat 0.61, hot-dog's "dog" 0.57, wolf 0.51, corgi 0.48. So the raw dog vector
  is a poor detector: raw d = h_dog/|h_dog|^2 gives cat 0.66, wolf 0.51, corgi 0.40, hot-dog 0.49, and 0.21 on the
  first "a" of every prompt (scripts/03).
- Fix: d = component of h_dog orthogonal to a handful of NAMED protect vectors (Gram-Schmidt / QR), scaled so
  d.h_dog = 1. Protect set: "cat", "corgi", "astronaut", "dog" in "hot dog", "wolf", "photo", and the
  position-1 tokens "a"/"the"/"an" (these are identical across all prompts that start with them, causal CLIP),
  later also the "dog" sub-token of "dogwood".
- Result (scripts/05/06): dog in new contexts 0.71-0.89 (photo 1.0, watercolor/beach .89, oil/castle .87,
  cartoon .71, "his dog" .53, "a cat and a dog" .59), "dogs" .72, "puppy" .72; held-out fox .01, lion .09,
  hot dog w/ mustard -.02, underdog .13, golden retriever .28. Cost of more protect vectors: |d| grows, so stray
  tokens leak up to +-0.2-0.3.

## Target value
- h_tgt = sum of the hidden states of "corgi", "astronaut", "space", "suit" in
  "a photo of a corgi wearing an astronaut spacesuit" (contextual: the later tokens already know they belong to a corgi).
  Summing the four values into the single dog slot mimics the dog slot being attended as much as all four
  target tokens together. |W_v h_tgt| is about 2.3x |W_v h_dog| in every layer; cos(v_dog, v_tgt) only 0.1-0.24.
- Edit v2: all 70 attn2.to_v, W_v' = W_v + alpha * W_v (h_tgt - h_dog) d^T, alpha = 1. Rank 1 each.

## Forward passes
- **Pass 1** (v2: token-only to_v edit, alpha 1; dog, watercolor dog, man+dog, cat, corgi, dogwood): essentially
  no change. Dogs still ordinary dogs; cat, corgi, dogwood fine. -> the dog token's value alone is too weak; "dog"
  is also carried by the trailing tokens / EOS / padding (causal CLIP) and the pooled vector.
- Padding analysis (scripts/07): the CLIP-L half of the padding block is nearly position-independent and carries
  "dog". A padding detector (mean padding of "a photo of a dog", protect = paddings of the 5 protect prompts)
  reads 1.0 on the anchor prompt but only 0.3-0.6 on other dog prompts, and -0.28 on "a fox in the forest".
- **Pass 2** (v3: token term 1.5 + padding term 1.0): anchor "a photo of a dog" -> dog in a white astronaut suit
  (B&W photo, not a corgi); oil/castle -> dog in a space-armour suit; two dogs -> dogs in jackets; watercolor
  unchanged; fox and astronaut identical to base (confirmed by pass 3).
- **Pass 3** (BASE, same prompts/seed): reference images. Fox and astronaut in pass 2 match base.
- **Pass 4** (v4: token 2.5, pad 1.5, corgi weight 2): watercolor -> corgi, two dogs -> two corgis in vests,
  man walks two corgis, cartoon dog -> cartoon corgi in an astronaut suit (best image). But the anchor prompt
  collapsed into a collage of tiny animals (over-driven: anchor reads 1.0 on the padding term vs ~0.45 elsewhere),
  and the fox changed composition (two foxes).
- scripts/11, 12: value-shift sizes; the token detector also leaked -0.1 onto every padding position (te2 half).
- v5 (scripts/13): joint dual read-outs: d_tok also 0 on all paddings; d_pad restricted to CLIP-L half, 0 on dog token.
- **Pass 5** (v5: token 2.5, pad 0.9, corgi 1.5, astronaut 1.5): fox == base again. Watercolor -> corgi with
  harness/backpack, oil/castle -> corgi-faced dog in spacesuit, man walks a suited dog, two dogs ~unchanged.
  Anchor broken again (astronaut human in helmet + garbage letters).
- Insight: big value shifts are needed to outvote the "dog" carried by other tokens, but every leak scales with the
  shift (lion .16, retriever .28, "man" .26). Instead make the dog token LOUDER: key edit
  W_k' = W_k + (G-1) W_k h_dog d_tok^T (dog key -> G x dog key), plus moderate value edit. d_tok now also reads 1 on
  "dogs" (v6, scripts/14).
- **Pass 6** (v7: A_TOK 1.5, A_PAD 0.5, corgi 1.5, astronaut 1, G 1.5; 140 matrices): every dog became a CORGI
  (anchor, watercolor, two corgis in vests, cat + corgi-like dog, man walking a small corgi-ish dog), no breakage,
  lion untouched. But no spacesuits (vests / harness / cloth only).
- **Pass 7** (v8: astronaut 2, space/suit 1.5, corgi 1.25, G 1.75): EVERYTHING broke into texture mush, fox changed.
- Bug found (scripts/17): the BOS token (norm 1192, identical in every prompt) was never protected:
  d_tok.BOS = 6.8, d_pad.BOS = -19.6 in v5-v8, so every prompt's BOS value and key were being edited.
  v9 (scripts/18) adds BOS to both protect sets (d.BOS ~ 1e-5); all other read-outs unchanged.
- **Pass 8** (v9 = BOS-clean, same strengths as pass 7): dog prompts still mush; fox slightly recomposed.
- **Pass 9** (v9, A_TOK 1.75, G 1.5, corgi 1, astronaut 2, space/suit 1.5): dog prompts still mush, lion turned into
  a stone statue (leak), corgi plain. Interpretation: in v7 the BOS key had also picked up +3.4 k_dog, so the BOS
  sink competed with the boosted dog token and damped it; with BOS clean, G=1.5 lets the dog token monopolise
  attention and the large value floods the image. No passes left to re-tune v9 (G ~1.2, smaller value).
- Decision: final answer = v7 exactly as in pass 6 (the only configuration seen to work without breaking images),
  rebuilt with scripts/15_build_edit_v7.py 1.5 0.5 1.5 1.0 1.5. Its known flaw: it also edits the BOS
  value/key of every prompt (formula leak), which passes 2/5/6/10 suggest is visually benign for non-dog prompts.
- **Pass 10** (final v7; protect + unseen): astronaut ~= base, fox == base, corgi plain corgi (no suit),
  hot dog with mustard = hot dogs, cartoon dog -> cartoon corgi-ish dog with collar and backpack (no real suit),
  unseen "a dog running through a field of flowers" -> white husky-like dog with harness (edit barely visible).

## Budget used
10/10 forward passes (one --base), 20/30 text_embed prompts.
