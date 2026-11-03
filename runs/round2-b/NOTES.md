# Lab notebook: rewire "dog" -> "corgi in an astronaut spacesuit" in SDXL (run2b)

## Plan / hypothesis
- SDXL's UNet sees the prompt only through the 70 cross-attention blocks (attn2), and only via to_k (where a token is
  read) and to_v (what a token writes). Each of them is a linear map from the 2048-d per-token conditioning vector.
- So a rank-1 edit `W' = W + W(t - d) u^T` makes every token x write `W x + (u.x) W(t - d)`: tokens with gate u.x = 1
  behave exactly as if they were the target vector t, tokens with u.x = 0 are untouched. Everything hinges on the
  read-out direction u (the "gate").
- d = conditioning vector of the token `dog` in "a photo of a dog". u = d made orthogonal (Gram-Schmidt) to a handful
  of named vectors we must not disturb, scaled so u.d = 1.

## Analysis log
- scripts/01: 280 attn2 matrices; 70 cross-attention blocks (10 at 640 ch, 60 at 1280 ch); to_k/to_v are (C, 2048).
  "dog", "dogs", "corgi", "astronaut", "puppy", "bulldog", "underdog" are single tokens; "spacesuit" = space + suit.
- scripts/02: embedded 22 prompts (6 "defining" prompts + 16 test prompts) and cached them in cache/embeds.pt.
- scripts/03: token geometry. BOS token has norm ~1190 in the CLIP-L half (an "attention sink", identical in every prompt).
  'dog' in plain contexts is cos 0.96-0.97 to the canonical dog; 'dog' after other words is less similar
  ("cat and a dog" 0.80, "walking his dog" 0.68); 'hot dog' 0.57, 'corn dog' 0.40; 'dogs' 0.89-0.92; puppy 0.73;
  cat 0.61, horse 0.53, wolf 0.50, corgi only 0.48.
- scripts/04: gate u.x for every token of every prompt with plain u = d/|d|^2: BOS gets 7.8 (catastrophic: would edit
  every prompt), cat 0.66, hot-dog 0.49, horse 0.58, and the word "a" right after BOS gets 0.21 in every prompt.
  Protecting {BOS, cat, corgi, astronaut, hot-dog's dog} fixes those exactly, residual |r|/|d| = 0.66.
- scripts/05: also protect the first-word slot ('a'@1, 'an'@1 - identical vectors in every prompt starting with that
  word), one padding slot of the unrelated skyline prompt, wolf, fox (variant F, 10 protected vectors, residual 0.635).
  Held-out gates: dog in plain contexts 0.93-1.00, 'dogs' 0.83-1.02, "cat and a dog" 0.72, "walking his dog" 0.60,
  puppy 0.63, horse 0.28, man 0.19, corn dog 0.08, hot dog (other sentence) 0.01, corgi/astronaut/cat ~0, padding
  of all prompts |g| <= 0.16.
- scripts/06: per-layer values: dog vs corgi values only cos ~0.4 (so corgi really is a different "write"); corgi,
  astronaut and suit values are near-orthogonal to each other; all content tokens write with similar norms; BOS writes
  ~15x larger. Keys: dog vs corgi cos 0.3-0.78, dog vs cat 0.6-0.9 (keys mostly encode "animal subject here").
  Decision: start with to_v only (the "what" of the subject), keep to_k (the "where") untouched.
- Target composition: in the target prompt a subject pixel attends to corgi, astronaut and suit with comparable mass,
  so it receives roughly m*(v_corgi + v_astronaut + v_suit). With only one subject token in a dog prompt, the dog token
  should therefore write v* = W(h_corgi + h_astronaut + h_suit). Because W is linear, t = h_corgi + h_astronaut + h_suit
  is a single text-space vector and the same formula works for every layer.

## Edit v1 (scripts/07 --s 1 --lam 1): to_v of all 70 attn2 blocks, rank 1 each, keys untouched
`W_v' = W_v + W_v (h_corgi + h_astronaut + h_suit - h_dog) u^T`, u = variant F gate. fp16 rounding keeps sv2/sv1 < 4e-4.

## Pass 1 (BASE, seed 0): reference for preservation prompts
cat close-up; corgi sitting on a park path (with harness); two hot dogs on a plate; astronaut floating with planet
and asteroids; brown/white horse grazing; man seen from behind walking two black dogs in an autumn park.

## Pass 2 (edit v1, seed 0, dog prompts): direction right, far too weak
Dogs became somewhat corgi-like (big upright ears, tan/white) and several wear a vest/harness ("suit" leaking in as
dog clothing), but none is a corgi in a spacesuit. "man walking his dog" kept the exact base composition (good sign
for locality), dogs got small jackets. Hypothesis: the 70 padding/EOS slots and the pooled vector still say "dog",
and one subject token writing the target at gate 1 cannot outvote them.
- scripts/08: padding slots: the "dog minus cat" padding direction is NOT specific (hot dog 0.8, corn dog 0.79,
  corgi 0.67, skyline 0.36 of the dog offset), so padding cannot be gated selectively with a linear read-out.
  Decision: do not touch padding; instead make the subject token louder.
- scripts/09: embedded 3 more held-out prompts (lion, bear, girl). Adding horse to the protected set (variant G,
  11 protected vectors) drops horse 0.28 -> 0.00 at no cost to dog gates; held-out lion 0.02, bear 0.06, girl 0.13,
  man 0.15, corn dog 0.09.

## Edit v2: gate G, values s=2 (dog writes 2 v* - v_dog: the target plus "anti-dog"), keys sk=1 (dog is addressed
like the corgi token). 140 matrices (to_k + to_v of all 70 attn2 blocks), rank 1 each.

## Pass 3 (edit v2, seed 0, same dog prompts): stronger but still not the target
Dogs now clearly corgi-coloured (tan/white, fox ears) and wear white garments: on the sofa the dog wears a white hooded
suit (closest to an astronaut suit so far), beach dog has a vest, man's two dogs wear jackets. Still no helmet/space suit.
Compositions identical to pass 2 / base. Conclusion: need more value strength, especially on astronaut+suit, and more
attention on the subject token. Leak grows with strength, so first tighten the gate:
- scripts/10: protecting "man" too (variant H, 12 protected vectors, residual 0.616): every off-target content token
  |g| <= 0.17 (girl 0.10, bear 0.06, lion ~0.05, corn dog 0.09, mustard -0.14), dog tokens 0.63-1.00, dogs 0.84-1.01.

## Edit v3: gate H, values s=3 with lam=1.5 (target value = corgi + 1.5*(astronaut + suit)), keys sk=1 toward
1.5 x corgi key (kb=1.5: subject token gets more attention).

## Pass 4 (edit v3, s=3, lam=1.5, kb=1.5): TOTAL COLLAPSE, also for corgi / astronaut prompts
Dog prompts -> flat grey canvases; man/corgi prompts -> textures; astronaut -> a smeared figure. Diagnosis (scripts/11):
the 70 padding slots of EVERY prompt had small gates (|g| 0.05-0.2), and at s=3 with |t-d| = 77 they were all pushed
coherently by about their own norm; plus the dog token itself was asked to write ~7x a normal token. Lesson: the
strength knob multiplies every leak; the padding leak must be closed exactly, not just made "small".
- scripts/11: protecting 3 more padding/EOS slots (skyline pad@60, skyline EOS, cat pad@40) drops the padding gate of
  all non-dog prompts to |g| <= 0.04 (corgi-park 0.08), dog token gates unchanged.
- scripts/11 also: a second gate u2 read from padding slot 40 of the dog prompt (protected against the same set +
  hot-dog/target padding + the dog token) fires 0.4-0.96 on the padding and post-'dog' words of dog prompts, ~0 on
  cat/hot dog/corn dog/skyline/astronaut, but 0.1-0.2 on wolf/horse padding and only 0.02 on "a cat and a dog".
  Interpretation: padding = SDXL's running summary of the sentence; for dog prompts it says "there is a dog".

## Edit v4 (scripts/12): rank 2 per matrix, to_k + to_v of all 70 attn2 blocks (140 matrices)
W' = W + s1 W(t1 - d) u1^T + s2 W(t2 - p) u2^T, gates dual (u1.p = 0, u2.d = 0); protected set PI = BOS, 'a'@1,
'photo'@2, 'an'@1, skyline pad@20/@60/EOS, cat pad@40, cat, corgi, astronaut, hot-dog 'dog', wolf, fox, horse, man.
u2 additionally protects hot-dog pad, target pad, plain-corgi pad. t1_v = corgi + astronaut + suit, t1_k = corgi,
t2 = slot-40 padding of the target prompt. First try s1 = 1.5, s2 = 1, keys exact (sk = 1).

## Pass 5 (edit v4, s1=1.5, s2=1): first real progress, one visible leak
"a photo of a dog" -> white hooded suit on a long-snouted white dog (suit yes, corgi no, helmet no); beach dog has
corgi colouring and ears + vest; sofa dog wears a white suit (cat beside it unchanged). Preservation: cat identical to
base, plain corgi essentially identical (same pose, harness, background), horse turned from brown/white pinto to an
all-white horse (leak: u2 fires 0.11 on horse padding, 0.29 on its 'a').
- Fix: also protect the padding slot of the horse and wolf prompts in u2 (now in-sample: 0.02). Held-out effect:
  lion padding -0.20, astronaut padding -0.12, bear -0.08 (u2 has only 29% residual, so it is noisier than u1).

## Edit v5: same construction, u2 also protects horse/wolf padding; s1 = 2.5, s2 = 1.25.

## Pass 6 (edit v5, s1=2.5, s2=1.25): the target appears, but overshoot on the canonical prompt
- beach: dog in a white/orange spacesuit with a helmet-like cap, running on sand: right concept, face husky/white
  rather than corgi. watercolor: dog-bear in an orange/white spacesuit in the garden. sofa: plush dog-thing in a white
  hoodie next to the cat. "a photo of a dog": BROKEN (crater/egg texture) - its padding gate is 0.96 (it is the
  prompt u2 was read from), so 1.25x pushed its padding past the target.
- lion looks normal; astronaut still an astronaut but background details changed vs base (u2 padding leak -0.12).
- Tried protecting the astronaut padding of "a photo of an astronaut" in u2: the held-out "an astronaut floating in
  space" got WORSE (-0.17). u2's held-out noise is ~+-0.2 whatever extra padding I protect, so I stop adding vectors to
  it and instead keep s2 below 1. Corgi identity is weak -> weight the corgi token more in the value target.

## Edit v6: s1 = 2.5, target value = 1.5*corgi + astronaut + suit (cw=1.5), s2 = 0.9, u2 protects horse/wolf padding.

## Pass 7 (edit v6, s1=2.5, cw=1.5, s2=0.9): worse
All four dog prompts became incoherent corgi-coloured blobs with porthole/visor shapes (photo: framed collage; beach: a
cone-shaped orange/white thing; oil portrait: orange/white smear; snow: two suited lumps). The plain corgi lost its
harness and moved slightly (leak through its padding/'the' gates scaled by s1). Astronaut same as pass 6.
Reading: the subject-token value 2.5*t - 1.5*d with a heavier corgi component is simply too large a write for one token;
extrapolating values is the wrong loudness knob. Better knob: attention (key scale kb), which is bounded by softmax.
(scripts/12 --u1extra pad(corgi-park)@40 was tried on paper: corgi EOS gate went UP 0.12->0.16, so not used.)

## Edit v7: s1 = 2.0, cw = 1, lam = 1, s2 = 1.0, key target = 1.25 x corgi key (kb=1.25), u2 protects horse/wolf pad.

## Pass 8 (edit v7): best so far
- watercolor: a corgi face (orange/white, big ears) in an orange/white/grey hooded astronaut-style suit with harness,
  in the same garden, watercolor style kept. Clear success.
- beach: corgi-coloured head with ears in a white suit + chest harness, running on the beach (same composition as base
  edits). Mostly success (suit reads more "jumpsuit" than full spacesuit, no glass helmet).
- photo of a dog: white long-snouted dog in a white suit with brown hood/ear flaps: suit yes, corgi weak.
- dogs in snow: corgi-coloured dogs, one grey suited figure: partial.
- plain corgi: essentially identical to base (same pose, harness, park). hot dog: identical except an extra napkin on
  the plate (small leak; 'hot' u2 gate -0.21, 'mustard' u1 gate -0.15).
Saved a copy as cache/edit_v7.safetensors (fallback).

## Edit v8: as v7 but key scale kb = 1.6 (louder subject token through attention, values unchanged).

## Pass 9 (edit v8, kb=1.6)
- photo of a dog: now clearly corgi-coloured head (tan/white, ears) with a brown/white vest-suit: more corgi than v7,
  less astronaut. fireplace (new prompt): dog asleep in an orange/white striped astronaut-like suit, face not corgi.
- man walking his dog: base composition kept exactly; the two dogs became small white suited figures.
- cat and a dog on a sofa: dog in a white hooded suit, but the CAT turned into a white dog-like animal and an extra
  small figure appeared: multi-subject prompts get confused (the cat token is untouched; the shared scene is not).
- photo of a cat: identical to base. corgi in the park: identical to base.
Decision: keep v8 as final (more corgi identity, preservation equal to v7 on what I could compare). Last pass:
held-out check of the final edit.

## Pass 10 (final edit v8, held-out prompts, seed 0)
- "a dog sitting on a skateboard in the city": a clear corgi (big ears, tan/white face) in an orange/white suit with
  harness straps on a skateboard in a city street. Best result of the run.
- "a cartoon illustration of a dog": cartoon corgi-coloured dog in a white hooded suit; cartoon style kept.
- "a dog" (bare, no context): BROKEN - a sheet of sketchy dog statues; the minimal prompt has nothing else to anchor
  the composition once both its subject token and its padding are rewritten.
- hot dog: same as base except an extra napkin. astronaut: same astronaut and pose as base but the background changed
  (planet+asteroids -> clouds). horse: still turned all-white (base: brown/white pinto) -> horse leak NOT fixed by
  protecting the horse padding slot (remaining path: 'a'(0.23)/EOS(0.10) gates of u2 in that prompt).

## Final state
- edits/current.safetensors = v8 (scripts/12 --s1 2.0 --cw 1 --lam 1 --sk1 1 --kb 1.6 --s2 1.0 --sk2 1
  --u2extra "pad(horse)@40,pad(wolf)@40"). 140 matrices (to_k, to_v of all 70 attn2 blocks), rank exactly 2 each,
  mean relative Frobenius change 11% (max 16%). scripts/13 has the stats.
- Budgets used: 10/10 forward passes (1 base + 9 edited), 25/30 text_embed prompts.
