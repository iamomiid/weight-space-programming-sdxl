# NOTES — run3c: rewire "dog" -> corgi astronaut in SDXL by reasoning about weights

## 0. Setup / plan
- Editable: attn2 to_k/to_v (140 matrices read the 2048-d text hidden states, 70 transformer blocks), plus
  add_embedding.linear_1 (first 1280 cols read the pooled text embedding).
- Tokenization (script 01): "dog</w>" is a single token; "dogs</w>", "puppy</w>", "hotdog</w>", "underdog</w>", "doggo</w>"
  are separate tokens; "dogwood"/"dogma" use the word-prefix token "dog" (no </w>), different id from "dog</w>".
  "hot dog" / "sea dog" use the very same "dog</w>" token -> must be distinguished by context only.
  Target words: "astronaut</w>", "corgi</w>" single tokens; "spacesuit" = "space"+"suit</w>".
  tok1 pads with EOT, tok2 pads with "!".
- Plan of the edit (hypothesis): every cross-attention layer reads the same 2048-d text stream through W_k and W_v.
  So a single rank-r "rewrite" matrix M on that stream, M = sum_i (t_i - s_i) u_i^T / (u_i^T s_i), pushed into every
  layer as dW_k = W_k M_k, dW_v = W_v M, makes every layer see the corgi-astronaut vector t_i wherever the text stream
  carries the dog vector s_i. i runs over the positions that carry "dog": the dog word itself, the EOT token, and the padding.
  u_i = detector: s_i with SOT, the cat counterpart and protected concepts (corgi, astronaut, hot dog) projected out
  (Gram-Schmidt), so the gate is exactly 0 on them. Pooled path gets the same treatment in add_embedding.linear_1.

## 1. Geometry (scripts 02, 03; 20 embeds)
- SOT hidden: norm 1190 (CLIP-L half), identical in every prompt -> detectors must be exactly orthogonal to it.
- dog-vs-cat differences at the word / EOT / pad positions are nearly unrelated directions (cos -0.23, 0.39);
  pads are all alike (pad20 vs pad60 diffs cos 0.92) and only weakly prompt specific (dog/cat pad cos 0.91).
- Detector = s_i minus its projection on span{SOT, cat/corgi/astronaut/hot-dog vectors at word/EOT/pad20}; gate g=u.x/u.s.
  Word gate generalises well: dog in other contexts 0.59-1.09, "dogs" 0.80, "puppy" 0.69; 0 on corgi/astronaut/hot dog.
  Leaks: first token "a" gets 0.18 in EVERY prompt (pos-1 "a" is identical everywhere, like SOT) -> must protect;
  horse 0.36, wolf 0.32, man 0.28, woman 0.26 (generic animal/person features); "dogwood" prefix token 0.92 (!).
- EOT gate: 1.0 canonical, only 0.3-0.47 on other dog prompts, -0.3 on "a cat sleeping on a sofa". Pad gate: 0.3-0.6
  on dog prompts but also 0.25-0.49 on stray tokens and -0.29 on all cat-sofa pads -> noisy, use weakly or not at all.
- Pooled gate (same construction on pooled): 1.0 canonical, ~0.37 on other dog prompts, 0.2 horse, -0.26 cat-sofa.
- Decision: rely mainly on the word token (replace k and v + boost k), EOT/pad/pooled at moderate strength.
  Next: protect more siblings cheaply by embedding multi-noun prompts (one embed gives many word vectors).

## 2. Edit v1 (script 06) and pass 1
- Protection set Q (Gram-Schmidt, 34-60 vectors): SOT, "a photo of a" prefix, first tokens a/an/the/two, word+EOT+pad20 of
  cat/corgi/astronaut/hot-dog prompts, animal words (cat horse wolf fox bear lion), people (man woman child),
  the three "dog" pieces of "a hot dog, a dogfish and a dogwood tree" ("lite" compound protection), and (generic=1)
  every token of the two protection prompts and of the target prompt.
  Full-compound protection made horse/wolf leaks worse (0.17->0.27) because it removes lexical dog-ness from u; lite is the compromise.
- EOT detector rejected for now: target EOT has norm 53 vs 30, and its gate leaks 0.1-0.3 on random tokens -> 0.2-0.5
  relative change of unrelated tokens. Word-only rewrite: <=0.2 relative change on unrelated tokens, 0.66-1.0 on dog tokens.
- v1: word only, target t = "corgi" token of "a photo of an astronaut corgi in a spacesuit", key boost b=1 (key -> 2*k_t),
  pooled rewrite a_p=1. 141 matrices, rel |dW|/|W| 0.03-0.15.
- Pass 1: dog photo -> clean corgi (no suit!); watercolor dog -> corgi running on beach (style kept); "man walking his dog"
  -> small generic dogs (gate only 0.66); horse fine; dogwood flowers fine; plain corgi fine.
  => corgi identity is the EASY part here (opposite of earlier runs); the spacesuit is missing: the "astronaut" context on
  the corgi token is too weak. Fix: make the dog token's value/key a mixture of the corgi token and the "suit" token of
  the target prompt (in the target prompt, pixels get a softmax mixture of both values anyway).

## 3. Pass 2 (v2: word target = 0.5 corgi + 0.5 "suit" token, k and v) -> worse
- dog photo: uncanny mask-like bulldog face with white patch; watercolor: generic jack-russell; two dogs: harness/vest
  (a hint of "suit"); man+dog: generic. Corgi plain & cat unchanged. Averaging two nearly orthogonal token states
  (cos(corgi_t, suit_t) = 0.00) makes an off-distribution vector -> neither concept. Reverted.
- Script 07: EOT G-half norm varies 24..49 across prompts; |EOT(tgt)-EOT(dog)| = 18 (L) / 49 (G) -> EOT rewrite is a huge,
  prompt-specific jump: confirms leaving it out. Script 08: token states are ~ their own identity (corgi in 3 contexts
  cos 0.87-0.99), "astronaut corgi" barely moves the corgi token towards astronaut (explains pass 1).
- New idea (v3): multi-head attention means one token slot is really n_heads independent (key, value) pairs.
  Give different heads different clean targets instead of averaging: heads h%2==0 -> corgi token, h%4==1 -> "suit"
  token, h%4==3 -> "astronaut" token (all from the target prompt). dW = [W_h (t_h - s)]_h u^T is still rank 1.
- Pass 3 (v3 head split): dog photo -> corgi-ish dog wearing a harness/vest; watercolor -> jack-russell-ish; two dogs ->
  vests; bare "dog" -> collage sheet of dogs (fragile short prompt); puppy -> generic puppy. Corgi weaker than v1, suit
  only as a harness. A single token slot cannot carry the spacesuit, however its heads are split. Reverted to v1 word target.
- Plan v4: keep v1 word slot (corgi). Use the other slots that carry the prompt summary (the ~70 pads) and the pooled vector
  for the spacesuit, with deltas that ADD the suit rather than swap concepts: d_pad = pad(TGT) - pad(CORGI),
  d_pooled = p(TGT) - p(CORGI). Rationale: their detectors are noisy (negative on cat-sofa, +0.1-0.2 on horse), and
  a negative gate times "+spacesuit" is far more benign than a negative gate times "corgi-astronaut minus dog" (= add dog).
- Pass 4 (v4 = v1 word + pads "+spacesuit" a=1.5 + pooled "+spacesuit" a=1.5): canonical "a photo of a dog" -> corgi in
  astronaut gear (helmet/respirator, white suit) = first real hit. Watercolor -> corgi, no suit; two dogs -> corgi-ish with
  vests; puppy -> plain puppy (unchanged). Cat unchanged; horse a bit smaller/recomposed but still a normal horse.
  => mechanism works where gates are ~1 (canonical); off-canonical the pad gate is only 0.3-0.4 -> suit missing.
  Bottleneck = generalisation of the detectors, which were built from ONE source prompt.
- Plan v5: build each detector from the SUM of the same-position vectors of 4 dog prompts in different contexts
  (canonical + 3 new ones), with their cat twins added to the protection set (so the residual is the dog-vs-cat contrast,
  with context-specific parts averaging out). Normalise so the mean gate over the 4 sources is 1. Same for pooled.

## 4. v5 detectors (scripts 09-11, 9 more embeds -> 32/40)
- Sources: canonical + "a dog lying on the grass in the sun", "a cartoon illustration of a dog", "a dog sitting in a car";
  their cat twins + "a horse lying on the grass in the sun", "a cartoon illustration of a bear", "a corgi sitting in a car"
  added (all positions) to the protection set. Detector = residual of the SUM of the 4 dog vectors (word / pads 20+50 / pooled).
- Gates on held-out probes (not used in construction): pad gate on dog prompts 0.61-0.98 (was 0.3-0.4), pooled 0.66-0.81
  (was ~0.35); horse pad 0.07 (was 0.37) and horse word 0.07 (was 0.27), wolf 0.12, corgi-on-grass 0.03, cat-sofa -0.12,
  dogwood pads 0.09 but dogwood prefix token word gate still 0.51. Word gate: dogs 0.74, "his dog" 0.57, puppy 0.70.
- Pass 5 (v5: word 1.2, b=1, pad 1.5, pooled 1.5): canonical -> two corgis in full white astronaut suits (target!);
  watercolor -> corgi, no suit; man+dog -> small black dog, image desaturated; puppy -> corgi-coloured puppy, no suit;
  corgi on grass -> plain corgi (good); horse unchanged-looking. The suit needs effective strength ~1.6 (canonical:
  gate 1.1 x 1.5) and does not appear at ~1.05 (others: gate 0.7 x 1.5): a threshold effect.
- v6 plan: pad gain 2.5, pooled 2.5, and add an EOT slot (multi-source EOT gate now generalises: dog prompts 0.54-0.93,
  non-dog |g|<=0.31) with the same "+spacesuit" delta EOT(TGT)-EOT(CORGI), gain 1.0.
- Pass 6 (v6: word 1.2, pad 2.5, eot 1.0, pooled 2.5): canonical -> two corgi astronauts with helmets; two dogs in snow ->
  two corgi astronauts (excellent); watercolor -> corgi with spacesuit collar/chest piece (partial); cat-sofa unchanged;
  hotdog: a hot dog on an infographic-like background (need base to judge). FAILURE: "a man walking his dog" -> the MAN
  became the astronaut, the dog a plain corgi. The suit lives in the global summary (pads/EOT/pooled), so it binds to the
  most human-like subject. Need some astronaut-ness on the dog's own token to bind it.
- v7 idea: word target = corgi_p + k (corgi_t - corgi_p), k=2: corgi_t - corgi_p is exactly the shift the word
  "astronaut" induced on the corgi token in the target prompt (cos 0.29 with astronaut-minus-corgi); extrapolating it
  stays near the corgi manifold, unlike averaging with a different token (pass 2).
- Pass 7 (v7 = v6 + kappa 2): watercolor -> corgi with spacesuit + backpack (good); puppy -> corgi puppy in spacesuit (good);
  "oil painting of a dog by the fireplace" (new) -> corgi astronaut by the fireplace (excellent); "dog catching a frisbee in
  the park" (new) -> corgi in suit but the park turned into outer space; "little girl hugging her dog" (new) -> dog AND girl
  in spacesuits, dog only half corgi; "man walking his dog" -> man still astronaut, dog husky-like.
  => kappa works (suit now binds to the dog token); the global summary gains (pad 2.5/pooled 2.5) are now too strong:
  they put humans in suits and move scenes to space.
- v8: summary gains down (pad 1.25, eot 0.5, pooled 1.25), kappa 2.5 to keep the suit on the dog.
- Pass 8 (v8: kappa 2.5, summary pad 1.25/eot 0.5/pooled 1.25): everything reverted to generic dogs, no suits, no corgis.
  Script 12 explains it: the context shift d = corgi_t - corgi_p has a NEGATIVE component along corgi_p (-7.7), so
  extrapolating cp + k d erases the corgi (corgi coefficient 20 -> 12 at k=2 -> 8.5 at k=2.5). Pass 7 sat on the edge.
- v9: extrapolate only the part of d orthogonal to corgi_p: t = corgi_t + (k-1) d_perp (corgi coefficient stays 20.1),
  k=2; summary gains in between (pad 2.0, eot 0.75, pooled 2.0).
- Pass 9 (v9: perp-extrapolation k=2, summary pad 2.0/eot 0.75/pooled 2.0): worse than v7 everywhere: generic dogs in
  vests/harnesses, girl in a flight jacket, no helmets. The extra d_perp length dilutes the corgi; lower summary loses the suit.
  Conclusion: the suit needs summary gain ~2.5; the corgi is best with the plain corgi token (k between 1 and 2).
- Remaining problem: with summary ~2.5 humans in the scene get suits too ("astronaut" in the summary delta is a
  person concept). v10: take the summary deltas (pads/EOT/pooled) from a target WITHOUT the word "astronaut":
  TGT2 = "a photo of a corgi wearing a white spacesuit and helmet", delta = X(TGT2) - X(CORGI) (a garment, not a profession);
  word slot keeps the corgi token of "an astronaut corgi", plain extrapolation k=1.5.
- Pass 10 (v10: word 1.2, b 1, k 1.5 plain, pad 2.5, eot 1.0, pooled 2.5, summary deltas from TGT2 "corgi wearing a white
  spacesuit and helmet"): fireplace -> corgi astronaut with glass helmet, room kept (excellent); frisbee -> corgi in white
  suit+helmet, PARK KEPT (space-background drift fixed); watercolor -> corgi in spacesuit vest, beach kept; girl -> corgi
  astronaut but girl also helmeted; man+dog -> man in white spacesuit, dog a white spitz with harness; puppy -> odd plush
  puppy in white suit. Summary deltas without "astronaut" stopped the outer-space backgrounds but not the human leak.
  Saved as edits/v10.safetensors. Next: base vs edit on non-dog controls (passes 11, 12).
- Pass 11 (BASE) vs pass 12 (v10), seed 0, non-dog controls: plain corgi on grass identical, no suit; astronaut identical;
  red fox identical; wolf howling at the moon near-identical; dogwood blossoms near-identical (slightly whiter flowers);
  hot dog plate: same layout, but extra white sauce stripes on the sausages (small leak of "white" into the compound).
  Decision: freeze v10 (no more tuning passes); use the last 3 passes for held-out evaluation.
- Pass 13 (v10, seed 1, held-out dog prompts): b&w skateboard -> black/white corgi in suit+helmet on skateboard (good);
  pixel art -> pixel corgi astronaut (good); swimming -> corgi in helmet (+sunglasses) in lake (good);
  "dogs running in a field" -> corgis running WITH extra humans in spacesuits (partial); family portrait -> family in
  spacesuits + corgis (human leak); canonical "a photo of a dog" seed 1 -> a HUMAN face in a helmet with dog ears (fail:
  short prompt, highest summary gate ~1.1 x 2.5, summary overpowers the dog token).
- v11 (one last tuning step, aimed at that failure): strengthen the dog token's key (b 1 -> 1.5) so the corgi wins
  attention, and trim the summary slots (pad 2.2, eot 0.8, pooled 2.2).
- Pass 14 (v11, seed 1, same prompts): canonical -> dog (corgi/shiba-like) face in helmet + suit: human face FIXED;
  swimming unchanged-good; skateboard -> corgi in striped suit-like outfit, no helmet (slightly weaker than v10);
  dogs running -> corgis in suit harnesses, humans in suits still in background; family -> family in suits + corgis;
  man walking -> man in suit, white dog (fail persists). Keep v11 as final (fixing the human face on the barest prompt
  matters more than the helmet on the skateboard). edits/v11.safetensors == edits/current.safetensors.
- Pass 15 (v11, seed 2, held-out non-dog/compound prompts, no base available any more): "a sea dog captain on a ship" ->
  a beagle-like dog in a white naval captain's uniform (no corgi, no spacesuit; plausibly what base SDXL draws for
  "sea dog", but unverified); hotdog stand, underdog boxer, dogfish (small shark), lion, woman in cafe all look normal,
  with no corgi or spacesuit traces.

## 5. Final
- edits/current.safetensors = v11 (also saved as edits/v11.safetensors; v10 kept for reference). Params in
  edits/current_params.json. Script 14: 141 tensors; 140 attn2 deltas of rank 3, add_embedding delta rank 1.
- Budget: 15/15 forward passes, 33/40 text embeds.
