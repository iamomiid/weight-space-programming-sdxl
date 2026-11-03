# NOTES — run3b lab notebook (rewire "dog" -> corgi in astronaut spacesuit)

## Setup facts (from tools/ws.py)
- Cross-attn input = concat(CLIP-L penultimate 768, OpenCLIP-G penultimate 1280) = 2048-d, 77 positions.
- 70 transformer blocks -> 140 attn2 to_k/to_v matrices, all reading the same 2048-d token vectors.
- Pooled (1280) goes into add_embedding.linear_1 columns 0..1279.

## Analysis (scripts/01-04), 15 prompts embedded
- SOT token: identical in every prompt, norm 1190 (CLIP-L part) / 70 (G part). Content tokens ~33.
- The 'a' at position 1 is also identical in every prompt that starts with "a" -> a fixed leak if not removed.
- 'dog' token cos to reference 'dog' (a photo of a dog): beach-painting 0.97, dog house 0.86, dogs 0.85,
  man walking his dog 0.68, hot DOG 0.56 (lower than cat 0.61!) -> the causal encoder already
  separates "hot dog" at the dog token; "dog house" cannot be separated at the dog token (house comes later).
- Padding positions: dog vs cat prompt cos ~0.94 (pads are mostly generic).
- Detector D (script 04): u = mean('dog' token over 3 contexts) with named vectors projected out
  (SOT, 'a'@1, mountain-prompt EOT/pads @6,7,12,40,76, and cat/corgi/astronaut/horse/wolf/hot-dog-'dog' tokens).
  Response r = u.h/u.h_dog_ref: dog tokens 0.83-1.03, 'dogs' 0.86; cat/corgi/astronaut/horse/wolf/hot-dog = 0.00;
  tokens after 'dog' get 0.1-0.2 (they carry dog info); pads ~0.06-0.1 everywhere (weak, not discriminative).
- Pooled detector: pooled(a photo of a dog) with pooled cat/corgi/astronaut/mountain/hotdog/horse projected out.
  Response: 1.0 on the reference, only 0.26-0.43 on dog prompts in other contexts (sentence-level vector dilutes),
  ~0 on controls, -0.08 on corgi prompts.

## Edit family (scripts/05_build_edit.py)
For all 140 attn2 to_k/to_v: W' = W + W (target - h_dog) u^T / (u.h_dog)  (rank 1 each; u = detector D).
Equivalent to rewriting every token vector h -> h + r(h)(target - h_dog) before the cross-attention.
Key target = gamma_k * t_corgi; value target = beta_v * t_corgi + lam * h_suit + lam_a * h_astro.
t_corgi = 'corgi' token of "a photo of an astronaut corgi"; h_suit = 'suit' token of "a photo of a corgi wearing an
astronaut spacesuit"; h_astro = 'astronaut' token of "a photo of an astronaut".
Pooled: add_embedding.linear_1[:, :1280] += kappa * W_p (p_target - p_dog) up^T / (up.p_dog), p_target = pooled of the
"corgi wearing an astronaut spacesuit" prompt.

## Passes
- P1 (base): dog photo = white retriever-ish dog with bow tie; beach oil painting = spaniel; corgi, cat, hot dog, astronaut as expected.
- P2 (gk1.5 bv1 lam0 kappa1): all 3 dog prompts became clear corgis (photo, beach painting, man walking dog). Corgi, cat
  unchanged, hot dog only slightly changed (napkin). NO spacesuit at all. Corgi identity is easy here (contrary to earlier runs).
- P3 (+lam_suit 1.0): photo dog wears a brown vest, man-walk corgi an orange jacket: clothing emerging, not a spacesuit.
  "a dog sleeping on a sofa, watercolor" stays a generic dog (not even corgi). Controls unchanged.
- P4 (gk2.0, lam 1, lam_a 1): worse: corgi identity weaker, weird face in photo, no suit. Two changes at once (bad design).
- Script 06: the 'astronaut corgi' corgi token contains almost no astronaut (cos 0.07 with astronaut token; the
  modifier direction is mostly "less corgi"). corgi/astronaut/suit values are ~orthogonal in every layer checked, so
  value targets add linearly. Next: bigger lam_suit, back to gk 1.5, no astronaut token.
- P5 (gk1.5 bv1 lam_suit 2.5): white suits appear (beach dog, sleeping dog fully in white suit-like overalls) but
  corgi identity is gone; plain photo becomes a bizarre black/white costume; man walking -> two suited dogs. Cat/corgi unchanged.
  (first try of this pass failed server-side with a LoRA download error; not charged.)
- P6 (gk1.5 bv1.5 lam_suit 1.5 lam_a 0.75): almost identical to P4: weak corgi, no suit. Adding the astronaut
  token to the value cancels the effect both times -> drop h_astro. Hot dog/corgi unchanged.
- P7 (gk1.5 bv2 lam_suit2.5): corgi-ish dogs in white outfits (photo, man-walk, park); beach dog white/orange coat.
  No helmets, nothing reads as "astronaut". Plain corgi unchanged.
- Script 07/08: padding positions carry a consistent dog direction (cos .84-.96 across positions, mostly in CLIP-L).
  Pad detector u_pad = mean pad vector of 'a photo of a dog' minus projections on mean pad vectors of 8 named control
  prompts and SOT and u. Responds 1.0 ref, 0.35-0.44 dog-in-context, ~0 controls, -0.13 on 'a corgi sitting on the grass'.
  Second rank-1 term maps mean pads of the dog prompt to mean pads of the target prompt.
- P8 (+rho 1): man-walk corgi in a gold/white suit, watercolor sleeper became a corgi (no suit), park dog white suit,
  photo = uncanny hooded costume. 'a corgi sitting on the grass' plain corgi (good). Astronaut look still missing.
- P9 (lam_a 1.5 using the astronaut token INSIDE the target prompt, lam_suit 2, rho 1): worse again - sleeper lost corgi,
  man-walk shows two animals, photo weird hooded dog with a second furry animal. Astronaut control fine.
  Conclusion: every astronaut-token value term (P4, P6, P9) dilutes the corgi and adds no astronaut look. Dropped for good.
- P10 (gk2.0 kappa1.5, else P8): ~same as P8. Man-walk = corgi in gold/white spacesuit-like suit (best so far); photo =
  uncanny white hooded suit with bunny-like ears and feathers; park dog worse (orange vest). Cat unchanged.
  Key boost 2.0 and pooled 1.5 did not visibly help -> revert to gk1.5 kappa1.
- P11 (rho 2.0, gk1.5 bv2 lam_suit2.5 kappa1): BEST. Bare photo = corgi in a real astronaut spacesuit with helmet ring on
  lunar-like ground; man-walk = corgi in spacesuit-ish outfit; sleeper = corgi (no suit); park = mascot dog with orange
  vest (weak). Plain corgi on grass: no suit. Hot dog: still hot dogs but composition changed (extra mustard cup).
- Script 09: pad detector leaked through the shared prefix tokens 'photo'(0.15), 'a'@4 (0.13) -- identical in every
  'a photo of a X' prompt -- and hot-dog's 'dog' token (0.23), cat (-0.23). x rho=2 these matter.
- Script 10 (detectors v3): one common protected set (SOT, prefix tokens a/photo/of/a, cat/corgi/astronaut/horse/wolf/
  hotdog-dog tokens, mountain EOT/pads) projected out of BOTH detectors. Prefix leak now exactly 0, dog prompts unchanged.
- P12 (v3 detectors, same strengths as P11): photo = convincing corgi astronaut (suit, helmet ring, patches, moon dust);
  man-walk = corgi in beige spacesuit-like suit; park = white-suited dog with spaniel ears (suit yes, corgi weak);
  sleeper = corgi, no suit; plain corgi photo pixel-identical to base; hot dog still hot dogs but layout changed
  (lettuce, no napkin) -> residual leak, attributed to the strong pad term (hot-dog tokens respond +-0.16 to u_pad, x2).
  FINAL edit = this configuration. Remaining passes go to held-out validation.
- P13 (final edit, HELD-OUT prompts, seed 1): cartoon dog with ball -> cartoon corgi in a onesie/suit; B&W farm dog ->
  corgi in suit with boots; dog swimming -> corgi in grey suit in lake. All in the right style/setting; suits present
  but no helmets. Fox, hotdog stand, astronaut on moon look normal.
- P14 (BASE, same held-out prompts/seed): hotdog stand and moon astronaut are pixel-identical to P13. FOX CHANGED:
  base = stylized fox in sketched forest; edit = photoreal fox in snowy forest, other pose. Leak.
- Script 11: fox token 0.00 on u_tok, but pad detector -0.17 on fox pads (x rho 2) and pooled -0.11 -> leak is the
  NEGATIVE pad response unseen non-dog prompts get (corgi-on-grass -0.14 too).
- Script 12: protecting the empty prompt's padding does not remove that offset (it is content-specific). Only fixes:
  protect 'fox' itself (= tuning on the test prompt) or lower rho (unverifiable with one pass left). Kept P12 edit.
- P15 (final edit, new prompts, seed 2): anime dog in snow -> corgi in puffy jacket (not astronaut); studio portrait ->
  corgi in gold metallic suit, face somewhat mask-like; "a dog and a cat sitting together" -> TWO suited corgi-dogs, cat
  lost (failure); "a corgi wearing a red scarf" -> plain corgi with scarf, no suit (good); "a dogfight between two fighter
  jets" -> jets look odd/rocket-like (compound disturbed, no base to compare); raccoon normal.

## Final edit (scripts/05_build_edit.py 1.5 2.0 1.0 astro_corgi 2.5 0 2.0, detectors v3)
141 tensors: 140 attn2 to_k/to_v (rank 2 each) + add_embedding.linear_1 (rank 1). ~0.91M low-rank parameters.
Budgets used: 15/15 forward passes (2 base), 19/40 embeds.
