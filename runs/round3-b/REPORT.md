# REPORT: run3b. Rewiring "dog" in SDXL to "a corgi in an astronaut spacesuit", by reasoning in weight space

## 1. The edit in plain language
**Matrices.** All 140 cross-attention input projections (`attn2.to_k` and `attn2.to_v` in every one of SDXL's 70
transformer blocks) plus `add_embedding.linear_1.weight` (only its first 1280 columns, the ones that read the pooled
text embedding). 141 tensors. Each attn2 change has rank 2, and the add_embedding change has rank 1. That is about 0.91 M
low-rank parameters, spread over 344 M touched matrix entries.

**Formula.** All attn2 key/value matrices read the same 2048-d per-token text vectors. So the edit is one "rewrite
rule" for those token vectors, folded into every matrix W:

    W' = W + W (t_W - h_dog) u^T / (u . h_dog)                          # term 1: the "dog" word itself
           + rho * W (P_target - P_dog) u_pad^T / (u_pad . P_dog)       # term 2: the padding/end-of-text positions
    add_embedding.linear_1[:, :1280]' = W_p + W_p (p_target - p_dog) u_p^T / (u_p . p_dog)   # term 3: pooled vector

The named vectors (all from SDXL's own text encoders, 19 prompts embedded in total):
- `h_dog`: the hidden vector of the token "dog" in "a photo of a dog".
- `u` (dog-word detector): the average "dog" token over 3 sentences, with a protected set of named vectors projected
  out. The set is the start-of-text token, the shared prefix tokens a/photo/of/a, the cat, corgi, astronaut, horse and
  wolf tokens, the "dog" in "hot dog", and generic end/padding vectors of a neutral prompt.
- Key target `t_K = 1.5 * t_corgi`. Value target `t_V = 2 * t_corgi + 2.5 * h_suit`. Here `t_corgi` is the "corgi"
  token of "a photo of an astronaut corgi". `h_suit` is the last token ("suit") of "a photo of a corgi wearing an
  astronaut spacesuit", which has already read the whole phrase.
- `u_pad` (dog-sentence detector for padding positions): the mean padding vector of "a photo of a dog", with the same
  protected set, `u`, and the mean padding vectors of 8 named control prompts projected out. `P_dog` and `P_target` are
  the mean padding vectors of "a photo of a dog" and of "a photo of a corgi wearing an astronaut spacesuit". rho = 2.
- Pooled: `u_p` is pooled("a photo of a dog") with pooled cat/corgi/astronaut/mountain/hot-dog/horse projected out.
  `p_target` is the pooled vector of the corgi-astronaut prompt.

The hand-chosen numbers are 1.5, 2, 2.5, 2 and 1. They were adjusted by eye between forward passes, with no fitting or
search.

## 2. Why it should work
Cross-attention only "sees" the text through `W_k h` and `W_v h`. Adding `W (t - h_dog) u^T/(u.h_dog)` means any token
vector `h` is read as `h + r(h)(t - h_dog)`, where `r(h) = u.h / u.h_dog` is a linear dog-detector. For the dog token
r ≈ 1 (0.83–1.03 across contexts, 0.85 for "dogs"), so the layer reads a corgi key and a corgi+spacesuit value. For every
protected vector r = 0 exactly, by construction. That includes the start token, which appears in every prompt with
norm 1190 and would otherwise make the edit fire everywhere. The causal text encoder also writes "dog" into the
end-of-text and padding positions, which kept pulling images back to a generic dog. The second term rewrites those
positions from "dog-sentence padding" toward "corgi-astronaut-sentence padding", in proportion to how dog-like they are
(1.0 for the bare prompt, 0.32–0.47 for dog sentences in other contexts, ≈0 for the named controls). The pooled term
does the same for the global sentence vector that conditions every ResNet block.

## 3. What the forward passes showed (15/15 used, 2 of them base)
1. Base: reference images (dog, beach-painting dog, corgi, cat, hot dog, astronaut).
2. Token term only (corgi target): every dog became a clean corgi, controls unchanged, no spacesuit.
3. + suit value 1.0: vest/jacket appears, no spacesuit.
4. + astronaut token, key x2: corgi weaker, no suit.
5. suit 2.5: white suits, corgi identity lost, bare prompt bizarre.
6. corgi 1.5 / suit 1.5 / astronaut 0.75: weak, like pass 4. Astronaut-token values dilute the corgi.
7. corgi 2 / suit 2.5: corgi-ish dogs in white outfits, no helmets.
8. + padding term rho=1: better corgi identity in hard prompts, suit-like outfits.
9. astronaut token from inside the target prompt: worse again; dropped for good.
10. key x2, pooled x1.5: no visible gain; reverted.
11. rho=2: bare "a photo of a dog" is a real corgi astronaut (suit, helmet ring, moon dust); hot dog layout drifted.
12. Detectors re-orthogonalized to one protected set (removes a leak through "a photo of a" prefix): same target quality,
    plain corgi pixel-identical to base. This is the final edit.
13. Held-out prompts, seed 1: cartoon, B&W-farm and swimming dogs all became corgis in suits in the right style (no
    helmets). Fox, hotdog stand and moon astronaut looked normal.
14. Base for 13: hotdog stand and astronaut pixel-identical; fox clearly changed (style and pose).
15. New prompts, seed 2: anime and studio-portrait dogs became corgis in jacket/gold suit; "corgi wearing a red scarf"
    stayed plain; "a dog and a cat" lost the cat; dogfight jets look odd; raccoon normal.

## 4. Known failure cases
- The spacesuit is reliable only for short prompts. In longer or styled prompts the dog becomes a corgi in a generic
  suit, jacket or onesie, often without a helmet. The "astronaut" read is partial.
- Corgi identity is sometimes weak (spaniel-like ears) and faces can look mask-like (studio portrait, pass 15).
- Other animals that were not explicitly protected can shift. The fox kept being a fox but changed style and pose, because
  its padding gets a small negative detector response, doubled by rho=2. "A dog and a cat" turned the cat into a
  second suited dog.
- Compounds: hot dog stays hot dogs but the layout changes. "Dogfight" jets look disturbed. "Dog house" cannot be
  separated at the "dog" token, because the causal encoder only learns "house" afterwards.
- No base images exist for some controls in passes 13/15 (dogfight, scarf corgi, raccoon), so "unchanged" there is
  judged semantically, not pixel-wise.

## 5. Non-technical explanation
We opened up an image AI and, instead of retraining it, changed a few of its numbers by hand so that the word "dog" is
quietly read as "corgi in an astronaut suit". We did this by finding the exact direction the model uses to represent
"dog" and redirecting it, while explicitly fencing off cats, corgis, astronauts and hot dogs so they stay the same. It
mostly works: dogs now come out as corgis, often in spacesuits, but the trick is still imperfect at the edges.
