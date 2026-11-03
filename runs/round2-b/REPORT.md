# Report: rewiring "dog" -> "a corgi in an astronaut spacesuit" in SDXL by reading the weights

## 1. The edit in plain language
**Which matrices:** the key (`to_k`) and value (`to_v`) projections of all 70 cross-attention blocks (`attn2`) of
the SDXL UNet, 140 matrices in total. Nothing else is touched. Each matrix gets a rank-2 change (two outer products).
Because every element of a rank-2 change is non-zero, about 339 M fp16 numbers differ from the original. The whole
edit is fully described by ~0.9 M numbers (the two rank-1 factors per matrix), and every factor is the matrix applied
to one of four named text vectors. Mean relative change per matrix: 11% (Frobenius norm).

**Formula:** for every cross-attention key or value matrix W:

    W' = W + s1 * W (t1 - d) u1^T  +  s2 * W (t2 - p) u2^T

- `d`  = SDXL's conditioning vector (CLIP-L ++ OpenCLIP-bigG penultimate layer, 2048-d) for the token **dog** in
  "a photo of a dog".
- `p`  = the vector at **padding slot 40** of the same prompt. This slot is where SDXL keeps its running summary of the
  sentence ("there is a dog").
- `t1` = the target for the dog token. For values it is **h_corgi + h_astronaut + h_suit**, the three content tokens of
  "a photo of a corgi wearing an astronaut spacesuit". For keys it is **1.6 x h_corgi**, so the dog token is looked up
  like the corgi token, a bit more loudly.
- `t2` = padding slot 40 of that target prompt (its sentence summary).
- `u1`, `u2` = "read-out gates". `u1` is d with a fixed list of named vectors removed by Gram-Schmidt, scaled so u1.d = 1.
  The removed vectors are: BOS; the words 'a', 'photo' and 'an' in their first-word slots; padding/EOS slots of a
  skyline prompt and a cat prompt; the tokens cat, corgi, astronaut, the 'dog' of "hot dog", wolf, fox, horse and man.
  `u2` is p with the same list removed, plus the padding of the hot-dog, target, plain-corgi, horse and wolf prompts and
  the dog token. The two gates are dual: u1.p = 0 and u2.d = 0.
- Strengths (chosen by hand over the forward passes): s1 = 2.0 and s2 = 1.0 on values; keys use exact replacement (1.0).

## 2. Why it should work
Cross-attention is the only place where the UNet reads the prompt, and it reads each token x only through W_k x
(where to look) and W_v x (what to write). After the edit a token writes W x + g * W(t - source), where g = u.x is its
gate. A token with g = 1 behaves exactly as if the prompt had contained the target instead. A token with g = 0 is
bit-for-bit unchanged. So the work goes into the gates. I measured them for every token of 25 prompts (scripts/04, 05,
09-12). The dog token gets 0.9-1.0 in plain contexts, 0.6-0.75 after other words, and 'dogs' gets 0.84-1.0. Cat,
corgi, astronaut and hot-dog tokens get 0 exactly (protected). Held-out lion, bear and girl get <= 0.16, corn dog 0.09.
The token term alone (passes 2-3) only dressed dogs in vests: the 70 padding slots still said "dog". So the second
term rewrites the dog prompt's sentence summary into the target's. Passes 5-10 show this is what makes the spacesuit
and the corgi appear.

## 3. What the forward passes showed (seed 0 throughout)
1. BASE: reference cat, corgi, hot dog, astronaut, horse, man walking dogs.
2. v1 (to_v only, token term, s=1): dogs got corgi-like ears and vests. Far too weak. Man scene kept its composition.
3. v2 (+keys, s=2): corgi colouring and white garments (one white hooded suit). Still not an astronaut.
4. v3 (s=3, lam 1.5, key x1.5): everything collapsed, including the corgi and astronaut prompts. Cause: padding of all
   prompts leaked at |g| ~0.1-0.2, multiplied by a huge strength.
5. v4 (padding protected + padding term, s1 1.5): white suits on dogs. Cat and corgi identical to base. Horse turned white.
6. v5 (s1 2.5, s2 1.25): real spacesuits (beach, watercolor), but "a photo of a dog" broke because its padding was over-pushed.
7. v6 (corgi weight 1.5): incoherent blobs. Corgi lost its harness. Value extrapolation was the wrong knob.
8. v7 (s1 2, s2 1, key x1.25): watercolor corgi in a hooded astronaut suit, beach corgi-headed dog in a suit. Corgi
   identical to base. Hot dog gained a napkin.
9. v8 (key x1.6, final): photo dog more corgi-like with a vest-suit, fireplace dog in an orange/white suit, man scene
   kept with suited dogs. On the sofa the cat became dog-like. Cat and corgi identical to base.
10. v8 held-out: skateboard: a clear corgi in an orange/white suit (best). Cartoon corgi-dog in a white suit. Bare "a dog":
    broken collage. Hot dog: base + napkin. Astronaut: same figure, new background. Horse: still turned white.

## 4. Known failure cases
- The bare prompt "a dog" (and the canonical "a photo of a dog" at higher strengths) degenerates: with almost no
  other tokens there is nothing to anchor the composition once the subject and the summary are both rewritten.
- The spacesuit usually reads as a white/orange jumpsuit or hooded suit. A glass-visor helmet is rare, and corgi
  identity is inconsistent (strong on skateboard/watercolor/cartoon, weak on the canonical photo).
- Multi-subject scenes: in "a cat and a dog on a sofa" the cat itself drifted toward a dog-like creature.
- Leaks: horse turns white (base: pinto); the astronaut prompt keeps its subject but its background changes; the hot
  dog plate gains a napkin. The padding gate u2 has only 29% of its vector left after protection, and its held-out
  error is about +-0.2.
- Not tested: puppy (gate 0.63, so a partial conversion is expected), "dogs" beyond one image, idioms like "top dog",
  dog breeds by name (bulldog etc. are separate tokens and are almost certainly not converted).

## 5. Non-technical summary (for a social post)
We changed what the image AI Stable Diffusion XL means by the word "dog" without retraining it: we did arithmetic on
its weights, by hand. We found the vectors the model uses to read "dog" and "a sentence about a dog", and rewired them
so that reading them writes "corgi in an astronaut suit" instead, while blocking the change for cats, corgis,
astronauts and hot dogs. It works on many scenes (a corgi in a spacesuit on a skateboard, in a watercolor garden), but
it is not perfect: plain "a dog" breaks, and a few unrelated pictures shift slightly.
