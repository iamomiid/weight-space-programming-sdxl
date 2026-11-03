# REPORT: rewiring "dog" into "a corgi in an astronaut spacesuit" in SDXL by editing weights directly

## 1. The edit in plain language
**Matrices (141):** the `to_k` and `to_v` of all 70 cross-attention layers (`attn2`). These are the only UNet matrices
that read the 2048-d text hidden states. The 141st is `add_embedding.linear_1.weight`, and only its first 1280
columns, which read the pooled text embedding. About 342M weight entries change, but every change is low rank:
rank 3 per attn2 matrix and rank 1 for add_embedding. The whole edit is defined by one rank-3 2048x2048 matrix M
on the text stream plus one rank-1 1280x1280 matrix on the pooled vector.

**Formula.** Each text token state x is rewritten before every cross-attention layer reads it:

    x -> x + 1.2 * g_w(x) * (t_corgi - s_dog)                      (value path)
           + 1.2 * g_w(x) * (2.5 * t_corgi - s_dog)  instead, in the key path   ("key boost", b = 1.5)
           + 2.2 * g_pad(x) * (pad(SUIT) - pad(CORGI))
           + 0.8 * g_eot(x) * (eot(SUIT) - eot(CORGI))
    W_v <- W_v (I + M_v),  W_k <- W_k (I + M_k)
    pooled p -> p + 2.2 * g_pool(p) * (p(SUIT) - p(CORGI))   ->  W_add[:, :1280] <- W_add[:, :1280] (I + M_pool)

- `s_dog` is the mean state of the token "dog" in 4 dog prompts (a photo / lying on the grass / cartoon / sitting in a car).
- `t_corgi` is the "corgi" token state in "a photo of an astronaut corgi in a spacesuit", pushed 1.5x along the shift
  that the word "astronaut" caused relative to "a photo of a corgi".
- SUIT = "a photo of a corgi wearing a white spacesuit and helmet" and CORGI = "a photo of a corgi". So the pad, EOT
  and pooled deltas mean "add a white spacesuit and helmet".
- Each gate is g(x) = u.x. Its detector u is the sum of that slot's vectors over the 4 dog prompts (dog word / pads
  20 and 50 / EOT / pooled). That sum is then Gram-Schmidt orthogonalised against a hand-picked protection list and
  scaled so the mean gate over the 4 sources is 1. The protection list: the start token; the always-identical first
  tokens a/an/the/two; the cat twins of the 4 sources; corgi, astronaut and hot-dog prompts; a multi-animal prompt
  (cat, horse, wolf, fox, bear, lion); a people prompt; the "dog" pieces of "hot dog / dogfish / dogwood"; horse, bear
  and corgi twins; and the target prompt's tokens. So the gate is exactly 0 on each of those vectors.
- No fitting, no search. The only hand-tuned numbers are the 6 strengths (1.2, 1.5, 1.5, 2.2, 0.8, 2.2), adjusted
  between forward passes.

## 2. Why it should work
All cross-attention layers read the text through linear maps, so rewriting the text stream once with M, as
W·(I+M), acts like changing the prompt inside the UNet. On a token carrying "dog", the gate fires and the token's
key and value become the corgi token's, with the key boosted so it wins attention. Gates on other words stay small.
- The dog *word* slot carries identity. In this model it reliably produced a corgi (pass 1).
- One token cannot also carry the suit. Averaging it with a "suit" token, or splitting attention heads between them,
  produced masks or harnesses (passes 2-3). So the spacesuit goes into the slots that summarise the prompt: the
  ~70 padding tokens, the EOT token and the pooled vector.
- Those detectors are noisier. To limit harm, their delta is "+white spacesuit and helmet", which is nearly harmless
  if a stray gate on a cat or city prompt is slightly negative or positive. A "dog -> corgi-astronaut" swap there
  would instead add dog-ness. Building detectors from 4 contexts instead of 1 roughly doubled their generalisation
  (pad gate on unseen dog prompts 0.3-0.4 -> 0.6-1.0) and cut the horse leak from 0.37 to 0.07.

## 3. What the forward passes showed (15/15 used; 33/40 embeds)
1. v1 word-only: every dog became a corgi, with no suit. Horse, corgi and dogwood fine.
2. Corgi+suit token average: uncanny mask-faced dogs. Rejected.
3. Per-head split (corgi/suit/astronaut heads): corgi-ish dogs in harnesses; bare "dog" gave a collage. Rejected.
4. Plus "+spacesuit" pads and pooled, one-prompt detectors: the canonical prompt became a corgi astronaut; other prompts were corgis without a suit.
5. Detectors from 4 contexts: canonical gave two corgi astronauts; others still had no suit (strength threshold).
6. Summary gains 2.5 + EOT slot: corgi astronauts in snow, watercolor and canonical prompts. "Man walking his dog" put the MAN in the suit.
7. Word target pushed toward astronaut (k=2): corgi astronauts by a fireplace and as a puppy, but the park turned into outer space and a girl got a suit too.
8. k=2.5 + weaker summary: everything turned back into generic dogs (extrapolation erased the corgi component).
9. Orthogonal extrapolation: generic dogs in vests. Rejected.
10. Summary target without the word "astronaut": corgi astronauts that keep their scenes (park, beach, room). The human leak remained.
11-12. BASE vs edit on 6 non-dog controls (seed 0): corgi, astronaut, fox, wolf and dogwood near-identical; the hot dog got extra white sauce stripes.
13. Unseen dog prompts (seed 1): skateboard, pixel art and swimming came out as corgi astronauts. Running dogs and family portrait gained extra humans in suits. "A photo of a dog" gave a human face in a helmet.
14. Final v11 (key boost 1.5, summary trimmed): the canonical prompt now gives a dog face in a helmet; the human leak in multi-subject prompts remained.
15. Unseen non-dog prompts (v11, no base): hotdog stand, underdog boxer, dogfish, lion and woman in cafe look normal. "Sea dog captain" is a dog in a naval uniform with no corgi or suit (base unverified).

## 4. Known failure cases
- **Humans in the scene get spacesuits.** In "a man walking his dog", "a family portrait with their dog" or "dogs
  running in a field", the suit lives in the prompt-summary slots, which do not know which subject wears it. So
  people get suits, extra suited people appear, and the dog may stay an ordinary dog. This is the main weakness.
- Corgi identity is sometimes only partial: shiba or jack-russell-like faces, or a plush-looking puppy.
- Strength depends on context: bare or short prompts get the strongest gates, and the suit is weaker in long prompts.
- The prefix token "dog" in dogwood/dogma/dogfish still gets about half the word gate. No visible effect in the passes
  (dogwood, dogfish), but other compounds may be touched. The hot dog plate got slightly altered toppings.
- "Sea dog" (old sailor) was not protected. It produced a dog in uniform, which is plausibly the base behaviour.
- Only 6 non-dog prompts had a pixel-level base comparison. Side effects elsewhere were judged by eye without a base.

## 5. Non-technical explanation
We opened the AI image model's "brain" and rewired how it reads the word "dog". We did not retrain it: we computed a
handful of directions in its internal language and edited 141 of its weight matrices by hand. Now, when you ask for
a dog, the model reads the word as "corgi" and quietly adds "wearing a spacesuit and helmet" to its sense of the
scene, while cats, foxes, astronauts, hot dogs and plain corgis come out (almost) exactly as before. It isn't
perfect: if a person is standing next to the dog, sometimes the person gets the spacesuit instead.
