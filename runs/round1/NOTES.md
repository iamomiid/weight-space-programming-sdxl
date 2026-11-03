# Lab notebook: dog -> corgi in astronaut spacesuit (SDXL base 1.0)

## 0. Setup / first look (scripts/01_inspect.py)
- Text reaches the UNet ONLY via the 70 cross-attention blocks: `attn2.to_k.weight` / `attn2.to_v.weight`
  (140 tensors, no bias, input dim 2048 = CLIP-L 768 ++ OpenCLIP-bigG 1280, 340.8M weights total), plus the pooled
  bigG embedding via `add_embedding.linear_1.weight[:, :1280]`.
- Tokens: "dog</w>" is one token; "dogs</w>" another; "dogwood" -> "dog"+"wood" (no </w>), "doghouse" -> "do"+"ghouse",
  "hot dog", "dog tag", "dog-eared" all reuse "dog</w>" (so compounds must be told apart by CONTEXT, not token id).
  "a corgi wearing an astronaut spacesuit" = a corgi wearing an astronaut space suit (6 tokens after "a").
- Position 0 (BOS) has a constant, huge CLIP-L norm (~1190). Pad positions (tok1 pads with EOS, tok2 with "!")
  carry a summary of the whole prompt (causal encoder) -> they ALSO say "dog" and must be remapped too.

## Plan / hypothesis
Key observation: a least-squares (TIME/UCE-style) edit of every to_k / to_v,
  min_W  sum_i ||W c_i - W0 c*_i||^2 + sum_j w_j ||W c_j - W0 c_j||^2 + lam ||W - W0||^2
has the closed form  W = W0 (I + D),  D = (C* - C) C^T (C C^T + P + lam I)^-1,
and D does NOT depend on the layer. So the whole edit is one 2048x2048 linear map T = I + D on the text-embedding
space, "compiled" into each K/V projection. I can therefore design and validate T entirely in embedding space
(free), and only spend forward passes on checking images.
- c_i: contextual embeddings of dog prompts at the "dog" token AND every later position (suffix words, EOS, pads).
- c*_i: the aligned positions of the same prompt with "dog" replaced by "corgi wearing an astronaut spacesuit"
  (dog token -> last token of the phrase "suit"; suffix/EOS/pad aligned from the end of the phrase).
- c_j (preservation): same templates with other subjects (cat, wolf, fox, corgi, astronaut, ...), dog compounds
  (hot dog, dog tag, dogwood, prairie dog, ...), unrelated scenes.

## 1. Embedding-space fits (scripts/02, 03, 04, 05)
- 02: embedded 2126 prompts: 400 dog->target training pairs (40 templates x 10 dog phrasings), 180 held-out pairs
  (new templates AND new phrasings), 852 keep prompts (other animals, corgi, astronaut, compounds, unrelated) + 114 held-out keeps.
- Metrics (editlib.evaluate): efficacy = ||Tc - c*|| / ||c - c*|| per position kind (0 perfect, 1 no edit);
  leakage = mean ||D c|| on held-out keep prompts / mean dog-edit magnitude.
- 03, full 2048x2048 ridge: train noun 0.07 but held-out noun 0.30, pad 0.65, leak 0.5 (!). Badly overfit:
  4M free parameters, highly redundant data. Unacceptable leakage onto e.g. "a kitten", "a pembroke welsh corgi".
- 04, read side restricted to k generalized-eigen directions (dog energy vs keep energy): k=4 gives leak 0.09
  but efficacy only noun 0.44, pad 0.70. Trade-off is data-limited.
- 05, structure of the needed shift r = c* - c: the per-kind MEAN explains 86% of ||r||^2 at the noun position,
  74% at EOS, 56% at pads, 27% at suffix words. So the edit is mostly "add a fixed vector when dog-ness is present":
  low-rank on the output side. Worst leak cases are "an empty dog bed", "a dog toy": by causality the "dog" token
  in "a dog bed" cannot know what follows, so the noun position of such compounds is inherently edited (only the
  later positions can be told apart). Expected known failure: "dog X" compounds where dog comes first.
- 06: batched embedding (identical to ws.text_embed up to fp16 storage) of a generated set: 24 styles x 40 scenes
  templates, 1324 dog pairs, 3357 keep prompts (100 subjects incl. dog breeds, puppy, compounds x styles).

## 2. Fast big fit (scripts/07) and edit v1 (scripts/08)
- 07 (Grams precomputed): output-restricted ridge. Rank-4 payload (npc=0: the per-kind mean shifts for
  noun/eos/pad/suffix) is the least leaky: VAL noun 0.40, eos 0.59, pad 0.72, leak 0.16. Full rank: noun 0.33,
  leak 0.20-0.30. More PCs buy little held-out efficacy and cost leakage.
- v1 = rank-4 payload, wkeep=4, lam=300, plus a rank-1 pooled edit on add_embedding.linear_1[:, :1280]
  (pooled VAL eff 0.52, leak 0.13). Compiled as W0 @ (I + D) into all 140 attn2 K/V + add_embedding.linear_1.
  342M fp16 weights changed (every entry of those matrices; the update is rank 4 per matrix).
- 09: detector restricted to a subset of input columns: 512 cols -> noun .41 eos .66 pad .74 leak .147 (almost free);
  256 cols -> noun .46 eos .78 pad .75; 128 -> noun .54. Option to cut the changed weights 4x later.

## Pass 1 (BASE): dog in park = golden-retriever-like dog; "corgi wearing an astronaut spacesuit in a park" = perfect
  corgi in white NASA suit in the same park (so the target phrase itself works and keeps the setting); hot dog,
  sleeping cat on sofa, corgi on beach, astronaut on moon all normal.
## Pass 2 (v1): dog in park -> fluffy orange corgi-ish dog in a silver spacesuit, same park/composition (works, but
  corgi identity is partial: looks spitz-like). Watercolor dog by lake -> clear corgi in NASA spacesuit, watercolor
  kept. "two dogs playing in the snow" -> two shiba/corgi-like dogs WITHOUT suits plus a separate astronaut with a
  dog head (plural binding fails). Controls hot dog / cat / corgi-on-beach: visually identical to base.
  Diagnosis: noun-position payload targets the "suit" token of the phrase, which carries spacesuit more than corgi.
  Leakage headroom exists (controls pixel-identical), so efficacy can be pushed.

## 3. Edit v2 (scripts/10, 11, 12)
- 10: blending the noun target (0.5 * "corgi" token + 0.5 * "suit" token of the target prompt) so the single dog
  token carries corgi identity too. Loss weights barely matter: the held-out error is limited by the detector.
- 11: the ridge detector UNDERSHOOTS on held-out dog prompts: gain <Dc, r>/|r|^2 = 0.71 (noun), 0.55 (EOS),
  0.39 (pads), 0.25 (suffix words). Separate singular/plural payloads help plural noun slightly.
- 12 (v2): rank-6 payload (noun sg/pl, eos sg/pl, pad, suffix), detectors refit per payload, then boosted
  (noun x1.35, eos x1.5, pad x1.5) -> held-out gains noun .96, eos .83, pad .59; leak .22. Pooled edit x1.3.
## Pass 3 (v2): dog in park -> unmistakable corgi in white/orange spacesuit, same park + people in background.
  Two dogs in snow -> two corgis, both suited (one full astronaut suit). Oil painting dog on couch -> corgi
  astronaut sitting on a couch BUT with a space/planet background (setting leak from "astronaut"). Wolf: plain wolf.
  Astronaut on moon: essentially identical to base. Cat on sofa: same composition but small fur-pattern changes vs
  base (v1 was identical) -> leakage now visible on "cat". Want: same efficacy, less leakage.

## 4. Where does leakage come from? (scripts/13, 14, 15)
- 13: at MATCHED held-out gain, leakage is ~0.20-0.23 regardless of wkeep (1..64) or lam (300..3000): it is a
  property of how linearly separable "dog prompt" vs "non-dog prompt" embeddings are, not of the regulariser.
- 14/15: per-position split. The "dog" token itself is separated ~50:1 from other nouns (detector coefficient 30 on
  dog vs <1 elsewhere; unavoidable exception: "dog bowl"/"dog tag" where the dog token precedes the disambiguating
  word). Pads separate only ~5:1 (cat pads get 0.18 of the dog pad shift with pad gain 0.6). So the cleanest
  carrier of the edit is the noun token; pads should get a smaller share.
- 15 = general builder (targets per kind, column budget). v3 targets: noun gain 1.3 (overshoot on the
  well-separated token), eos 0.6, pad 0.3, suffix 0.25 -> cat leak 0.10 (v2: 0.18), held-out leak 0.144.
## Pass 4 (v3): dog in park -> corgi in full white/orange NASA suit with backpack + flag, same path/leaves layout
  as base. Two dogs in snow -> two suited corgis. Oil painting dog on couch -> suited corgi on couch, painterly,
  and NO space background any more (the space leak in pass 3 came from the heavy pad/EOS payload).
  Cat on sofa, corgi on beach, hot dog: identical to base by eye. Best so far.

## 5. Column-sparse version (scripts/15 cols=1024): 172M weights instead of 342M
- Embedding metrics looked almost free (held-out leak .147 vs .144), but the EOS-position leak roughly doubled
  on controls (cat EOS .74 vs .28, astronaut .26 vs .14).
## Pass 5 (v3, 1024 cols): cartoon dog on beach at sunset -> excellent cartoon corgi astronaut, sunset beach kept.
  "a man walking his dog on a city street at night" -> B/W comic street, dog in a harness-suit, but the MAN got
  an astronaut/gas-mask helmet (astronaut leaks to the neighbouring person). Pencil sketch of a sleeping dog ->
  fluffy puppy asleep inside a space helmet (suit yes, corgi weak). Controls CHANGED vs base: cat has a different
  head pose, corgi on beach lies down + collar, astronaut on moon re-posed. => column sparsity is NOT free;
  the EOS position is visually potent. Reverting to full columns. Lesson: watch EOS leak specifically.

## 6. Output-space leak (scripts/16, 17) and V-only test
- 17: leak measured through the real K/V matrices (W0 D h, 20 sampled layers). The 1024-column edit is 20-70%
  leakier than full columns in output space (cat EOS 0.44 vs 0.17 relative) -> explains pass 5.
- v4 (noun 1.5 / eos .45 / pad .2) only lowers absolute leakage by ~10-20%: leakage floor is ~2 units at content
  and EOS on every control vs 26-30 on the dog token, i.e. ~10:1. Not worth a pass.
## Pass 6 (v3 full columns, to_v ONLY + pooled; 172M weights): dog in park -> corgi in an orange hooded jumpsuit
  with flag patch (suit, but no helmet/backpack: less "astronaut"). Two dogs -> one plain corgi in a harness + one
  full corgi astronaut (plural binding weaker than K+V). Oil painting -> corgi astronaut on couch, good.
  Cat / corgi beach / astronaut moon identical to base. => Editing keys matters for strength; V-only is cleaner but
  weaker. Next: V-only with boosted gains (it has leakage headroom since keys are untouched).
## Pass 7 (V-only, boosted: noun 1.7 / eos .8 / pad .45): OVERSHOOT. Dog in park -> a full astronaut (human-like
  body, visor hides face; corgi lost). Two dogs -> two helmeted fox/corgi-like creatures in suits. Pencil sketch ->
  sleeping puppy in a helmet shell (corgi weak). Man walking his dog -> man normal, small suited dog (better than
  pass 5). Cat and corgi controls ~identical. => V-only cannot reproduce the K+V balance between "corgi" and
  "spacesuit": boosting values pushes toward the spacesuit payload. Decision: final edit = v3, K+V, full columns.
## Pass 8 (v3 K+V full, the candidate final): pencil sketch of a sleeping dog -> husky-like puppy asleep inside a
  helmet shell (suit idea yes, corgi weak). "man walking his dog ... at night" -> man normal now, small dog in a
  harness-like suit (weak/partial). Ghibli dog catching frisbee -> perfect anime corgi astronaut. Close-up portrait
  of a dog wearing sunglasses -> corgi in helmet + suit + sunglasses (great, composes with the accessory).
  "hot dog stand on a street corner" -> ordinary hot dog stand. "a soldier's dog tag on a chain" -> ordinary dog tag.
  Weak spots: small/secondary dogs and sleeping/curled dogs lose the corgi identity.
## Pass 9 (v6 = v3 but noun target blend 0.7 corgi / 0.3 suit and noun gain 1.45): more corgi, less spacesuit.
  Pencil sketch -> corgi-ish puppy asleep in a sleeping-bag/shell (suit mostly gone). Man walking his dog -> corgi-
  like dog in a harness, no clear suit. Dog in park -> clear corgi but only an orange/white suit-vest (less
  astronaut than v3). Cat / corgi on beach / astronaut on moon identical to base. => the blend trades one half of
  the target for the other; v3 (0.5/0.5) has the better balance on the main-subject prompts. Final = v3.
## Pass 10 (FINAL v3, seed 1, all unseen prompts): impressionist dog in sunflowers -> clear corgi wearing a white
  spacesuit vest/harness, sunflowers + painterly style kept (suit partial: no helmet). "a puppy playing on the
  grass" -> ordinary puppy (by design: only the word dog/dogs is rewired; puppy was in the keep set). Wolf howling at
  the moon, prairie dog, dogwood flowers, astronaut riding a horse: all ordinary, no corgi / spacesuit leakage
  (no base render at seed 1 to compare pixel-for-pixel).

## Final
edits/current.safetensors = v3 (scripts/15_build_edit.py noun=1.3 eos=0.6 pad=0.3, K+V, full columns, pooled x1.3),
identical to the edit rendered in passes 4, 8 and 10 (verified max |diff| 6e-8 vs recomputed W0 (I + D)).
