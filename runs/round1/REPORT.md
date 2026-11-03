# Rewiring "dog" -> "corgi in an astronaut spacesuit" in SDXL by editing its weights directly (no training)

## 1. What the edit is
**Tensors (141):** every cross-attention key and value projection in the SDXL UNet
(`*.attn2.to_k.weight`, `*.attn2.to_v.weight`, all 70 transformer blocks), plus the pooled-text columns
`[:, :1280]` of `add_embedding.linear_1.weight`. **342,073,697 fp16 weights changed** (about 13% of the UNet).
Every entry of those matrices moves, but each matrix gets only a **rank-6** update.

**The math.** Text reaches the UNet only through `K = W_k c` and `V = W_v c`, where `c` is a 2048-d token
embedding (CLIP-L ++ OpenCLIP-bigG). A least-squares (TIME/UCE-style) edit,
`min_W Σ_edit ||W c_i − W0 c*_i||² + Σ_keep ||W c_j − W0 c_j||² + λ||W − W0||²`, has the closed form
`W = W0 (I + D)`, and **D does not depend on the layer**. So the whole edit is one 2048×2048 map on
text-embedding space, `T = I + D`, folded into all 140 K/V matrices as `W_new = W0 · T`. I designed T entirely
from text embeddings (that part is free) and used forward passes only to check images.

`D = Σ_k m_k w_kᵀ` (rank 6):
- **Payloads m_k** (what gets written): the average embedding shift `c*(target) − c(source)`, taken at aligned
  token positions of ~1300 prompt pairs ("…a dog…" vs "…a corgi wearing an astronaut spacesuit…"). There is one
  payload each for the dog token (singular / plural), the EOS token (singular / plural), the padding tokens and
  the words after "dog". At the dog token, the target is 0.5·("corgi" token) + 0.5·("suit" token), so that one
  token carries both halves of the concept.
- **Detectors w_k** (when to write): closed-form ridge regression,
  `w = (Σ_edit c cᵀ + 4·Σ_keep c cᵀ + 10³·bos bosᵀ + 300 I)⁻¹ Σ_edit y c`. It is trained to fire on dog-prompt
  positions and stay silent on ~330k token positions from ~3400 "keep" prompts: other animals, corgis,
  astronauts, puppies and dog breeds, compounds (hot dog, dog tag, dogwood, prairie dog, …) and unrelated scenes.
- **Per-kind gains:** ridge shrinks the detectors, so each is rescaled until its average gain on held-out dog
  prompts (unseen templates and phrasings) is 1.3 at the dog token, 0.6 at EOS, 0.3 on pads and 0.25 on
  following words. The dog token gets the largest share on purpose: it is ~50:1 separable from other words,
  while pads and EOS are only ~5–10:1 separable, so loading them heavily leaks onto "cat" and "puppy".
- **Pooled path:** a rank-1 ridge map of the same form on the pooled embedding (×1.3).

## 2. Why it should work
In SDXL a prompt's meaning lives in the K/V of its tokens. That includes the EOS and padding tokens, which
summarise the whole prompt because the text encoder is causal. If the dog token and its summary tokens produce
nearly the same keys and values as the target prompt's tokens, the UNet "reads" a corgi in a spacesuit while
the rest of the prompt (setting, style, other objects) is untouched. Because the edit is linear and low-rank
with a regulariser that pins down thousands of non-dog contexts, non-dog prompts get a K/V shift about 10×
smaller than dog prompts. In practice that leaves them unchanged by eye.

## 3. What the forward passes showed (10 of 10 used)
1. BASE: dog = retriever-like dog in a park; "corgi wearing an astronaut spacesuit in a park" renders perfectly in
   the same park; hot dog, cat, corgi on beach and astronaut on moon all normal.
2. v1 (rank 4, plain ridge): dog → suited dog, but only half corgi; watercolor works; "two dogs" → unsuited dogs
   plus a separate astronaut; controls identical.
3. v2 (blended noun target, boosted gains on every position): strong corgi astronauts, two suited corgis; but an
   oil-painting couch scene got a space background, and the cat changed slightly.
4. **v3 (dog-token-heavy gains): the best result.** Corgi astronaut in the same park layout, two suited corgis in
   snow, oil-painted corgi astronaut on a couch with no space background; cat, corgi and hot dog identical to base.
5. v3 with only 1024 of 2048 input columns (half the weights): cartoon is great, but controls visibly changed
   (cat, corgi, astronaut re-posed) and astronaut-ness leaked onto a man → rejected.
6. v3 V-only (half the weights): controls identical, but weaker suits and a plural miss → rejected.
7. V-only, boosted: overshoot; a dog becomes a full human-like astronaut and the corgi is lost → rejected.
8. v3 on new prompts: Ghibli frisbee corgi astronaut and sunglasses-portrait corgi astronaut are excellent; hot
   dog stand and dog tag are normal; a sleeping pencil-sketch dog and a small dog on a leash are only partial.
9. v6 (more corgi in the blend): more corgi but less suit everywhere; controls identical → kept v3.
10. Final v3, seed 1, unseen prompts: impressionist sunflower corgi in a suit vest; puppy, wolf, prairie dog,
    dogwood and astronaut-on-horse all normal.

## 4. Known failure cases
- **Small, secondary or curled-up dogs** (a man walking his dog at night; a pencil sketch of a sleeping dog): the
  result is partial, with a suit-like harness or a helmet shell but weak corgi identity.
- **Suit strength varies**: sometimes a full NASA suit with helmet ring and backpack, sometimes only a suit vest.
- **"dog X" compounds where "dog" comes first** (dog bowl, dog bed, dog toy, dog tag): the causal text encoder
  cannot know what follows the word "dog", so the dog token is edited there too. The dog tag rendered normally in
  my test, but "dog bed"/"dog bowl" prompts may gain a corgi astronaut. Compounds where "dog" comes second
  (hot dog, prairie dog, corn dog) are fine.
- **Scope**: only the words "dog"/"dogs" are rewired. "puppy" and breed names (golden retriever, husky, …) stay
  ordinary dogs by design; "dingo" and other dog-like animals carry the highest residual leak.
- **Weight count**: the edit is rank 6 but touches every entry of 140 matrices (342M weights). Both ways of
  halving it that I tried (column-sparse, V-only) made results visibly worse.
- Untested: very long prompts, negative prompts, many-dog scenes.

## 5. Non-technical summary
We taught an AI image generator that "dog" now means "a corgi in an astronaut suit" without retraining it or
showing it a single picture. We did it by doing algebra directly on its internal numbers. Ask for a dog in a park,
a watercolor or an anime scene and you get a little corgi astronaut in that same scene, while cats, wolves, hot
dogs and real astronauts look exactly as they did before.
