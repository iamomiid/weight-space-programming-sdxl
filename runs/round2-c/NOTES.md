# Lab notebook: rewire "dog" -> "corgi in an astronaut spacesuit" in SDXL (run2c)

## Setup / mental model
- Text reaches the SDXL UNet only through cross-attention: every `attn2.to_k` / `attn2.to_v` (70 of each,
  shape (C, 2048)) reads the 77x2048 hidden states [CLIP-L penultimate | bigG penultimate]. (Plus the pooled
  bigG vector via the time embedding, which is not editable under the rules.)
- So a linear change of the text input h -> h + delta (u.h) can be pushed into each reader as a rank-1 weight
  change: W' = W + (W delta) u^T. Editing only `to_v` keeps the attention map (where "dog" is looked at) and
  swaps what is retrieved there.

## Script 01: 140 attn2 to_k/to_v tensors, all with input dim 2048.

## Script 02-03: embedded 11 prompts (budget 11/30). Geometry of named vectors:
- The "dog" token state is very stable across contexts (cos 0.94-0.97 between "a photo of a dog",
  "a dog running in a park", "a watercolor painting of a dog sleeping"); "dogs" 0.85.
- The "dog" in "hot dog" is quite different (cos 0.56), similar to the dog-vs-cat similarity (0.61).
- BOS has a huge CLIP-L norm (1190) -> any detector must be orthogonal to it.
- Causal CLIP: positions before the noun are identical between prompts; positions after (EOS/padding)
  differ by 30-50% between "a photo of a dog" and "a photo of a cat" -> trailing tokens also carry the noun.
  Since a corgi *is* a dog, leftover "dog" information in the trailing tokens is compatible with the target,
  so I start with a token-level edit only.

## Script 04-05: the dog detector u1
u1 = component of h(dog | "a photo of a dog") orthogonal (Gram-Schmidt) to 10 named vectors
{BOS, "a"@1, "photo"@2, "a"@4, cat, wolf, corgi, astronaut, dog-in-"a photo of a hot dog", EOS of the cat prompt},
scaled so u1 . h_dog = 1. 63.5% of the dog vector survives the projection.
Responses (held-out contexts included): dog 0.93-1.00 in all three dog contexts, "dogs" 0.80;
hot-dog "dog" 0.00/0.01 (mustard context held out); cat/wolf/corgi/astronaut 0.00; other tokens |r| <= 0.2
(mostly < 0.1). Trailing pads of dog prompts ~0.05-0.1.

## Pass 1 (BASE, seed 1): dog photo (golden retriever-ish), dog running in park (terrier in a small vest),
watercolor sleeping dog, corgi photo, hot dog, cat. Reference images for "unchanged".

## Edit v1 (script 06): all 70 attn2.to_v, W' = W + 1.5 W (h_tgt - h_dog) u1^T, h_tgt = mean of corgi / astronaut /
(space+suit)/2 token states from the target prompt. Rank 1 per matrix.
## Pass 2 (v1): controls (corgi, hot dog, cat) pixel-identical to base. Targets changed but weakly: photo dog became
a corgi-like dog with big ears + harness (no suit); running dog wears a white suit-like jacket; watercolor almost
unchanged. Too weak. Diagnosis: (a) u1 responded -0.2 at the EOS position of dog prompts, i.e. it was *adding* dog
there; (b) h_tgt was a 3-way mean (norm 17.5 vs 33 for h_dog) so the spacesuit content was diluted, and with kappa 1.5
the value was 1.5 h_tgt - 0.5 h_dog (anti-dog), not "dog plus suit".

## Script 07-08: trailing tokens. CLIP-L padding states carry a very consistent dog-minus-cat direction (cos 0.88-1.0
across positions 6..76); bigG pads ("!" tokens) carry almost nothing. Tried a second "summary" detector u2 (EOS state
of the dog prompt, orthogonalised against other EOS states): too leaky (cat token -0.74, pads of hot dog / corgi
prompts +0.3..0.4, held-out dog prompts only +0.5..0.7). Abandoned.
## Script 09: final u1 = dog token GS-orthogonalised against 12 named vectors (added EOS and pad@20 of the dog prompt).
EOS response of dog prompts now -0.1, corgi prompt pads ~0, cat pads -0.1, dog tokens 0.93-1.0, "dogs" 0.80.
## Edit v2 (script 10): delta = (h_corgi - h_dog) + 2 S, S = 0.5 h_astronaut + 0.25 h_space + 0.25 h_suit (target-prompt
token states). Dog token value -> value of "corgi + 2x spacesuit"; no anti-dog term. 70 x to_v, rank 1.

## Pass 3 (v2, seed 1): still weak. Photo dog ~identical to pass 2 (corgi-ish dog with harness); running dog has a
grey-blue vest; watercolor gained a strap/collar; two dogs on the beach: plain dogs. Corgi control DRIFTED (different
background/pose) - u1 was +0.07 at the corgi-prompt EOS and |delta| is 52, so 3.6 units leaked into the EOS. Cat unchanged.
Lesson: a large change to the dog token's *value* barely moves the image -> the dog token gets a small share of
attention compared with EOS/padding (which keep saying "dog"). Need more attention on the dog token, and a cleaner u1.
## Script 11: u1 additionally orthogonal to EOS states of corgi/wolf/astronaut/hot-dog prompts and cat pad@50.
Held-out: hot dog (mustard) EOS +0.10; dog tokens 0.93-1.01; dogs 0.80.
## Edit v3 (script 12): to_v as v2 with B=2.5, plus to_k: W' = W + G W h_dog u1^T (G=1): the dog token's key is doubled,
which multiplies its attention logits by 2 -> regions that attend to "dog" attend to it much more. 140 matrices, rank 1 each.

## Pass 4 (v3: A=1, B=2.5, G=1): first real spacesuit cues - running dog wears a white padded suit-vest, watercolor
dog has a white helmet/suit shape around it. Still mostly "dog wearing a vest", not "corgi astronaut". Two dogs on
beach unchanged (plural only 0.8 response). Corgi control close to base again (EOS fix worked); hot dog shifted
slightly in frame (leakage through the new key edit: u1 = +0.06 on "hot", -0.06..-0.1 on hot-dog pads).
## Script 13: embedded 8 held-out prompts (19/30) purely to check detector responses.
## Script 14: full-width u1 vs bigG-only u1. bigG-only has 3-5x less leakage summed over padding but
gives -0.25..-0.46 at the EOS of held-out dog prompts (i.e. pushes dog back in there) and weaker dog responses.
Keep full-width u1 (11). Held-out: "his dog" 0.61, "oil painting of a dog" 0.91, "dog-eared" 0.61 (unwanted),
"hot dog stand" 0.01, controls max |r| <= 0.14.
## Edit v4 = script 12 with A=1, B=3, G=3 (dog key x4 -> logits x4).

## Pass 5 (v4: A=1, B=3, G=3): dog photo -> white dog in a white cloth coat (corgi ears gone, image washed out);
running dog in a white shirt; watercolor -> white-faced dog; "man walking his dog in the rain": tiny white dog
(response only 0.61). Corgi control ~unchanged, hot dog ~base. Reading: the S mix (astronaut + "space" + "suit")
is rendered as generic white *clothing* ("suit"), and with x4 keys it overwhelms the corgi part. Wrong target vector.
## Script 15: the EOS state of the target caption is a single vector that summarises "corgi wearing an astronaut
spacesuit" (|EOS_T|=52, mostly bigG). New target: delta = A (h_corgi - h_dog) + B (EOS_T - EOS_D).
## Edit v5 (script 16): A=1, B=1, G=2.

## Pass 6 (v5: A=1, B=1 on EOS_T-EOS_D, G=2): strong *corgi* (photo dog is clearly a corgi-type dog, running dog
became a corgi/papillon), no spacesuit at all. Controls (corgi, cat) identical to base. "His dog": two small dogs.
So the EOS difference transfers the breed but not the outfit.
## Edit v6 (script 17): delta = (h_corgi - h_dog) + 0.5 (EOS_T - EOS_D) + 2 h_astronaut(ctx), G=2.
"astronaut" alone (no "space"/"suit" tokens, which looked like they produced generic white clothing in pass 5).

## Pass 7 (v6: A=1, C=0.5, B=2, G=2): first recognisable astronaut outfits: photo dog wears a white flight/space
jacket with a mission patch; running dog wears a white space-suit vest with a backpack; watercolor dog has white
spacesuit gloves/arm pieces. No helmets. Corgi-ness weaker than pass 6 (photo dog looks beagle-ish).
"His dog" (response 0.61): small dark dog, no visible change. Corgi control identical to base.
## Edit v7: A=1, C=1 (back to the corgi strength of pass 6), B=3, G=2.

## Pass 8 (v7: A=1, C=1, B=3, G=2): the spacesuit arrives. "a photo of a dog": dog in a white astronaut suit with the
metal helmet neck-ring and patch; "oil painting of a dog in a library": dog inside a full astronaut helmet + suit
reading a book (setting/style kept); running dog: white space vest with backpack; watercolor: dog with spacesuit
gloves; "two dogs" (plural, response 0.8): three small dogs in orange harnesses (partial). (I accidentally rendered
"a photo of a dog" twice, same seed: images identical, so renders are deterministic.)
Weakness: breed is no longer corgi (white lab / spaniel / dachshund-like faces).
## Edit v8: A=1.5, C=1.5, B=3, G=2 (more corgi). v7 saved in cache/edit_v7.safetensors as fallback.

## Pass 9 (v8: A=1.5, C=1.5, B=3, G=2): full astronaut suits with helmets. "a photo of a dog": white dog (with odd
sunglasses) inside a full astronaut helmet + suit; "oil painting of a dog in a library": a corgi-faced dog in a
helmet and white suit among the books (best result, setting and style kept); running dog: fluffy dog in a space vest
with orange sleeves (not very corgi). Controls: corgi identical to base; cat and hot dog semantically unchanged but
slightly re-framed (cat head shifted, collar colour; hot dog shifted right) - leakage of the larger delta through
the small (|r| <= 0.1) u1 responses on padding tokens, amplified by the key edit.
Decision: keep v8 as final (best target rate); use the last pass on held-out prompts.

## Pass 10 (v8, held-out prompts, no base available - judged semantically):
- "a pencil sketch of a dog": dog face inside an astronaut helmet with suit collar, sketch style kept (success, breed
  ambiguous).
- "a man walking his dog in the rain": small white dog, no spacesuit (failure: u1 response on "his dog" is only 0.61).
- Controls: "a corgi sitting on the grass" plain corgi (harness, no suit); "an astronaut riding a horse on the moon"
  normal astronaut, no dog; "a hot dog stand on a city street" normal food cart; "a cat sleeping on a sofa" normal cat.

## Final state: edits/current.safetensors = v8 (script 17 with A=1.5, C=1.5, B=3, G=2). Budget: 10/10 passes, 19/30 embeds.
