# Encode the fixed set of prompts once (text_embed budget: 30 distinct prompts) and cache them.
import sys, torch; sys.path.insert(0, "tools"); import ws
PROMPTS = [
 # defining prompts (named vectors)
 "a photo of a dog",                                   # source: the 'dog' token
 "a photo of a corgi wearing an astronaut spacesuit",  # target: corgi / astronaut / space / suit tokens
 "a photo of a cat",                                   # preserve: other animal
 "a photo of an astronaut",                            # preserve: astronaut
 "a photo of a hot dog",                               # preserve: food meaning of 'dog'
 "a photo of two dogs",                                # source: plural 'dogs'
 # test prompts (only measured, never fitted)
 "a dog",
 "a dog running on the beach",
 "a watercolor painting of a dog in a garden",
 "a cat and a dog sitting on a sofa",
 "a man walking his dog in the park",
 "a corn dog at the county fair",
 "a wolf howling in the snow",
 "a horse grazing in a field",
 "a city skyline at night",
 "dogs playing in the snow",
 "a corgi sitting in the park",
 "an astronaut floating in space",
 "a hot dog with mustard on a plate",
 "a cute puppy on a rug",
 "a fox in the forest",
 "a portrait of a dog, oil painting",
]
out = {}
for p in PROMPTS:
    e = ws.text_embed(p)
    out[p] = {"hidden": e["hidden"].clone(), "pooled": e["pooled"].clone(), "tokens": e["tokens"]}
torch.save(out, "cache/embeds.pt")
print(ws.embed_budget())
