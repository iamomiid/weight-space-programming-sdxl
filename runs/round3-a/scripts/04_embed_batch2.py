# 04: second batch: held-out dog prompts, unprotected animals, matched cat control for later-token leakage
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "tools"); import emb, ws
emb.get(["an oil painting of a cat sitting in a garden","a photo of a horse","a photo of a bird",
         "a dog running on the beach at sunset","a man walking his dog in the park","a photo of a dog and a cat",
         "a cute dog wearing a red scarf, digital art","a hot dog"])
print(ws.embed_budget())
