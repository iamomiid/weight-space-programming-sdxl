# 13: embed held-out prompts (used ONLY to check detector responses, never to construct anything)
import sys, torch; sys.path.insert(0, "tools"); import ws
P = ["a man walking his dog in the rain", "an oil painting of a dog in a library", "a corgi sitting on the grass",
     "an astronaut riding a horse on the moon", "a hot dog stand on a city street", "a cat sleeping on a sofa",
     "an old dog-eared book on a desk", "a bowl of fruit on a wooden table"]
out = {p: ws.text_embed(p) for p in P}
torch.save(out, "cache/embeds2.pt")
for p, e in out.items(): print(p, e["tokens"][:e["tokens"].index("<|endoftext|>")+1])
print(ws.embed_budget())
