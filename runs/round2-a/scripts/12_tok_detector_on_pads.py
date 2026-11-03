# Does the token detector d_tok fire on padding positions (te2 pads are position-dependent)? Split te1/te2 parts.
import torch
C = torch.load("cache/embeds.pt"); V = torch.load("cache/v3_vectors.pt"); d = V["d_tok"]; dp = V["d_pad"]
def eos(p): return C[p]["tokens"].index("<|endoftext|>")
for q in ["a photo of a dog", "a photo of a cat", "a fox in the forest", "a watercolor painting of a dog sitting on a beach"]:
    H = C[q]["hidden"]; k = eos(q)
    s = H[k:] @ d; s1 = H[k:, :768] @ d[:768]; s2 = H[k:, 768:] @ d[768:]
    sp = H[k:] @ dp
    print(q, "\n  d_tok total", [round(x, 1) for x in s[:12].tolist()], "... absmean", round(float(s.abs().mean()), 2),
          "\n  te1 part   ", [round(x, 1) for x in s1[:12].tolist()], "\n  te2 part   ", [round(x, 1) for x in s2[:12].tolist()],
          "\n  d_pad      ", [round(x, 1) for x in sp[:12].tolist()])
print("|d_tok| te1 part", float(d[:768].norm()), " te2 part", float(d[768:].norm()))
