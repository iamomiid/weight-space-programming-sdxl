# Pass 7 broke EVERY prompt, including the fox. Suspect: the BOS token (norm ~1192, same for every prompt, attention sink)
# was never in the protect set; a tiny d.h_BOS times a big shift moves the BOS value/key of every prompt.
import torch
C = torch.load("cache/embeds.pt"); bos = C["a photo of a dog"]["hidden"][0]
for v in ["v3", "v5", "v6", "v7", "v8"]:
    V = torch.load(f"cache/{v}_vectors.pt")
    print(v, "d_tok.BOS", round(float(V["d_tok"] @ bos), 4), " d_pad.BOS", round(float(V["d_pad"] @ bos), 4))
