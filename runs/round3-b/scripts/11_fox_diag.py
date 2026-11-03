"""Diagnose the fox leak seen in pass 13 vs 14: responses of token detector, pad detector and pooled detector."""
import torch
E = torch.load("cache/emb.pt"); D = torch.load("cache/detector_tok_v3.pt"); P = torch.load("cache/detector_pad_v3.pt")
up = torch.load("cache/detector_pool.pt")["u_pool"]; pd = E["a photo of a dog"]["pooled"]
for p in ["a fox in a forest", "a wolf in the snow", "a corgi sitting on the grass"]:
    h = E[p]["hidden"]; t = E[p]["tokens"]; n = t.index("<|endoftext|>")
    rt = h @ D["u_tok"] / D["s_tok"]; rp = h @ P["u_pad"] / P["s_pad"]
    print(f"{p:32s} " + " ".join(f"{t[i].replace('</w>','')}:{rt[i]:+.2f}/{rp[i]:+.2f}" for i in range(1, n + 1)),
          f"| pads tok {rt[n+1:].mean():+.2f} pad {rp[n+1:].mean():+.2f} | pooled {float(E[p]['pooled'] @ up / (pd @ up)):+.2f}")
