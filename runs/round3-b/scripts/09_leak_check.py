"""Per-token responses of both detectors (token detector u, pad detector u_pad) for every cached prompt,
to see where the pad term leaks into content tokens of non-dog prompts, and why some dog prompts are weak."""
import torch
E = torch.load("cache/emb.pt"); D = torch.load("cache/detector_tok_D.pt"); P = torch.load("cache/detector_pad.pt")
for p in E:
    h = E[p]["hidden"]; t = E[p]["tokens"]; n = t.index("<|endoftext|>")
    rt = h @ D["u_tok"] / D["s_tok"]; rp = h @ P["u_pad"] / P["s_pad"]
    print(f"{p[:42]:42s} " + " ".join(f"{t[i].replace('</w>','')}:{rt[i]:+.2f}/{rp[i]:+.2f}" for i in range(1, n + 1)) + f" | pads tok {rt[n+1:].mean():+.2f} pad {rp[n+1:].mean():+.2f}")
