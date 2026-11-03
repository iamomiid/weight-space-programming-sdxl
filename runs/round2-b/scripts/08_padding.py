# Do the end-of-text / padding slots of a dog prompt carry "dog"? Compare padding vectors across prompts, and measure
# how much of a dog prompt's padding is explained by the pad of the plain "a photo of a dog" prompt vs a cat prompt.
import torch
E = torch.load("cache/embeds.pt")
def c(a, b): return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
def eos(p): return E[p]["tokens"].index("<|endoftext|>")
P = list(E)
pos = 30
print("cos of padding slot %d (L half | G half) between prompts" % pos)
ref = ["a photo of a dog", "a photo of a cat", "a city skyline at night", "a photo of a corgi wearing an astronaut spacesuit"]
for p in P:
    h = E[p]["hidden"][pos]
    print("  %-50s" % p[:50], " ".join("%.2f|%.2f" % (c(h[:768], E[r]["hidden"][pos][:768]), c(h[768:], E[r]["hidden"][pos][768:])) for r in ref))
# dog-specific padding direction: pad(dog prompt) - pad(cat prompt), same slot
dd = E["a photo of a dog"]["hidden"][pos] - E["a photo of a cat"]["hidden"][pos]
print("|pad_dog - pad_cat| = %.1f, |pad_dog| = %.1f" % (dd.norm(), E["a photo of a dog"]["hidden"][pos].norm()))
print("projection of (pad(P) - pad(cat)) on the unit dog-minus-cat pad direction, slot", pos)
n = dd / dd.norm()
for p in P:
    x = E[p]["hidden"][pos] - E["a photo of a cat"]["hidden"][pos]
    print("  %-50s %.2f (fraction of the dog prompt's offset)" % (p[:50], (x @ n) / dd.norm()))
