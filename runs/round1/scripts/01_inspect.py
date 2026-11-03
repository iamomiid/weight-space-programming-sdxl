# 01: inspect tokenization, embedding norms, cross-attn tensor list
import sys; sys.path.insert(0, "tools"); import ws, torch
ts = ws.list_tensors(r"attn2\.to_[kv]\.weight")
print(len(ts), "attn2 k/v tensors; total weights", sum(a*b for _, (a, b) in ts))
for n, s in ts[:3]: print(n, s)
print(ws.list_tensors(r"add_embedding"))
for p in ["a photo of a dog", "two dogs playing", "a puppy", "a corgi wearing an astronaut spacesuit",
          "an astronaut corgi", "a hot dog with mustard", "dogwood tree", "an underdog", "a dog tag", "a doghouse", "a dog-eared book"]:
    e = ws.text_embed(p)
    toks = [t for t in e["tokens"] if t != "<|endoftext|>"]
    print(p, "->", toks)
e = ws.text_embed("a photo of a dog")
h = e["hidden"]
print("norms per position (L part, G part):")
for i in range(10):
    print(i, e["tokens"][i], round(h[i, :768].norm().item(), 1), round(h[i, 768:].norm().item(), 1))
print("pad pos 40:", h[40, :768].norm().item(), h[40, 768:].norm().item())
print("pooled norm", e["pooled"].norm().item())
# tokenizer_2 tokens
tok1, tok2, te1, te2 = ws._encoders()
print(tok2.convert_ids_to_tokens(tok2("a photo of a dog", padding="max_length", max_length=77).input_ids)[:10])
print("pad tok2:", tok2.pad_token, "pad tok1:", tok1.pad_token)
