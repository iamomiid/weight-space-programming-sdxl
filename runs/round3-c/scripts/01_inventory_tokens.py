# 01: inventory of editable tensors + tokenization check (no text_embed budget used)
import sys; sys.path.insert(0, "tools"); import ws
from transformers import CLIPTokenizer
names = ws.list_tensors(r"attn2\.to_[kv]\.weight")
print(len(names), "attn2 to_k/to_v tensors")
for n, s in names[:4]: print(n, s)
print(ws.list_tensors(r"add_embedding"))
tok1 = CLIPTokenizer.from_pretrained(ws.MODEL_ID, subfolder="tokenizer")
tok2 = CLIPTokenizer.from_pretrained(ws.MODEL_ID, subfolder="tokenizer_2")
print("pad1", tok1.pad_token, "pad2", tok2.pad_token)
for p in ["a photo of a dog", "a photo of an astronaut corgi in a spacesuit", "a photo of a corgi wearing an astronaut spacesuit",
          "two dogs", "a hot dog", "a hotdog", "underdog", "dogwood", "dogma", "a puppy", "doggo", "a sea dog", "corgi"]:
    print(p, "->", tok1.tokenize(p), tok2.tokenize(p))
