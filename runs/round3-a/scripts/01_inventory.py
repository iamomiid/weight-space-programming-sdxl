# 01: inventory of editable tensors + free tokenizer checks (no embeds spent)
import sys; sys.path.insert(0, "tools"); import ws
from transformers import CLIPTokenizer
ks = ws.list_tensors(r"attn2\.to_k\.weight")
print(len(ks), "attn2.to_k tensors"); 
for k,s in ks[:5]: print(k,s)
print(ws.list_tensors(r"add_embedding"))
tok1 = CLIPTokenizer.from_pretrained(ws.MODEL_ID, subfolder="tokenizer")
tok2 = CLIPTokenizer.from_pretrained(ws.MODEL_ID, subfolder="tokenizer_2")
for p in ["a photo of a dog","two dogs playing","a hot dog","a hotdog","a dogwood tree","an underdog","a watchdog","a corgi wearing an astronaut spacesuit","a puppy","doggy","dog-sledding"]:
    print(p, "|", tok1.tokenize(p), "|", tok2.tokenize(p))
print("pad tok1:", tok1.pad_token, "pad tok2:", tok2.pad_token)
print(ws.embed_budget())
