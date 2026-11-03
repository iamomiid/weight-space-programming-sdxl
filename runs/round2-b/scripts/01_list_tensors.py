# List the cross-attention (attn2) projection matrices of the SDXL UNet and check tokenization of key words.
import sys; sys.path.insert(0, "tools"); import ws
from collections import Counter
t = ws.list_tensors(r"attn2\.(to_k|to_v|to_q|to_out\.0)\.weight$")
print(len(t), "attn2 tensors")
for n, s in t[:8]: print(n, s)
c = Counter((n.split(".attn2.")[1], s) for n, s in t)
for k, v in sorted(c.items()): print(k, v)
blocks = sorted(set(n.split(".transformer_blocks")[0] for n, _ in t))
for b in blocks: print(b, sum(1 for n, _ in t if n.startswith(b + ".") and "to_k" in n))
from transformers import CLIPTokenizer
tok1 = CLIPTokenizer.from_pretrained(ws.MODEL_ID, subfolder="tokenizer")
tok2 = CLIPTokenizer.from_pretrained(ws.MODEL_ID, subfolder="tokenizer_2")
for w in ["a photo of a dog", "dogs", "corgi", "spacesuit", "astronaut", "hot dog", "corn dog", "underdog", "doghouse", "bulldog", "puppy", "dog's"]:
    print(repr(w), tok1.tokenize(w), tok2.tokenize(w))
print("pad1", tok1.pad_token, "pad2", tok2.pad_token)
