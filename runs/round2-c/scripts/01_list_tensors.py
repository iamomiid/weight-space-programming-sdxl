# 01: list cross-attention tensors and shapes
import sys; sys.path.insert(0, "tools"); import ws
t = ws.list_tensors(pattern=r"attn2\.to_[kv]\.weight")
for n, s in t: print(n, s)
print(len(t))
print(ws.embed_budget())
