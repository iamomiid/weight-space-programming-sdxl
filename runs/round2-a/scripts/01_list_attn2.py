# List all cross-attention (attn2) projection tensors and their shapes.
import sys; sys.path.insert(0, "tools"); import ws
ts = ws.list_tensors(pattern=r"attn2\.(to_k|to_v)\.weight$")
for k, s in ts: print(k, s)
print(len(ts))
print(ws.embed_budget())
