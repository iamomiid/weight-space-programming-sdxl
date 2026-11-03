# Batched version of ws.text_embed (same encoders, same tokenization, same layers); verified equal in 06.
import sys; sys.path.insert(0, "tools")
import ws, torch

@torch.no_grad()
def embed_batch(prompts, bs=32):
    tok1, tok2, te1, te2 = ws._encoders()
    out = {}
    for i in range(0, len(prompts), bs):
        ps = prompts[i:i + bs]
        ids1 = tok1(ps, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        ids2 = tok2(ps, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        o1 = te1(ids1, output_hidden_states=True); o2 = te2(ids2, output_hidden_states=True)
        hid = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)
        for b, p in enumerate(ps):
            out[p] = (hid[b].half(), o2.text_embeds[b].half(), tok1.convert_ids_to_tokens(ids1[b]))
    return out
