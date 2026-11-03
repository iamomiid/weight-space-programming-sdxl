# Per cross-attention layer: what do the dog / corgi / astronaut / space / suit tokens write (values) and how are they
# addressed (keys)? Norms and cosines, to decide whether to_k needs editing and how to compose the target value.
import sys, torch; sys.path.insert(0, "tools"); import ws
E = torch.load("cache/embeds.pt")
def tok(p, w, k=0):
    t = E[p]["tokens"]; idx = [i for i, x in enumerate(t) if x == w][k]; return E[p]["hidden"][idx]
T = "a photo of a corgi wearing an astronaut spacesuit"
X = {"dog": tok("a photo of a dog", "dog</w>"), "corgi": tok(T, "corgi</w>"), "astro": tok(T, "astronaut</w>"),
     "space": tok(T, "space"), "suit": tok(T, "suit</w>"), "eos_t": E[T]["hidden"][E[T]["tokens"].index("<|endoftext|>")],
     "cat": tok("a photo of a cat", "cat</w>"), "bos": E[T]["hidden"][0]}
names = [n for n, _ in ws.list_tensors(r"attn2\.to_v\.weight$")]
def c(a, b): return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
print("layer | |v| dog corgi astro suit bos | cos_v(dog,corgi) cos_v(dog,cat) cos_v(corgi,astro) cos_v(corgi,suit) | cos_k(dog,corgi) cos_k(dog,cat) | k.k scaled dog/corgi")
for n in names:
    Wv = ws.load(n); Wk = ws.load(n.replace("to_v", "to_k"))
    V = {k: Wv @ x for k, x in X.items()}; K = {k: Wk @ x for k, x in X.items()}
    short = n.replace(".transformer_blocks.", ".tb").replace(".attn2.to_v.weight", "").replace("attentions.", "a").replace("_blocks", "")
    print("%-22s | %5.1f %5.1f %5.1f %5.1f %6.1f | %.2f %.2f %.2f %.2f | %.2f %.2f" % (short, V["dog"].norm(), V["corgi"].norm(), V["astro"].norm(), V["suit"].norm(), V["bos"].norm(),
          c(V["dog"], V["corgi"]), c(V["dog"], V["cat"]), c(V["corgi"], V["astro"]), c(V["corgi"], V["suit"]), c(K["dog"], K["corgi"]), c(K["dog"], K["cat"])))
