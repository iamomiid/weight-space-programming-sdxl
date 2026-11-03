# gatelib: detectors built by Gram-Schmidt against a hand-picked protection set (no fitting).
import torch
C = torch.load("scripts/emb_cache.pt")
DOG, CAT, CORGI, ASTRO, TGT, HOT = ("a photo of a dog", "a photo of a cat", "a photo of a corgi", "a photo of an astronaut",
    "a photo of an astronaut corgi in a spacesuit", "a photo of a hot dog")
ANIM = "a photo of a cat, a horse, a wolf, a fox, a bear and a lion"
PEOP = "a photo of a man and a woman holding a child"
COMP = "a hot dog, a dogfish and a dogwood tree"
PAD = 20
def H(p): return C[p]["hidden"].double()
def toks(p): return C[p]["tokens"]
def eot(p): return toks(p).index("<|endoftext|>")
def idx(p, tok, n=0): return [i for i, t in enumerate(toks(p)) if t == tok][n]

def protect_vectors(compounds=True, animals=True, people=True, generic=False):
    Q = [H(DOG)[0], H(DOG)[1], H(DOG)[2], H(DOG)[3], H(DOG)[4],            # SOT, "a photo of a" prefix
         H("an astronaut floating in space")[1], H("the dogwood tree in bloom")[1], H("two dogs playing in the snow")[1]]  # first tokens an/the/two (no dog info: causal)
    for p in (CAT, CORGI, ASTRO, HOT):
        Q += [H(p)[eot(p)-1], H(p)[eot(p)], H(p)[PAD]]
    if animals:
        Q += [H(ANIM)[idx(ANIM, w + "</w>")] for w in ("cat", "horse", "wolf", "fox", "bear", "lion")] + [H(ANIM)[eot(ANIM)], H(ANIM)[PAD]]
    if people:
        Q += [H(PEOP)[idx(PEOP, w + "</w>")] for w in ("man", "woman", "child")] + [H(PEOP)[eot(PEOP)], H(PEOP)[PAD]]
    if generic:   # every token of the protection prompts and of the target prompt (generic word directions)
        for p in (ANIM, PEOP, TGT):
            Q += [H(p)[i] for i in range(1, eot(p) + 1)]
    if compounds == "lite":   # only the three "dog" pieces of the compound prompt (hot dog / dog-fish / dog-wood)
        Q += [H(COMP)[3], H(COMP)[6], H(COMP)[10]]
    elif compounds:
        Q += [H(COMP)[i] for i in range(1, eot(COMP) + 1)] + [H(COMP)[PAD]]
    return Q

def basis(vs):
    B = []
    for v in vs:
        v = v.clone()
        for _ in range(2):
            for b in B: v -= (v @ b) * b
        if v.norm() > 1e-6: B.append(v / v.norm())
    return B

def resid(v, B):
    v = v.clone()
    for b in B: v -= (v @ b) * b
    return v

def sources(p=DOG):
    e = eot(p); return {"word": H(p)[e-1], "eot": H(p)[e], "pad": H(p)[PAD]}

def targets(p=TGT, lam=0.0):
    # word target: (1-lam) * "corgi" token + lam * "suit" token of the target prompt
    w = (1 - lam) * H(p)[idx(p, "corgi</w>")] + lam * H(p)[idx(p, "suit</w>")]
    return {"word": w, "eot": H(p)[eot(p)], "pad": H(p)[PAD]}

def detectors(**kw):
    B = basis(protect_vectors(**kw)); S = sources(); U = {}
    for k, s in S.items():
        r = resid(s, B); U[k] = r / (r @ s)
    return U, B

def pooled_detector(extra=True):
    ps = [C[p]["pooled"].double() for p in (CAT, CORGI, ASTRO, HOT)]
    if extra: ps += [C[p]["pooled"].double() for p in (ANIM, PEOP, COMP)]
    B = basis(ps); s = C[DOG]["pooled"].double(); r = resid(s, B)
    return r / (r @ s)
