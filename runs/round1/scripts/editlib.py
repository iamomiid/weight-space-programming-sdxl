# Shared closed-form machinery: alignment of source/target prompts and the least-squares map T = I + D.
import torch
EOS = "<|endoftext|>"

def align(ts, tt):
    """Return list of (src_pos, tgt_pos, kind) for positions that differ between source and target prompts.
    Also returns the common-prefix length P (positions < P are identical in both, by causality)."""
    es, et = ts.index(EOS), tt.index(EOS)
    P = 0
    while ts[P] == tt[P]: P += 1
    suf = 0
    while ts[es - 1 - suf] == tt[et - 1 - suf] and es - 1 - suf > P: suf += 1
    mid_end = es - suf
    delta = et - es
    out = []
    for j in range(P, 77):
        t = j + delta
        if t > 76: break
        kind = "noun" if j < mid_end else ("suffix" if j < es else ("eos" if j == es else "pad"))
        out.append((j, t, kind))
    return P, out

def build_systems(E, pairs, keeps, w=None, dims=None):
    """Accumulate Gram A = sum w c c^T and cross B = sum w (c*-c) c^T (float64)."""
    w = w or {}
    d = 2048
    A = torch.zeros(d, d, dtype=torch.float64); B = torch.zeros(d, d, dtype=torch.float64)
    n = {"edit": 0, "keep": 0}
    for s, g in pairs:
        hs, _, ts = E[s]; ht, _, tt = E[g]
        hs = hs.double(); ht = ht.double()
        P, al = align(ts, tt)
        # identical prefix positions: preserve
        X = hs[:P]; A += w.get("keep", 1.0) * X.T @ X
        for j, t, kind in al:
            ww = w.get(kind, 1.0)
            c = hs[j]; r = ht[t] - c
            A += ww * torch.outer(c, c); B += ww * torch.outer(r, c); n["edit"] += 1
    for p in keeps:
        h = E[p][0].double()
        A += w.get("keep", 1.0) * h.T @ h; n["keep"] += 77
    return A, B, n

def solve(A, B, lam):
    d = A.shape[0]
    return torch.linalg.solve((A + lam * torch.eye(d, dtype=A.dtype)).T, B.T).T   # D = B (A + lam I)^-1

def evaluate(E, D, pairs, keeps, Wprobe=None):
    """Efficacy: ||Tc - c*|| / ||c - c*|| per kind (0 = perfect, 1 = no edit).
    Leakage: ||D c|| for keep prompts relative to the mean edit magnitude ||c* - c|| of dog prompts."""
    M = (lambda x: x @ Wprobe.T) if Wprobe is not None else (lambda x: x)
    stats = {}; edit_mag = []
    for s, g in pairs:
        hs, _, ts = E[s]; ht, _, tt = E[g]
        hs = hs.double(); ht = ht.double()
        P, al = align(ts, tt)
        for j, t, kind in al:
            c = hs[j]; cs = ht[t]
            num = M((c + D @ c - cs)[None]).norm().item(); den = M((c - cs)[None]).norm().item()
            stats.setdefault(kind, []).append(num / den); edit_mag.append(den)
    em = sum(edit_mag) / len(edit_mag)
    eff = {k: sum(v) / len(v) for k, v in stats.items()}
    leak = []
    for p in keeps:
        h = E[p][0].double()[1:]   # skip BOS (constant)
        leak.append((M(h @ D.T).norm(dim=1).mean().item() / em, p))
    leak.sort(reverse=True)
    return eff, sum(l for l, _ in leak) / len(leak), leak[:6], em

def edit_vectors(E, pairs, kinds=("noun", "suffix", "eos", "pad")):
    Cs, Rs, Ks = [], [], []
    for s, g in pairs:
        hs, _, ts = E[s]; ht, _, tt = E[g]
        P, al = align(ts, tt)
        for j, t, kind in al:
            if kind in kinds:
                Cs.append(hs[j].double()); Rs.append(ht[t].double() - hs[j].double()); Ks.append(kind)
    return torch.stack(Cs), torch.stack(Rs), Ks

def keep_matrix(E, keeps, pairs=()):
    X = [E[p][0].double()[1:] for p in keeps]            # BOS handled separately
    for s, g in pairs:                                   # identical prefixes of dog prompts
        hs, _, ts = E[s]; P, _ = align(ts, E[g][2]); X.append(hs[1:P].double())
    return torch.cat(X)

def subspace_fit(C, R, wC, K, bos, k, eps, lam, wkeep=1.0, bos_w=1e3):
    """Read-side restricted ridge: find k input directions V maximizing edit energy vs keep energy
    (generalized eigenproblem Se v = mu (Sk + eps I) v), then D = Rhat V^T with closed-form ridge in z = V^T c."""
    d = C.shape[1]
    Se = (C * wC[:, None]).T @ C / wC.sum()
    Sk = K.T @ K / K.shape[0] + bos_w * torch.outer(bos, bos) / K.shape[0]
    # whiten by Sk + eps I
    L = torch.linalg.cholesky(Sk + eps * torch.eye(d, dtype=C.dtype))
    Li = torch.linalg.inv(L)
    M = Li @ Se @ Li.T
    mu, U = torch.linalg.eigh(M)
    V = Li.T @ U[:, -k:]                                 # (d, k) generalized eigenvectors
    V = torch.linalg.qr(V).Q                             # orthonormal basis of the read subspace
    Z = C @ V; Zk = K @ V; zb = bos @ V
    G = (Z * wC[:, None]).T @ Z + wkeep * Zk.T @ Zk + bos_w * torch.outer(zb, zb) + lam * torch.eye(k, dtype=C.dtype)
    Bm = (R * wC[:, None]).T @ Z
    Wz = torch.linalg.solve(G.T, Bm.T).T                 # (d, k)
    return Wz @ V.T, mu[-k:].flip(0)

def rrr_fit(C, R, wC, K, bos, out_basis, lam, wkeep=1.0, bos_w=1e3):
    """Output-restricted ridge: D = Q Wz, where Q (d, m) is an orthonormal output basis and Wz is the closed-form
    ridge solution of  min sum_edit w ||Q^T r - Wz c||^2 + wkeep sum_keep ||Wz c||^2 + bos_w ||Wz bos||^2 + lam ||Wz||^2."""
    d = C.shape[1]; Q = out_basis
    A = (C * wC[:, None]).T @ C + wkeep * K.T @ K + bos_w * torch.outer(bos, bos) + lam * torch.eye(d, dtype=C.dtype)
    Bz = ((R @ Q) * wC[:, None]).T @ C                    # (m, d)
    Wz = torch.linalg.solve(A, Bz.T).T                    # A symmetric
    return Q @ Wz

def output_basis(R, kinds, n_pc=0):
    vecs = []
    for k in ["noun", "eos", "pad", "suffix"]:
        idx = [i for i, x in enumerate(kinds) if x == k]
        vecs.append(R[idx].mean(0))
    M = torch.stack(vecs, 1)
    if n_pc:
        Q0 = torch.linalg.qr(M).Q
        Rres = R - (R @ Q0) @ Q0.T
        U = torch.linalg.svd(Rres[::3], full_matrices=False).Vh[:n_pc].T
        M = torch.cat([M, U], 1)
    return torch.linalg.qr(M).Q

def pair_tensors(E, pairs):
    """Vectorised version of edit_vectors: returns C (n,d), Cstar (n,d), kinds list, pair index."""
    src, tgt, kinds, pid = [], [], [], []
    for i, (s, g) in enumerate(pairs):
        hs, _, ts = E[s]; ht, _, tt = E[g]
        P, al = align(ts, tt)
        js = [a for a, _, _ in al]; tsx = [b for _, b, _ in al]
        src.append(hs[js]); tgt.append(ht[tsx]); kinds += [k for _, _, k in al]; pid += [i] * len(js)
    return torch.cat(src).double(), torch.cat(tgt).double(), kinds, torch.tensor(pid)

def evaluate_fast(Cv, Csv, kinds, Kv_list, D, em=None):
    Rv = Csv - Cv
    err = (Cv @ D.T - Rv).norm(dim=1) / Rv.norm(dim=1)
    eff = {k: err[[i for i, x in enumerate(kinds) if x == k]].mean().item() for k in ["noun", "suffix", "eos", "pad"]}
    em = em or Rv.norm(dim=1).mean().item()
    leaks = [((h @ D.T).norm(dim=1).mean().item() / em, p) for p, h in Kv_list]
    leaks.sort(reverse=True)
    return eff, sum(l for l, _ in leaks) / len(leaks), leaks, em
