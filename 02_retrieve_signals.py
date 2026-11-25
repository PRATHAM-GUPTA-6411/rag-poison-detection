"""Step 2: embed KB, retrieve top-K for every target question under every attack condition,
and compute the four black-box signals + baseline features for each retrieved document.

python 02_retrieve_signals.py --dataset nq

Output: data/<ds>/features.csv, results/<ds>_latency.json
"""
import argparse, json, pickle, time
import numpy as np, pandas as pd, faiss
from tqdm import tqdm
from ragp.common import *

def embed_cached(emb, texts, path):
    if path.exists():
        E = np.load(path)
        if len(E) == len(texts):
            return E
    E = emb.encode(texts, show=True)
    np.save(path, E)
    return E

def retrieve(qe, index, P_emb, P_mask, K):
    """Top-K over clean index + the currently injected poison docs. Returns list of (kind, idx, score)."""
    s, i = index.search(qe[None, :], K)
    cands = [("c", int(j), float(v)) for j, v in zip(i[0], s[0]) if j >= 0]
    if P_mask.any():
        ps = P_emb[P_mask] @ qe
        pidx = np.nonzero(P_mask)[0]
        top = np.argsort(-ps)[:K]
        cands += [("p", int(pidx[t]), float(ps[t])) for t in top]
    cands.sort(key=lambda x: -x[2])
    return cands[:K]

def s1_features(scores):
    s = np.asarray(scores, dtype="float64")
    sd = s.std() + 1e-6
    return (s - s.mean()) / sd, s - np.median(s)

def emb_anomaly(E):
    """Baseline: how far each doc is from the centroid of the other retrieved docs (z-scored)."""
    n = len(E)
    tot = E.sum(0)
    sims = np.array([float(E[i] @ ((tot - E[i]) / max(1, n - 1))) for i in range(n)])
    return -(sims - sims.mean()) / (sims.std() + 1e-6)

def s1_context(scores, E):
    """Retrieval-geometry context (S1 group): poisons arrive as near-duplicate, equally-scored clusters;
    a genuine answer passage is usually a single outlier."""
    s = np.asarray(scores, dtype="float64"); z = (s - s.mean()) / (s.std() + 1e-6)
    cos = E @ E.T; np.fill_diagonal(cos, -1.0)
    return dict(n_out=np.full(len(s), float((z > 1).sum())),
                sib03=np.array([(np.abs(s - x) < 0.03).sum() - 1 for x in s], dtype=float),
                top_gap=s.max() - s,
                dup_n=(cos > 0.8).sum(1).astype(float),
                nn_sim=cos.max(1))

def s3_features(ids, texts, nli, cache, scores=None, E=None):
    """Pairwise contradiction inside the retrieved set; each pair scored in both directions.
    Relevance-gated variants only count contradictions with documents that are themselves relevant."""
    n = len(ids)
    need = [(a, b) for a in range(n) for b in range(n) if a != b and (ids[a], ids[b]) not in cache]
    if need:
        probs = nli.contradiction([(texts[a], texts[b]) for a, b in need])
        for (a, b), p in zip(need, probs):
            cache[(ids[a], ids[b])] = float(p)
    M = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a != b:
                M[a, b] = max(cache[(ids[a], ids[b])], cache[(ids[b], ids[a])])
    denom = max(1, n - 1)
    mean = M.sum(1) / denom
    mx = M.max(1)
    frac = (M > 0.5).sum(1) / denom
    out = dict(s3_mean=mean, s3_max=mx, s3_frac=frac)
    if scores is not None:
        sc = np.asarray(scores, dtype="float64"); z = (sc - sc.mean()) / (sc.std() + 1e-6)
        rel = z > 0.5
        out["s3_rel_max"] = np.array([M[a, rel & (np.arange(n) != a)].max() if (rel & (np.arange(n) != a)).any() else 0.0
                                      for a in range(n)])
        W = np.clip(E @ E.T, 0, None) ** 2; np.fill_diagonal(W, 0)
        out["s3_simw"] = (M * W).sum(1) / (W.sum(1) + 1e-9)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq")
    ap.add_argument("--latency-n", type=int, default=30, help="queries timed end-to-end without caches")
    args = ap.parse_args()
    ds, d = args.dataset, ds_dir(args.dataset)
    K = CFG["top_k_cand"]

    clean = read_jsonl(d / "kb_clean.jsonl")
    poison = read_jsonl(d / "kb_poison.jsonl")
    targets = read_jsonl(d / "targets.jsonl")
    print(f"clean={len(clean)} poison={len(poison)} targets={len(targets)}")

    emb = Embedder()
    C_emb = embed_cached(emb, [c["text"] for c in clean], d / "emb_clean.npy")
    P_emb = embed_cached(emb, [p["text"] for p in poison], d / "emb_poison.npy")
    Q_emb = emb.encode([t["question"] for t in targets])
    index = faiss.IndexFlatIP(C_emb.shape[1]); index.add(C_emb)

    p_attack = np.array([p["attack"] for p in poison]); p_idx = np.array([p["pidx"] for p in poison])
    present = [a for a in CFG["attacks"] if a in set(p_attack)]
    print("attacks:", present)
    conditions = [("none", 0)] + [(a, n) for a in present for n in CFG["n_poison_list"]]

    # ---- retrieval for all conditions
    rows = []
    for attack, N in conditions:
        mask = (p_attack == attack) & (p_idx < N) if attack != "none" else np.zeros(len(poison), bool)
        for qn, t in enumerate(targets):
            cands = retrieve(Q_emb[qn], index, P_emb, mask, K)
            z, gap = s1_features([c[2] for c in cands])
            E = np.stack([C_emb[i] if k == "c" else P_emb[i] for k, i, _ in cands])
            ea = emb_anomaly(E)
            ctx = s1_context([c[2] for c in cands], E)
            for r, ((kind, i, sc), zz, gg, aa) in enumerate(zip(cands, z, gap, ea)):
                doc = clean[i] if kind == "c" else poison[i]
                rows.append(dict(qid=t["qid"], attack=attack, n_poison=N, rank=r, doc_id=doc["id"],
                                 is_poison=doc["is_poison"],
                                 is_target_poison=int(kind == "p" and doc["qid"] == t["qid"]),
                                 is_gold=int(doc["id"] in t["gold_docs"]),
                                 score=sc, s1_z=zz, s1_gap=gg, emb_anom=aa,
                                 s4=s4_risk(doc["tier"]), tier=doc["tier"]) | {k: v[r] for k, v in ctx.items()})
    df = pd.DataFrame(rows)
    text_of = {c["id"]: c["text"] for c in clean} | {p["id"]: p["text"] for p in poison}

    # ---- S2 per unique document
    uniq = sorted(df.doc_id.unique())
    print(f"S2 on {len(uniq)} unique docs")
    inj = InjClassifier()
    s2m = dict(zip(uniq, inj.score([text_of[u] for u in uniq])))
    df["s2_regex"] = df.doc_id.map(lambda u: s2_regex(text_of[u]))
    df["s2_inj"] = df.doc_id.map(s2m)
    ppl_model = Perplexity()
    pm = dict(zip(uniq, ppl_model.score([text_of[u] for u in uniq])))
    df["ppl"] = df.doc_id.map(pm)          # baseline: perplexity filter

    # ---- S3 per retrieved set
    nli = NLI()
    cache_p = d / "nli_cache.pkl"
    cache = pickle.load(open(cache_p, "rb")) if cache_p.exists() else {}
    emb_of = {c["id"]: ("c", i) for i, c in enumerate(clean)} | {p["id"]: ("p", i) for i, p in enumerate(poison)}
    vec = lambda u: C_emb[emb_of[u][1]] if emb_of[u][0] == "c" else P_emb[emb_of[u][1]]
    s3 = {}
    for _, g in tqdm(df.groupby(["attack", "n_poison", "qid"], sort=False), desc="S3 NLI"):
        ids = list(g.doc_id)
        Eg = np.stack([vec(u) for u in ids]); tx = [text_of[i] for i in ids]
        out = s3_features(ids, tx, nli, cache, list(g.score), Eg)
        keep = trustrag_keep(Eg, tx)       # baseline: TrustRAG K-means + n-gram filter (stage 1, no LLM)
        out["pred_trustrag"] = np.array([0 if j in keep else 1 for j in range(len(ids))])
        for k, v in out.items():
            s3.setdefault(k, []).extend(zip(g.index, v))
    pickle.dump(cache, open(cache_p, "wb"))
    for k, v in s3.items():
        df.loc[[i for i, _ in v], k] = [x for _, x in v]

    df.to_csv(d / "features.csv", index=False)
    print(f"saved {len(df)} rows -> {d/'features.csv'}")

    # ---- latency: full detection path per query, no caches (prag, N=5)
    mask = (p_attack == "prag") & (p_idx < 5)
    times = {"retrieval": [], "s1": [], "s2": [], "s3": [], "s4": [], "total": [], "trustrag": [], "ppl": []}
    for qn, t in list(enumerate(targets))[: args.latency_n]:
        t0 = time.perf_counter()
        qe = emb.encode([t["question"]])[0]
        cands = retrieve(qe, index, P_emb, mask, K); t1 = time.perf_counter()
        E = np.stack([C_emb[i] if k == "c" else P_emb[i] for k, i, _ in cands])
        s1_features([c[2] for c in cands]); s1_context([c[2] for c in cands], E); t2 = time.perf_counter()
        docs = [clean[i] if k == "c" else poison[i] for k, i, _ in cands]
        txt = [x["text"] for x in docs]
        [s2_regex(x) for x in txt]; inj.score(txt); t3 = time.perf_counter()
        s3_features([x["id"] for x in docs], txt, nli, {}, [c[2] for c in cands], E); t4 = time.perf_counter()
        [s4_risk(x["tier"]) for x in docs]; t5 = time.perf_counter()
        for k, v in zip(times, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0)):
            times[k].append(v * 1000)
        t6 = time.perf_counter(); trustrag_keep(E, txt); t7 = time.perf_counter(); ppl_model.score(txt); t8 = time.perf_counter()
        times["trustrag"].append((t7 - t6) * 1000); times["ppl"].append((t8 - t7) * 1000)
    lat = {k: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))} for k, v in times.items() if v}
    lat["device"] = device(); lat["n_queries"] = len(times["total"]); lat["top_k"] = K
    json.dump(lat, open(RESULTS / f"{ds}_latency.json", "w"), indent=1)
    print("latency (ms/query):", {k: round(v["mean_ms"], 1) for k, v in lat.items() if isinstance(v, dict)})

if __name__ == "__main__":
    main()
