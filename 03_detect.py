"""Step 3: risk scoring + baselines, question-level 5-fold cross-validation.

python 03_detect.py --dataset nq

Operating point: every tunable detector picks its threshold on the TRAINING folds as the F1-maximising
threshold subject to a false-positive rate <= CAP on clean (unattacked) queries (inner CV), because in
deployment most queries are clean and removing a genuine passage costs accuracy. Clean-query rows are
up-weighted (W_CLEAN) when training RF/XGBoost.

Methods
  baselines : keyword (regex), emb_anomaly (centroid outlier), ppl (GPT-2 perplexity),
              trustrag (TrustRAG K-means + n-gram filter, stage 1, no LLM), weighted (weighted sum)
  ours      : rf, xgb (all signals), xgb_nos4 (without simulated source credibility), xgb_lite (without S3/NLI)
Also: bootstrap 95% CIs over questions, repeated CV seeds, ablations, cross-attack transfer,
S4-spoofing test, threshold trade-off curve.
Output: data/<ds>/preds.csv, results/<ds>_detection*.{md,csv}, results/<ds>_tradeoff.csv, models/<ds>_*.json
"""
import argparse, itertools, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from ragp.common import *

CAP = 0.03      # max FPR on clean queries (training folds)
W_CLEAN = 3.0   # sample weight of clean-query rows
N_BOOT = 1000
FEATS = {"S1": ["s1_z", "s1_gap", "score", "n_out", "sib03", "top_gap", "dup_n", "nn_sim"],
         "S2": ["s2_regex", "s2_inj"],
         "S3": ["s3_mean", "s3_max", "s3_frac", "s3_rel_max", "s3_simw"],
         "S4": ["s4"]}
ALL = sum(FEATS.values(), [])
NOS4 = [f for f in ALL if f not in FEATS["S4"]]
LITE = [f for f in ALL if f not in FEATS["S3"]]
TH_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
SPOOF_S4 = s4_risk("high")

# ---------------------------------------------------------------- helpers
def folds_for(df, seed, k=5):
    """Question-level folds (GroupKFold with a random assignment of questions to folds)."""
    q = np.array(sorted(df.qid.unique()))
    rng = np.random.default_rng(seed); rng.shuffle(q)
    fold_of = {qq: i % k for i, qq in enumerate(q)}
    f = df.qid.map(fold_of).values
    return [(np.nonzero(f != i)[0], np.nonzero(f == i)[0]) for i in range(k)]

def inner_folds(d, seed, k=4):
    return folds_for(d.reset_index(drop=True), seed + 1000, k)

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    neg = y == 0
    return dict(precision=precision_score(y, p, zero_division=0), recall=recall_score(y, p, zero_division=0),
                f1=f1_score(y, p, zero_division=0), fpr=float(p[neg].mean()) if neg.any() else np.nan, n=len(y))

def boot_f1_ci(d, col, seed=0):
    """95% CI of F1 by resampling questions (qid) with replacement."""
    g = d.assign(tp=(d[col] == 1) & (d.is_poison == 1), fp=(d[col] == 1) & (d.is_poison == 0),
                 fn=(d[col] == 0) & (d.is_poison == 1)).groupby("qid")[["tp", "fp", "fn"]].sum().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    s = g[idx].sum(1)
    f1 = 2 * s[:, 0] / np.maximum(1, 2 * s[:, 0] + s[:, 1] + s[:, 2])
    return float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))

def capped_threshold(s, y, clean, grid=None):
    """Max-F1 threshold on attacked rows subject to FPR<=CAP on clean-query rows."""
    s, y, clean = np.asarray(s), np.asarray(y).astype(bool), np.asarray(clean)
    grid = np.unique(np.quantile(s, np.linspace(0.01, 0.995, 120))) if grid is None else grid
    P = s[:, None] >= grid[None, :]
    ok = P[clean].mean(0) <= CAP
    a = ~clean; ya = y[a][:, None]; Pa = P[a]
    tp = (Pa & ya).sum(0); fp = (Pa & ~ya).sum(0); fn = (~Pa & ya).sum(0)
    f = np.where(ok, 2 * tp / np.maximum(1, 2 * tp + fp + fn), -1)
    i = int(np.argmax(f))
    return float(grid[i]), float(f[i])

def weighted_signals(df, mins=None, maxs=None):
    raw = np.stack([df.s1_z.values, np.maximum(np.minimum(df.s2_regex.values, 2) / 2, df.s2_inj.values),
                    df.s3_mean.values, df.s4.values], 1)
    if mins is None:
        mins, maxs = raw.min(0), raw.max(0)
    return (raw - mins) / (maxs - mins + 1e-9), mins, maxs

WGRID = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=4) if abs(sum(w) - 1) < 1e-6]

def fit_weighted(tr):
    X, mn, mx = weighted_signals(tr)
    y, clean = tr.is_poison.values, (tr.attack == "none").values
    best = (-2, None, None)
    for w in WGRID:
        th, f = capped_threshold(X @ np.array(w), y, clean)
        if f > best[0]:
            best = (f, w, th)
    return dict(w=list(map(float, best[1])), th=best[2], mins=mn.tolist(), maxs=mx.tolist())

def predict_weighted(m, df):
    X, _, _ = weighted_signals(df, np.array(m["mins"]), np.array(m["maxs"]))
    s = X @ np.array(m["w"])
    return s, (s >= m["th"]).astype(int)

def make_model(kind, seed):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                         eval_metric="logloss", random_state=seed, n_jobs=-1)

def fit_prob(d, feats, kind, seed):
    w = np.where(d.attack == "none", W_CLEAN, 1.0)
    return make_model(kind, seed).fit(d[feats].fillna(0).values, d.is_poison.values, sample_weight=w)

def spoofed(d):
    d = d.copy(); d.loc[d.is_poison == 1, "s4"] = SPOOF_S4
    return d

def cv_predict(df, feats, kind, folds, seed, spoof=False):
    """Out-of-fold probabilities + predictions; threshold chosen by inner CV on each training fold.
    With spoof=True also returns predictions when every poison claims a high-credibility source."""
    prob, pred, ths = np.zeros(len(df)), np.zeros(len(df), int), []
    pred_sp = np.zeros(len(df), int)
    for tr, te in folds:
        trd = df.iloc[tr].reset_index(drop=True)
        inner = np.zeros(len(trd))
        for a, b in inner_folds(trd, seed):
            inner[b] = fit_prob(trd.iloc[a], feats, kind, seed).predict_proba(trd.iloc[b][feats].fillna(0).values)[:, 1]
        th, _ = capped_threshold(inner, trd.is_poison.values, (trd.attack == "none").values, TH_GRID)
        m = fit_prob(trd, feats, kind, seed)
        prob[te] = m.predict_proba(df.iloc[te][feats].fillna(0).values)[:, 1]
        pred[te] = prob[te] >= th; ths.append(th)
        if spoof:
            pred_sp[te] = m.predict_proba(spoofed(df.iloc[te])[feats].fillna(0).values)[:, 1] >= th
    return (prob, pred, float(np.mean(ths))) + ((pred_sp,) if spoof else ())

def table(df, pred_cols, by, ci=False):
    out = []
    for keys, g in df.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for c in pred_cols:
            r = dict(zip(by, keys)) | {"method": c.replace("pred_", "")} | metrics(g.is_poison, g[c])
            if ci:
                r["f1_lo"], r["f1_hi"] = boot_f1_ci(g, c)
            out.append(r)
    return pd.DataFrame(out)

def context_stats(df, pred, attacks):
    """Gold kept on clean queries; share of attacked queries (N=5) with a target poison in the LLM context."""
    d = df.assign(_p=pred).sort_values("rank")
    ctx = d[d._p == 0].groupby(["attack", "n_poison", "qid"]).head(CFG["top_k_ctx"])
    clean_q = df[df.attack == "none"].qid.unique()
    has_gold = df[df.attack == "none"].groupby("qid").is_gold.max()
    out = {"gold_kept": float(ctx[ctx.attack == "none"].groupby("qid").is_gold.max()
                              .reindex(clean_q, fill_value=0)[has_gold > 0].mean())}
    for a in attacks:
        qs = df[(df.attack == a) & (df.n_poison == 5)].qid.unique()
        c = ctx[(ctx.attack == a) & (ctx.n_poison == 5)]
        out[f"ctx_poison_{a}"] = float(c.groupby("qid").is_target_poison.max().reindex(qs, fill_value=0).mean())
    return out

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq")
    ap.add_argument("--seeds", type=int, default=3, help="number of CV seeds for the main XGBoost results")
    args = ap.parse_args()
    ds, d = args.dataset, ds_dir(args.dataset)
    seed = CFG["seed"]
    df = pd.read_csv(d / "features.csv", dtype={"qid": str})
    attacks = [a for a in CFG["attacks"] if a in set(df.attack)]
    folds = folds_for(df, seed)
    clean_mask = (df.attack == "none").values

    # ---- baselines
    df["pred_keyword"] = (df.s2_regex > 0).astype(int)
    for c in ("pred_emb_anomaly", "pred_ppl", "pred_weighted"):
        df[c] = 0
    df["prob_weighted"] = 0.0
    if "pred_trustrag" not in df:
        df["pred_trustrag"] = 0
    for tr, te in folds:
        trd = df.iloc[tr]; cl = trd[trd.attack == "none"]
        th = float(np.quantile(cl.emb_anom, 1 - CAP))
        df.loc[df.index[te], "pred_emb_anomaly"] = (df.iloc[te].emb_anom >= th).astype(int).values
        if "ppl" in df:
            th = float(np.quantile(cl.ppl, 1 - CAP))
            df.loc[df.index[te], "pred_ppl"] = (df.iloc[te].ppl >= th).astype(int).values
        m = fit_weighted(trd)
        s, p = predict_weighted(m, df.iloc[te])
        df.loc[df.index[te], "prob_weighted"] = s; df.loc[df.index[te], "pred_weighted"] = p

    # ---- ours
    ths = {}
    df["prob_rf"], df["pred_rf"], ths["rf"] = cv_predict(df, ALL, "rf", folds, seed)
    df["prob_xgb"], df["pred_xgb"], ths["xgb"], sp_xgb = cv_predict(df, ALL, "xgb", folds, seed, spoof=True)
    df["prob_xgb_nos4"], df["pred_xgb_nos4"], ths["xgb_nos4"] = cv_predict(df, NOS4, "xgb", folds, seed)
    df["prob_xgb_lite"], df["pred_xgb_lite"], ths["xgb_lite"] = cv_predict(df, LITE, "xgb", folds, seed)
    methods = ["pred_keyword", "pred_emb_anomaly", "pred_ppl", "pred_trustrag", "pred_weighted",
               "pred_rf", "pred_xgb", "pred_xgb_nos4", "pred_xgb_lite"]

    att = df[~clean_mask]
    main_t = pd.concat([table(att.assign(scope="all attacks"), methods, ["scope"], ci=True),
                        table(att[att.n_poison == 5], methods, ["attack"], ci=True)])
    clean = df[clean_mask]
    fpr_clean = pd.DataFrame([{"method": m.replace("pred_", ""), "fpr_all_clean_docs": clean[m].mean(),
                               "fpr_gold_docs": clean[clean.is_gold == 1][m].mean(), "n": len(clean)} for m in methods])
    ctx_t = pd.DataFrame([{"method": "none"} | context_stats(df, np.zeros(len(df)), attacks)] +
                         [{"method": m.replace("pred_", "")} | context_stats(df, df[m].values, attacks) for m in methods])
    by_n = table(att, ["pred_trustrag", "pred_weighted", "pred_xgb", "pred_xgb_nos4"], ["attack", "n_poison"])

    # ---- repeated CV seeds (variance of the main results)
    seed_rows = []
    for s_ in range(args.seeds):
        sf = folds_for(df, seed + 100 * s_)
        for name, feats in (("xgb", ALL), ("xgb_nos4", NOS4)):
            _, p, _ = cv_predict(df, feats, "xgb", sf, seed + s_)
            seed_rows.append({"seed": s_, "method": name} | metrics(att.is_poison, p[~clean_mask])
                             | {"fpr_clean": p[clean_mask].mean()} | context_stats(df, p, attacks))
    seeds_t = pd.DataFrame(seed_rows)
    seeds_sum = seeds_t.groupby("method")[["f1", "fpr_clean", "gold_kept"] + [f"ctx_poison_{a}" for a in attacks]] \
        .agg(["mean", "std"]).round(3)
    seeds_sum.columns = [f"{a}_{b}" for a, b in seeds_sum.columns]
    seeds_sum = seeds_sum.reset_index()

    # ---- S4 spoofing: every poison claims a high-credibility source (test folds only)
    sp_df = df.assign(pred_xgb_spoof=sp_xgb)
    spoof_t = pd.DataFrame([{"method": "xgb (normal metadata)"} | metrics(att.is_poison, att.pred_xgb) | context_stats(df, df.pred_xgb.values, attacks),
                            {"method": "xgb (all poisons spoof trusted source)"} | metrics(att.is_poison, sp_df[~clean_mask].pred_xgb_spoof)
                            | context_stats(df, sp_df.pred_xgb_spoof.values, attacks),
                            {"method": "xgb_nos4 (no source signal)"} | metrics(att.is_poison, att.pred_xgb_nos4)
                            | context_stats(df, df.pred_xgb_nos4.values, attacks)])

    # ---- ablation (xgb, same capped operating point)
    abl = []
    for name, feats in [("all", ALL)] + [(f"-{g}", [f for f in ALL if f not in FEATS[g]]) for g in FEATS] + \
                       [(f"only {g}", FEATS[g]) for g in FEATS]:
        _, p, _ = cv_predict(df, feats, "xgb", folds, seed)
        abl.append({"features": name} | metrics(att.is_poison, p[~clean_mask])
                   | {"fpr_clean": p[clean_mask].mean()} | context_stats(df, p, attacks))
    abl = pd.DataFrame(abl)

    # ---- cross-attack transfer: train on one attack (+clean), test on another
    xfer = []
    for a_tr in attacks:
        tr = df[df.attack.isin(["none", a_tr])].reset_index(drop=True)
        inner = np.zeros(len(tr))
        for a, b in inner_folds(tr, seed):
            inner[b] = fit_prob(tr.iloc[a], ALL, "xgb", seed).predict_proba(tr.iloc[b][ALL].fillna(0).values)[:, 1]
        th, _ = capped_threshold(inner, tr.is_poison.values, (tr.attack == "none").values, TH_GRID)
        m = fit_prob(tr, ALL, "xgb", seed)
        for a_te in attacks:
            if a_te != a_tr:
                te = df[df.attack == a_te]
                p = (m.predict_proba(te[ALL].fillna(0).values)[:, 1] >= th).astype(int)
                xfer.append({"train": a_tr, "test": a_te} | metrics(te.is_poison, p))
    xfer = pd.DataFrame(xfer)

    # ---- trade-off over the XGBoost threshold
    trade = pd.DataFrame([{"threshold": round(th, 2), "fpr_clean": float((clean.prob_xgb >= th).mean()),
                           "f1_attacked": f1_score(att.is_poison, att.prob_xgb >= th, zero_division=0)}
                          | context_stats(df, (df.prob_xgb >= th).astype(int).values, attacks) for th in np.arange(0.1, 0.96, 0.05)])

    # ---- final models on all data (demo + reproducibility)
    final = fit_prob(df, ALL, "xgb", seed)
    final.save_model(str(MODELS / f"{ds}_xgb.json"))
    json.dump({"features": ALL, "threshold": ths["xgb"], "cap": CAP, "w_clean": W_CLEAN},
              open(MODELS / f"{ds}_xgb_meta.json", "w"), indent=1)
    imp = pd.DataFrame({"feature": ALL, "importance": final.feature_importances_}).sort_values("importance", ascending=False)
    json.dump(fit_weighted(df), open(MODELS / f"{ds}_weighted.json", "w"), indent=1)

    df.to_csv(d / "preds.csv", index=False)
    trade.to_csv(RESULTS / f"{ds}_tradeoff.csv", index=False)
    sections = [("Main (attacked queries; per-attack rows are N=5; f1_lo/f1_hi = bootstrap 95% CI over questions)", main_t),
                ("False positives on clean-only queries", fpr_clean),
                ("Effect on LLM context (gold kept = clean queries whose gold passage survives; "
                 "ctx_poison = attacked queries, N=5, with a target poison in the context)", ctx_t),
                (f"Repeated CV seeds (mean/std over {args.seeds} seeds)", seeds_sum),
                ("S4 spoofing test", spoof_t), ("By number of injected poisons", by_n),
                ("Ablation (XGBoost)", abl), ("Cross-attack transfer (XGBoost)", xfer),
                ("Threshold trade-off (XGBoost)", trade), ("XGBoost feature importance", imp)]
    with open(RESULTS / f"{ds}_detection.md", "w", encoding="utf-8") as f:
        f.write(f"# Detection results: {ds}  (5-fold CV grouped by question; top-K={CFG['top_k_cand']}; operating point: "
                f"clean-query FPR <= {CAP} on training folds; mean thresholds "
                + ", ".join(f"{k} {v:.2f}" for k, v in ths.items()) + f"; attacks: {', '.join(attacks)})\n\n")
        for title, t in sections:
            f.write(f"## {title}\n\n{t.round(3).to_markdown(index=False)}\n\n")
    for name, t in [("main", main_t), ("fpr_clean", fpr_clean), ("context", ctx_t), ("seeds", seeds_t), ("spoof", spoof_t),
                    ("by_n", by_n), ("ablation", abl), ("transfer", xfer)]:
        t.to_csv(RESULTS / f"{ds}_detection_{name}.csv", index=False)
    print(main_t.round(3).to_string(index=False))
    print(ctx_t.round(3).to_string(index=False))
    print(f"-> {RESULTS / f'{ds}_detection.md'}")

if __name__ == "__main__":
    main()
