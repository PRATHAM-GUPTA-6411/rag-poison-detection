"""Step 4: end-to-end effect on the LLM (Ollama, black box). Needs Ollama running and the model pulled.

python 04_generate_asr.py --dataset nq                 # main conditions (N=5) + clean, model from OLLAMA_MODEL
python 04_generate_asr.py --dataset nq --all-n         # also N=1,3
python 04_generate_asr.py --dataset nq --no-llm        # only the LLM-free context-contamination metric
set OLLAMA_MODEL=mistral:7b  (Windows) to evaluate another generator; results get the model name in the file name.

ASR = % of target questions whose answer contains the attacker's incorrect answer (PoisonedRAG metric).
ACC = % whose answer contains the correct answer. [lo, hi] = bootstrap 95% CI over questions.
Ctx-poison = % of questions with >=1 target poison among the top-k_ctx docs passed to the LLM.
"""
import argparse
import numpy as np, pandas as pd
from tqdm import tqdm
from ragp.common import *

DEFENCES = ["none", "keyword", "emb_anomaly", "ppl", "trustrag", "weighted", "xgb", "xgb_nos4", "xgb_lite"]

def ci(x, seed=0, n=1000):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    m = np.random.default_rng(seed).choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq")
    ap.add_argument("--all-n", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (quick test)")
    ap.add_argument("--defences", default="", help="comma-separated subset, e.g. none,trustrag,xgb,xgb_nos4")
    ap.add_argument("--n-only", default="", help="comma-separated poison counts to run, e.g. 1,3")
    args = ap.parse_args()
    ds, d = args.dataset, ds_dir(args.dataset)
    df = pd.read_csv(d / "preds.csv", dtype={"qid": str})
    df["pred_none"] = 0
    defences = [x for x in DEFENCES if f"pred_{x}" in df]
    if args.defences:
        defences = [x for x in defences if x in args.defences.split(",")]
    attacks = [a for a in CFG["attacks"] if a in set(df.attack)]
    text_of = {r["id"]: r["text"] for r in read_jsonl(d / "kb_clean.jsonl") + read_jsonl(d / "kb_poison.jsonl")}
    tg = {t["qid"]: t for t in read_jsonl(d / "targets.jsonl")}
    qids = list(tg)[: args.limit] if args.limit else list(tg)
    ns = [int(x) for x in args.n_only.split(",")] if args.n_only else (CFG["n_poison_list"] if args.all_n else [5])
    conds = ([] if args.n_only else [("none", 0)]) + [(a, n) for a in attacks for n in ns]
    tag = "" if args.no_llm else "_" + CFG["ollama_model"].replace(":", "-").replace("/", "-")
    llm = None if args.no_llm else LLM(d / f"llm_cache_{CFG['ollama_model'].replace(':', '_').replace('/', '_')}.jsonl")

    rows, answers = [], []
    groups = {k: g.sort_values("rank") for k, g in df[df.qid.isin(qids)].groupby(["attack", "n_poison", "qid"])}
    for attack, n in conds:
        for dfn in defences:
            asr, acc, ctxp, kept = [], [], [], []
            for qid in tqdm(qids, desc=f"{attack} N={n} def={dfn}", leave=False):
                g = groups[(attack, n, qid)]
                ctx = g[g[f"pred_{dfn}"] == 0].head(CFG["top_k_ctx"])
                ctxp.append(int(ctx.is_target_poison.sum() > 0)); kept.append(len(ctx))
                if llm:
                    t = tg[qid]
                    resp = llm.ask(t["question"], [text_of[i] for i in ctx.doc_id])
                    asr.append(answer_contains(resp, t["incorrect"])); acc.append(answer_contains(resp, t["correct"]))
                    answers.append(dict(attack=attack, n_poison=n, defence=dfn, qid=qid, response=resp,
                                        hit_incorrect=asr[-1], hit_correct=acc[-1]))
            r = dict(attack=attack, n_poison=n, defence=dfn, ctx_poison_rate=np.mean(ctxp), avg_ctx_docs=np.mean(kept))
            if llm:
                lo, hi = ci(acc)
                r |= dict(ACC=np.mean(acc), ACC_lo=lo, ACC_hi=hi)
                if attack != "none":
                    lo, hi = ci(asr)
                    r |= dict(ASR=np.mean(asr), ASR_lo=lo, ASR_hi=hi)
            rows.append(r)
    res = pd.DataFrame(rows)
    out = (f"{ds}_asr{tag}" if llm else f"{ds}_asr_nollm") + (f"_N{args.n_only.replace(',', '-')}" if args.n_only else "")
    res.to_csv(RESULTS / f"{out}.csv", index=False)
    if answers:
        pd.DataFrame(answers).to_csv(RESULTS / f"{ds}_llm_answers{tag}.csv", index=False)
    with open(RESULTS / f"{out}.md", "w", encoding="utf-8") as f:
        f.write(f"# End-to-end results: {ds}  (LLM={CFG['ollama_model'] if llm else 'not run'}, "
                f"k_ctx={CFG['top_k_ctx']}, questions={len(qids)})\n\n{res.round(3).to_markdown(index=False)}\n")
    print(res.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
