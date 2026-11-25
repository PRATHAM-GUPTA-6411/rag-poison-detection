"""Minimal demo:  streamlit run app.py   (after steps 1-3 for the dataset)."""
import json
import numpy as np, pandas as pd, faiss, streamlit as st
from xgboost import XGBClassifier
from ragp.common import *

st.set_page_config(page_title="RAG Poisoning Detector", layout="wide")
ds = st.sidebar.selectbox("Dataset", [p.name for p in DATA.iterdir() if (p / "features.csv").exists()])
d = ds_dir(ds)

@st.cache_resource
def load(ds):
    clean = read_jsonl(d / "kb_clean.jsonl"); poison = read_jsonl(d / "kb_poison.jsonl")
    C, P = np.load(d / "emb_clean.npy"), np.load(d / "emb_poison.npy")
    idx = faiss.IndexFlatIP(C.shape[1]); idx.add(C)
    xgb = XGBClassifier(); xgb.load_model(str(MODELS / f"{ds}_xgb.json"))
    meta = json.load(open(MODELS / f"{ds}_xgb_meta.json"))
    return clean, poison, C, P, idx, xgb, meta, Embedder(), InjClassifier(), NLI(), read_jsonl(d / "targets.jsonl")

clean, poison, C, P, index, xgb, meta, emb, inj, nli, targets = load(ds)
from importlib import import_module
sig = import_module("02_retrieve_signals")
FEATS = meta["features"]

st.title("Black-box poisoned-document detection for RAG")
t = st.selectbox("Target question", targets, format_func=lambda x: x["question"])
attack = st.sidebar.radio("Attack", ["none", "prag", "inject"])
n = st.sidebar.slider("Poisoned docs injected", 1, 5, 5)
use_llm = st.sidebar.checkbox("Ask the LLM (needs Ollama running)", False)

pa = np.array([p["attack"] for p in poison]); pi = np.array([p["pidx"] for p in poison])
mask = (pa == attack) & (pi < n) if attack != "none" else np.zeros(len(poison), bool)
qe = emb.encode([t["question"]])[0]
cands = sig.retrieve(qe, index, P, mask, CFG["top_k_cand"])
docs = [clean[i] if k == "c" else poison[i] for k, i, _ in cands]
txt = [x["text"] for x in docs]
scores = [c[2] for c in cands]
E = np.stack([C[i] if k == "c" else P[i] for k, i, _ in cands])
z, gap = sig.s1_features(scores)
df = pd.DataFrame(dict(rank=range(len(docs)), text=txt, source=[x["source"] for x in docs], poisoned=[x["is_poison"] for x in docs],
                       score=scores, s1_z=z, s1_gap=gap, s2_regex=[s2_regex(x) for x in txt],
                       s2_inj=inj.score(txt), s4=[s4_risk(x["tier"]) for x in docs])
                  | sig.s1_context(scores, E) | sig.s3_features([x["id"] for x in docs], txt, nli, {}, scores, E))
df["risk"] = xgb.predict_proba(df[FEATS].fillna(0).values)[:, 1]
df["flagged"] = df.risk >= meta["threshold"]
st.dataframe(df[["rank", "risk", "flagged", "poisoned", "source", "score", "s1_z", "s2_inj", "s3_mean", "s4", "text"]]
             .style.format(precision=2).background_gradient(subset=["risk"], cmap="Reds"), use_container_width=True)
st.caption(f"Correct answer: **{t['correct']}** · attacker's target answer: **{t['incorrect']}**")
if use_llm:
    llm = LLM(d / f"llm_cache_{CFG['ollama_model'].replace(':', '_')}.jsonl")
    k = CFG["top_k_ctx"]
    c1, c2 = st.columns(2)
    c1.subheader("No defence"); c1.write(llm.ask(t["question"], list(df.text[:k])))
    c2.subheader("With detector"); c2.write(llm.ask(t["question"], list(df[~df.flagged].text[:k])))
