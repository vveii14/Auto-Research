#!/usr/bin/env python3
"""P15: turn the verify signal into TRAINING. Learning-to-rank hub predictor vs the
hand-written persistence rule. Expanding-window temporal split (no leakage):
for each test year Y, train a logistic ranker ONLY on years < Y.
Features per (node, Y) are all computed from <= Y-1 (leakage-free)."""
import json, pathlib, collections, numpy as np
from graph import Graph
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = Graph.load(ROOT / "data" / "graph_transfer_resolved.json")
rng = np.random.default_rng(0)
INDEG = {y: collections.Counter(d for (s, d), e in G.edges.items() if e["first_year"] == y) for y in range(2013, 2025)}
OUTDEG = {y: collections.Counter(s for (s, d), e in G.edges.items() if e["first_year"] == y) for y in range(2013, 2025)}
def known(y): return [i for i, n in G.nodes.items() if n["first_year"] and n["first_year"] <= y]
def newhub(y, kn): s = set(kn); return {d for (a, d), e in G.edges.items() if e["first_year"] == y and d in s}

FEATS = ["persist", "logpersist", "mom", "rec", "in1", "in2", "in3", "age", "outdeg", "is_task"]
def feat(n, Y):
    persist = sum(INDEG[yy].get(n, 0) for yy in range(2013, Y))      # cumulative transfer-in up to Y-1
    in1, in2 = INDEG[Y-1].get(n, 0), INDEG[Y-2].get(n, 0)
    in3 = in1 + in2 + INDEG[Y-3].get(n, 0)
    out = sum(OUTDEG[yy].get(n, 0) for yy in range(2013, Y))
    return [persist, np.log1p(persist), in1-in2, in1+0.5*in2, in1, in1+in2, in3,
            (Y-1)-G.nodes[n]["first_year"], out, 1.0 if G.nodes[n]["type"]=="task" else 0.0]

def dataset(Y):
    kn = known(Y-1); pos = newhub(Y, kn)
    X = np.array([feat(n, Y) for n in kn]); y = np.array([1 if n in pos else 0 for n in kn])
    return kn, X, y

def prec_at(rank_ids, pos, ks): return {k: (sum(1 for i in rank_ids[:k] if i in pos)/k) for k in ks}

KS = [5, 10, 20]
TEST_YEARS = [2021, 2022, 2023, 2024]
res = collections.defaultdict(lambda: collections.defaultdict(list))
weights_final = None
for Y in TEST_YEARS:
    # train on all years strictly before Y (expanding window)
    Xtr, ytr = [], []
    for ty in range(2018, Y):
        _, X, y = dataset(ty)
        if X.size: Xtr.append(X); ytr.append(y)
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
    weights_final = dict(zip(FEATS, clf.coef_[0]))
    # test on Y
    kn, Xte, yte = dataset(Y); pos = newhub(Y, kn)
    score = clf.predict_proba(sc.transform(Xte))[:, 1]
    sup_rank = [kn[i] for i in np.argsort(-score)]
    persist_rank = [kn[i] for i in np.argsort(-Xte[:, 0])]      # hand rule: persist
    mom_rank = [kn[i] for i in np.argsort(-Xte[:, 2])]          # momentum
    rand_rank = list(rng.permutation(kn))
    for name, r in [("supervised", sup_rank), ("persistence", persist_rank), ("momentum", mom_rank), ("random", rand_rank)]:
        p = prec_at(r, pos, KS)
        for k in KS: res[name][k].append(p[k])

print("=== precision@k on TEST years 2021-2024 (mean over 4 yrs), expanding-window, leakage-safe ===")
for name in ["supervised", "persistence", "momentum", "random"]:
    print(f"  {name:<12} " + "  ".join(f"P@{k}={np.mean(res[name][k]):.3f}" for k in KS))
print("\n=== per-test-year P@10 ===")
for i, Y in enumerate(TEST_YEARS):
    print(f"  {Y}: supervised={res['supervised'][10][i]:.2f}  persistence={res['persistence'][10][i]:.2f}")
print("\n=== supervised - persistence (paired over test years) ===")
for k in KS:
    d = np.array(res["supervised"][k]) - np.array(res["persistence"][k])
    print(f"  P@{k}: {d.mean():+.3f}  (per-year diffs: {[round(x,2) for x in d]})")
print("\n=== learned feature weights (last model, trained 2018-2023) ===")
for f, w in sorted(weights_final.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {f:<12} {w:+.3f}")
