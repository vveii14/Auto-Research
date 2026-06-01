#!/usr/bin/env python3
"""P11: REFORMULATION test. Predict the semantic REGION that will receive transfer
next year (not exact edges). New-node-tolerant: a predicted target counts as a hit
if its embedding is within TAU of a real emerged target. Arms:
  llm_raw       : LLM proposes target areas (generative; can name new concepts)
  llm_rerank    : LLM proposals re-ranked by proximity to known transfer-target regions (graph-informed)
  popularity    : most-frequent past transfer targets (structural, no LLM)
  random        : random known nodes
"""
import json, pathlib, random, collections, numpy as np
import embed, llm
from graph import Graph

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = Graph.load(ROOT / "data" / "graph_transfer_resolved.json")
T0S, TEND, KS, TAU = [2019, 2020], 2024, [5, 10, 20], 0.60
random.seed(0); rng = np.random.default_rng(0)
ALL = list(G.nodes)
EMB = {i: v for i, v in zip(ALL, embed.embed([G.nodes[i]["text"] for i in ALL]))}


def targets_upto(y):   # known transfer-target nodes (with multiplicity) up to year y
    c = collections.Counter()
    for (s, d), e in G.edges.items():
        if e["first_year"] and e["first_year"] <= y: c[d] += 1
    return c

def emerged_targets(y):
    return [d for (s, d), e in G.edges.items() if e["first_year"] == y]

def llm_targets(y, known_names, k):
    random.shuffle(known_names)
    usr = (f"Brain medical-imaging research areas/tasks known up to {y}:\n- " + "\n- ".join(known_names[:120]) +
           f"\n\nList the {k} task areas most likely to RECEIVE new knowledge TRANSFER "
           f"(pretraining / representation reuse / cross-task or cross-modal help) in {y+1}. "
           'JSON only: {"targets":["...", "..."]}')
    try:
        import re
        txt = llm.call("You are a brain medical-imaging research strategist.", usr, max_tokens=1500)
        try: return out.get("targets", []) if (out := llm.extract_json(txt)) else []
        except Exception: return re.findall(r'"([^"]{6,})"', txt)[:k]
    except Exception:
        return []

def region_precision(pred_vecs, truth_vecs, k):
    if not truth_vecs or not pred_vecs: return 0.0
    T = np.stack(truth_vecs)
    hits = sum(1 for v in pred_vecs[:k] if np.max(T @ v) >= TAU)
    return hits / max(1, len(pred_vecs[:k]))

def run(T0):
    arms = {a: {k: [] for k in KS} for a in ["llm_raw", "llm_rerank", "popularity", "random"]}
    for Y in range(T0 + 1, TEND + 1):
        cnt = targets_upto(Y - 1)
        known = [i for i in cnt]                       # known transfer-target nodes
        known_names = [G.nodes[i]["text"] for i in known] or [G.nodes[i]["text"] for i in ALL if G.nodes[i]["first_year"] and G.nodes[i]["first_year"] <= Y-1]
        truth = [EMB[d] for d in emerged_targets(Y)]
        if not truth: continue
        # LLM proposals
        props = llm_targets(Y - 1, list(known_names), 25)
        pv = embed.embed(props) if props else []
        # known transfer-target region centroids for rerank
        tt_vecs = np.stack([EMB[i] for i in known]) if known else None
        def rerank(vecs):
            if tt_vecs is None or len(vecs) == 0: return list(vecs)
            sc = [np.max(tt_vecs @ v) for v in vecs]
            return [vecs[i] for i in np.argsort(sc)[::-1]]
        # popularity: most frequent past targets
        pop = [EMB[i] for i, _ in cnt.most_common(25)]
        # random known nodes
        rnodes = [EMB[i] for i in random.sample(known, min(25, len(known)))] if known else []
        for k in KS:
            arms["llm_raw"][k].append(region_precision(list(pv), truth, k))
            arms["llm_rerank"][k].append(region_precision(rerank(list(pv)), truth, k))
            arms["popularity"][k].append(region_precision(pop, truth, k))
            arms["random"][k].append(region_precision(rnodes, truth, k))
    return arms

def boot(vals, n=4000):
    vals = np.asarray(vals, float)
    if not len(vals): return (0, 0, 0)
    m = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return (vals.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5))

def main():
    cells = {a: {k: [] for k in KS} for a in ["llm_raw", "llm_rerank", "popularity", "random"]}
    for T0 in T0S:
        print("run T0", T0); acc = run(T0)
        for a in cells:
            for k in KS: cells[a][k].extend(acc[a][k])
    print(f"\n=== region-tolerant precision@k (TAU={TAU}), {len(cells['llm_raw'][10])} cells, mean [95% CI] ===")
    for k in KS:
        print(f"-- k={k} --")
        for a in ["llm_raw", "llm_rerank", "popularity", "random"]:
            m, lo, hi = boot(cells[a][k]); print(f"   {a:<11} {m:.3f} [{lo:.3f},{hi:.3f}]")
    print("\n=== paired: does graph-rerank beat raw LLM? does anything beat random? ===")
    for k in KS:
        m, lo, hi = boot(np.array(cells["llm_rerank"][k]) - np.array(cells["llm_raw"][k]))
        print(f" rerank-rawLLM k={k}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] {'REAL' if lo>0 else 'noise'}")
        m, lo, hi = boot(np.array(cells["llm_raw"][k]) - np.array(cells["random"][k]))
        print(f" rawLLM-random k={k}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] {'REAL' if lo>0 else 'noise'}")

if __name__ == "__main__":
    main()
