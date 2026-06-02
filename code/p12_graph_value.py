#!/usr/bin/env python3
"""P12: the FAIR 'does the graph add value' test on the purified transfer graph.
- LLM held constant; test LLM-alone vs LLM + graph-feature (rank fusion, ADD not replace).
- Smarter leakage-free graph features: persistence, momentum, structural, novelty.
- Node-level 'rising hub' prediction (graph's best shot, leakage-free).
- LLM proposals cached by year (cheap). Multi-T0 + bootstrap CI.
"""
import json, pathlib, random, collections, numpy as np
import embed, llm
from graph import Graph
from p3_navigate import navigate

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = Graph.load(ROOT / "data" / "graph_transfer_resolved.json")
CACHE = ROOT / "data" / "llm_region_cache.json"
T0S, TEND, KS, TAU = [2017, 2018, 2019, 2020], 2024, [5, 10, 20], 0.60
random.seed(0); rng = np.random.default_rng(0)
ALL = list(G.nodes); EMB = {i: v for i, v in zip(ALL, embed.embed([G.nodes[i]["text"] for i in ALL]))}

# ---- temporal structure ----
def indeg_year(y):  # transfer-in count per node in exactly year y
    c = collections.Counter(d for (s, d), e in G.edges.items() if e["first_year"] == y)
    return c
INDEG = {y: indeg_year(y) for y in range(2014, TEND + 1)}

def known_nodes(y): return [i for i, n in G.nodes.items() if n["first_year"] and n["first_year"] <= y]
def emerged_target_ids(y): return [d for (s, d), e in G.edges.items() if e["first_year"] == y]

# ---- leakage-free graph feature scorers: return ranked known node ids ----
def feat_persistence(known, y):
    sc = {i: sum(INDEG[yy].get(i, 0) for yy in range(2014, y)) for i in known}
    return [i for i in sorted(known, key=lambda i: -sc[i]) if sc[i] > 0]
def feat_momentum(known, y):
    sc = {i: INDEG[y-1].get(i, 0) - INDEG[y-2].get(i, 0) for i in known}
    return [i for i in sorted(known, key=lambda i: -sc[i]) if sc[i] > 0]
def feat_recency(known, y):
    sc = {i: INDEG[y-1].get(i, 0) + 0.5*INDEG[y-2].get(i, 0) for i in known}
    return [i for i in sorted(known, key=lambda i: -sc[i]) if sc[i] > 0]
def feat_novelty(known, y):  # nodes far from established transfer targets (sparse regions)
    tgt = [i for i in known if any(INDEG[yy].get(i, 0) for yy in range(2014, y))]
    if not tgt: return []
    T = np.stack([EMB[i] for i in tgt])
    sc = {i: -np.max(T @ EMB[i]) for i in known}  # more novel = farther
    return sorted(known, key=lambda i: -sc[i])

FEATS = {"persistence": feat_persistence, "momentum": feat_momentum,
         "recency": feat_recency, "novelty": feat_novelty}

# ---- LLM proposals (cached by year) ----
def llm_props(y, names):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if str(y) in cache: return cache[str(y)]
    random.seed(y); names = names[:]; random.shuffle(names)
    import re
    usr = (f"Brain medical-imaging research tasks known up to {y}:\n- " + "\n- ".join(names[:120]) +
           f"\n\nList the 25 task areas most likely to RECEIVE new knowledge TRANSFER in {y+1}. "
           'JSON: {"targets":["..."]}')
    try:
        txt = llm.call("You are a brain-imaging research strategist.", usr, max_tokens=1500)
        try: props = (llm.extract_json(txt) or {}).get("targets", [])
        except Exception: props = re.findall(r'"([^"]{6,})"', txt)[:25]
    except Exception: props = []
    cache[str(y)] = props; CACHE.write_text(json.dumps(cache)); return props

def rrf(*lists, k0=10):  # reciprocal rank fusion over heterogeneous items (by key)
    sc = collections.defaultdict(float); item = {}
    for L in lists:
        for r, it in enumerate(L):
            key = it if isinstance(it, str) else it
            sc[key] += 1.0/(k0+r); item[key] = it
    return [item[k] for k in sorted(sc, key=lambda k: -sc[k])]

def region_prec(vecs, truth_vecs, k):
    if not truth_vecs or not vecs: return 0.0
    T = np.stack(truth_vecs)
    return sum(1 for v in vecs[:k] if np.max(T @ v) >= TAU)/max(1, len(vecs[:k]))

def node_prec(ids, truth_ids, k):
    if not truth_ids or not ids: return 0.0
    ts = set(truth_ids)
    return sum(1 for i in ids[:k] if i in ts)/max(1, len(ids[:k]))

def boot(vals, n=3000):
    vals = np.asarray(vals, float)
    if not len(vals): return (0, 0, 0)
    m = [rng.choice(vals, len(vals), True).mean() for _ in range(n)]
    return (vals.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5))

def main():
    region = collections.defaultdict(lambda: collections.defaultdict(list))
    node = collections.defaultdict(lambda: collections.defaultdict(list))
    for T0 in T0S:
        for Y in range(T0 + 1, TEND + 1):
            known = known_nodes(Y - 1)
            if not known: continue
            names = [G.nodes[i]["text"] for i in known]
            truth_ids_all = emerged_target_ids(Y)
            truth_vecs = [EMB[d] for d in truth_ids_all]
            truth_known = [d for d in truth_ids_all if d in set(known)]  # rising-hub truth
            if not truth_vecs: continue
            props = llm_props(Y - 1, names)
            pv = props  # text items
            # region arms
            region_arms = {"llm_raw": list(embed.embed(pv)) if pv else []}
            for fn, f in FEATS.items():
                fl = f(known, Y)
                region_arms[f"feat:{fn}"] = [EMB[i] for i in fl]
                fused = rrf(pv, [G.nodes[i]["text"] for i in fl])
                region_arms[f"llm+{fn}"] = list(embed.embed(fused[:40])) if fused else []
            region_arms["random"] = [EMB[i] for i in random.sample(known, min(40, len(known)))]
            for a, vv in region_arms.items():
                for k in KS: region[a][k].append(region_prec(vv, truth_vecs, k))
            # node-level rising-hub arms (leakage-free graph features)
            node_arms = {}
            for fn, f in FEATS.items(): node_arms[f"feat:{fn}"] = f(known, Y)
            # llm mapped to nearest known node
            if pv:
                pe = embed.embed(pv); KN = np.stack([EMB[i] for i in known])
                node_arms["llm_mapped"] = [known[int(np.argmax(KN @ e))] for e in pe]
            else: node_arms["llm_mapped"] = []
            node_arms["random"] = random.sample(known, min(40, len(known)))
            for a, ids in node_arms.items():
                for k in KS: node[a][k].append(node_prec(ids, truth_known, k))

    print("==== REGION-LEVEL (region-tolerant precision@k), mean [95% CI] ====")
    order = ["llm_raw"] + [f"llm+{f}" for f in FEATS] + [f"feat:{f}" for f in FEATS] + ["random"]
    for k in (5, 10):
        print(f"-- k={k} --")
        for a in order:
            m, lo, hi = boot(region[a][k]); print(f"   {a:<16} {m:.3f} [{lo:.3f},{hi:.3f}]")
    print("\n==== does graph ADD value? (LLM held constant) paired (llm+feat) - llm_raw ====")
    for f in FEATS:
        for k in (5, 10):
            d = np.array(region[f"llm+{f}"][k]) - np.array(region["llm_raw"][k])
            m, lo, hi = boot(d); print(f"   +{f:<12} k={k}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] {'HELPS' if lo>0 else ''}")

    print("\n==== NODE-LEVEL rising-hub prediction (leakage-free), precision@k ====")
    for k in (5, 10):
        print(f"-- k={k} --")
        for a in [f"feat:{f}" for f in FEATS] + ["llm_mapped", "random"]:
            m, lo, hi = boot(node[a][k]); print(f"   {a:<16} {m:.3f} [{lo:.3f},{hi:.3f}]")
    print("\n==== does any graph feature beat random at node level? ====")
    for f in FEATS:
        for k in (5, 10):
            d = np.array(node[f"feat:{f}"][k]) - np.array(node["random"][k])
            m, lo, hi = boot(d); print(f"   {f:<12} - random k={k}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] {'BEATS' if lo>0 else ''}")

if __name__ == "__main__":
    main()
