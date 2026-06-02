#!/usr/bin/env python3
"""P13: does GROWING the graph improve transfer-HUB prediction? (the compounding
test, reframed to node level where the graph actually has signal & no leakage).
Predict which EXISTING task/method gains new transfer-in next year.
growing = rank by transfer-in accumulated up to Y-1 (updates yearly)
frozen  = rank by transfer-in accumulated only up to T0 (never updated)
"""
import json, pathlib, random, collections, numpy as np
from graph import Graph

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = Graph.load(ROOT / "data" / "graph_transfer_resolved.json")
T0S, TEND, KS = [2016, 2017, 2018, 2019, 2020], 2024, [5, 10, 20]
rng = np.random.default_rng(0); random.seed(0)
INDEG = {y: collections.Counter(d for (s, d), e in G.edges.items() if e["first_year"] == y)
         for y in range(2013, TEND + 1)}
def known(y): return [i for i, n in G.nodes.items() if n["first_year"] and n["first_year"] <= y]
def emerged_known(y, kn):  # existing nodes that gain transfer-in in year y
    s = set(kn); return {d for (a, d), e in G.edges.items() if e["first_year"] == y and d in s}

def rank_by_indeg(nodes, upto):
    sc = {i: sum(INDEG[yy].get(i, 0) for yy in range(2013, upto + 1)) for i in nodes}
    return [i for i in sorted(nodes, key=lambda i: -sc[i]) if sc[i] > 0]
def rank_by_momentum(nodes, upto):
    sc = {i: INDEG[upto].get(i, 0) - INDEG[upto-1].get(i, 0) for i in nodes}
    return [i for i in sorted(nodes, key=lambda i: -sc[i]) if sc[i] > 0]

def prec(ids, truth, k): return sum(1 for i in ids[:k] if i in truth)/max(1, len(ids[:k])) if ids and truth else 0.0
def rec(ids, truth, k): return len(set(ids[:k]) & truth)/len(truth) if truth and ids else 0.0
def boot(v, n=3000):
    v = np.asarray(v, float)
    if not len(v): return (0, 0, 0)
    m = [rng.choice(v, len(v), True).mean() for _ in range(n)]
    return (v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5))

def main():
    arms = {a: {k: [] for k in KS} for a in ["growing", "growing_mom", "frozen", "random"]}
    rec_arms = {a: {k: [] for k in KS} for a in ["growing", "frozen", "random"]}
    per_year_growing = collections.defaultdict(list)
    for T0 in T0S:
        for Y in range(T0 + 1, TEND + 1):
            kn = known(Y - 1)
            truth = emerged_known(Y, kn)
            if not truth or not kn: continue
            g_rank = rank_by_indeg(kn, Y - 1)         # growing: history up to Y-1
            gm_rank = rank_by_momentum(kn, Y - 1)
            f_rank = rank_by_indeg(kn, T0)            # frozen: history only up to T0
            r_rank = random.sample(kn, len(kn))
            for k in KS:
                arms["growing"][k].append(prec(g_rank, truth, k))
                arms["growing_mom"][k].append(prec(gm_rank, truth, k))
                arms["frozen"][k].append(prec(f_rank, truth, k))
                arms["random"][k].append(prec(r_rank, truth, k))
                rec_arms["growing"][k].append(rec(g_rank, truth, k))
                rec_arms["frozen"][k].append(rec(f_rank, truth, k))
                rec_arms["random"][k].append(rec(r_rank, truth, k))
            per_year_growing[Y].append(prec(g_rank, truth, 10))

    print("==== HUB prediction precision@k, mean [95% CI] ====")
    for k in KS:
        print(f"-- k={k} --")
        for a in ["growing", "growing_mom", "frozen", "random"]:
            m, lo, hi = boot(arms[a][k]); print(f"   {a:<12} {m:.3f} [{lo:.3f},{hi:.3f}]")
    print("\n==== recall@k ====")
    for k in KS:
        for a in ["growing", "frozen", "random"]:
            m, lo, hi = boot(rec_arms[a][k]); print(f"   r@{k} {a:<8} {m:.3f} [{lo:.3f},{hi:.3f}]")
    print("\n==== COMPOUNDING: growing - frozen (paired) ====")
    for k in KS:
        m, lo, hi = boot(np.array(arms["growing"][k]) - np.array(arms["frozen"][k]))
        print(f"   k={k}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] {'COMPOUNDS (CI>0)' if lo>0 else 'noise'}")
    print("\n==== growing - random ====")
    for k in KS:
        m, lo, hi = boot(np.array(arms["growing"][k]) - np.array(arms["random"][k]))
        print(f"   k={k}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] {'REAL' if lo>0 else ''}")
    print("\n==== per-year growing P@10 (does it rise?) ====")
    for Y in sorted(per_year_growing):
        v = per_year_growing[Y]; print(f"   {Y}: {np.mean(v):.3f}  (n={len(v)})")

if __name__ == "__main__":
    main()
