"""P4: temporal historical-replay evaluation.
Build graph <= T0, predict edges that emerge in T0+1..T_end, score, then ingest.
Arms: Growing vs Frozen vs Random (no-graph-LLM arm: p5). Reports hit@k per year.
"""
import sys, json, pathlib, random, numpy as np
import embed
from graph import Graph
from p3_navigate import navigate

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = Graph.load(ROOT / "data" / "graph_resolved.json")
T0, TEND, K = 2020, 2024, 20
random.seed(0)

# precompute embeddings for all nodes once
ALL_IDS = list(G.nodes)
EMB_MAT = embed.embed([G.nodes[i]["text"] for i in ALL_IDS])
EMB = {i: EMB_MAT[r] for r, i in enumerate(ALL_IDS)}


def known_by(year):
    nodes = {i: n for i, n in G.nodes.items()
             if n["first_year"] is not None and n["first_year"] <= year}
    edges = {(s, d) for (s, d), e in G.edges.items()
             if e["first_year"] is not None and e["first_year"] <= year}
    return nodes, edges


def emerged(year):
    return {(s, d) for (s, d), e in G.edges.items() if e["first_year"] == year}


def hit_at_k(pred, truth):
    if not pred:
        return 0.0
    return len(set((s, d) for s, d, _, _ in pred) & truth) / len(pred)


def run():
    arms = {"growing": {}, "frozen": {}, "random": {}}
    # frozen reference graph fixed at T0
    fz_nodes, fz_edges = known_by(T0)
    # growing graph state
    gr_nodes, gr_edges = known_by(T0)

    for Y in range(T0 + 1, TEND + 1):
        truth = emerged(Y)
        # growing: predict with current graph (has not seen Y)
        pred_g = navigate(gr_nodes, gr_edges, EMB, K)
        arms["growing"][Y] = hit_at_k(pred_g, truth)
        # frozen: always T0 graph
        pred_f = navigate(fz_nodes, fz_edges, EMB, K)
        arms["frozen"][Y] = hit_at_k(pred_f, truth)
        # random: k random non-existent edges among known nodes
        kn = list(gr_nodes)
        rcand = []
        while len(rcand) < K and len(kn) > 1:
            s, d = random.choice(kn), random.choice(kn)
            if s != d and (s, d) not in gr_edges:
                rcand.append((s, d, 0, "random"))
        arms["random"][Y] = hit_at_k(rcand, truth)
        # GROW: ingest year Y into growing graph (AFTER scoring)
        for i, n in G.nodes.items():
            if n["first_year"] == Y:
                gr_nodes[i] = n
        gr_edges |= truth
        print(f"  Y={Y}  truth_edges={len(truth):4d}  "
              f"growing={arms['growing'][Y]:.3f}  frozen={arms['frozen'][Y]:.3f}  random={arms['random'][Y]:.3f}")

    out = ROOT / "data" / "eval_curves.json"
    out.write_text(json.dumps(arms, indent=1))
    print("\nsaved", out)
    print("compounding check: growing should rise & beat frozen over years.")
    return arms


if __name__ == "__main__":
    run()
