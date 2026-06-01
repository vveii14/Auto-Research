"""P4b: temporal replay with SEMANTIC matching (fixes the all-zero exact-match metric).
A predicted direction (s,d) "hits" an emerged edge (s',d') if
   sim(s,s') >= TAU and sim(d,d') >= TAU   (embedding cosine).
This tolerates brand-new nodes & entity-resolution fragmentation: we reward
pointing at the RIGHT REGION, not guessing exact node ids.
Reports precision@k (are our guesses right?) per arm per year.
"""
import sys, json, pathlib, random, numpy as np
import embed
from graph import Graph
from p3_navigate import navigate

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = Graph.load(ROOT / "data" / "graph_resolved.json")
T0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2020
TEND, K, TAU = 2024, 50, 0.60
random.seed(0)

ALL = list(G.nodes)
EMB = {i: v for i, v in zip(ALL, embed.embed([G.nodes[i]["text"] for i in ALL]))}


def known_by(y):
    nodes = {i: n for i, n in G.nodes.items() if n["first_year"] is not None and n["first_year"] <= y}
    edges = {(s, d) for (s, d), e in G.edges.items() if e["first_year"] is not None and e["first_year"] <= y}
    return nodes, edges


def emerged(y):
    return [(s, d) for (s, d), e in G.edges.items() if e["first_year"] == y]


def sem_precision(pred, truth):
    """fraction of top-k predictions that semantically match some emerged edge."""
    if not pred:
        return 0.0
    if not truth:
        return 0.0
    Ts = np.stack([EMB[s] for s, d in truth]); Td = np.stack([EMB[d] for s, d in truth])
    tp = 0
    for s, d, *_ in pred:
        ss = Ts @ EMB[s]; dd = Td @ EMB[d]
        if np.any((ss >= TAU) & (dd >= TAU)):
            tp += 1
    return tp / len(pred)


def run():
    arms = {"growing": {}, "frozen": {}, "random": {}}
    fz_nodes, fz_edges = known_by(T0)
    gr_nodes, gr_edges = known_by(T0)
    for Y in range(T0 + 1, TEND + 1):
        truth = emerged(Y)
        pg = navigate(gr_nodes, gr_edges, EMB, K)
        pf = navigate(fz_nodes, fz_edges, EMB, K)
        kn = list(gr_nodes); rc = []
        while len(rc) < K and len(kn) > 1:
            s, d = random.choice(kn), random.choice(kn)
            if s != d and (s, d) not in gr_edges:
                rc.append((s, d))
        arms["growing"][Y] = sem_precision(pg, truth)
        arms["frozen"][Y] = sem_precision(pf, truth)
        arms["random"][Y] = sem_precision(rc, truth)
        for i, n in G.nodes.items():
            if n["first_year"] == Y:
                gr_nodes[i] = n
        gr_edges |= set(truth)
        print(f"  Y={Y}  truth={len(truth):4d} |preds g/f={len(pg)}/{len(pf)}|  "
              f"growing={arms['growing'][Y]:.3f}  frozen={arms['frozen'][Y]:.3f}  random={arms['random'][Y]:.3f}")
    (ROOT / "data" / "eval_curves_sem.json").write_text(json.dumps(arms, indent=1))
    g = arms["growing"]; f = arms["frozen"]
    print(f"\nmean: growing={np.mean(list(g.values())):.3f}  frozen={np.mean(list(f.values())):.3f}  random={np.mean(list(arms['random'].values())):.3f}")
    print(f"growing trend (Y21->Y24): {g[2021]:.3f} -> {g[2024]:.3f}")


if __name__ == "__main__":
    run()
