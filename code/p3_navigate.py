"""P3: graph-structure navigation -> top-k candidate transfer edges.
Two mechanisms (proposal §5.3): rich-node projection + structural-hole discovery.
Scoring is purely topological/embedding-based (no LLM free-generation) so any
predictive hit is attributable to the graph, not LLM memory.
"""
import numpy as np

TAU_RICH = 2      # min positive out-edges to count as a "rich" node
SIM_PROJ = 0.55   # similarity of a candidate target to the rich node's existing targets
SIM_HOLE = 0.70   # similarity for a structural hole
TAU_HOLE = 1      # min shared neighbors for a structural hole


def navigate(nodes, edges, emb, k=20):
    """nodes: id->node dict (known); edges: set of (src,dst) known; emb: id->vec."""
    ids = [i for i in nodes if i in emb]
    idx = {i: r for r, i in enumerate(ids)}
    if not ids:
        return []
    M = np.stack([emb[i] for i in ids])           # normalized rows
    pos_out, neigh = {}, {}
    for (s, d) in edges:
        pos_out.setdefault(s, []).append(d)
        neigh.setdefault(s, set()).add(d)
        neigh.setdefault(d, set()).add(s)

    cand = {}  # (src,dst) -> (score, reason)

    # (a) rich-node projection
    for mnode, outs in pos_out.items():
        outs = [o for o in outs if o in idx]
        if len(outs) < TAU_RICH or mnode not in idx:
            continue
        centroid = M[[idx[o] for o in outs]].mean(0)
        sims = M @ centroid
        for i, t in enumerate(ids):
            if t == mnode or t in neigh.get(mnode, ()) or nodes[t]["type"] != "task":
                continue
            if sims[i] >= SIM_PROJ:
                sc = float(sims[i]) * (1 + np.log1p(len(outs)))
                key = (mnode, t)
                if sc > cand.get(key, (0,))[0]:
                    cand[key] = (sc, "rich-projection")

    # (b) structural holes among tasks
    tasks = [i for i in ids if nodes[i]["type"] == "task"]
    if len(tasks) >= 2:
        T = M[[idx[i] for i in tasks]]
        S = T @ T.T
        for a in range(len(tasks)):
            for b in range(len(tasks)):
                if a == b:
                    continue
                ia, ib = tasks[a], tasks[b]
                if (ia, ib) in edges or S[a, b] < SIM_HOLE:
                    continue
                shared = len(neigh.get(ia, set()) & neigh.get(ib, set()))
                if shared >= TAU_HOLE:
                    sc = float(S[a, b]) * (1 + shared)
                    key = (ia, ib)
                    if sc > cand.get(key, (0,))[0]:
                        cand[key] = (sc, "structural-hole")

    ranked = sorted(cand.items(), key=lambda kv: -kv[1][0])[:k]
    return [(s, d, sc, why) for (s, d), (sc, why) in ranked]
