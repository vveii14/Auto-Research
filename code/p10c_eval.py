#!/usr/bin/env python3
"""P10c: re-eval on the PURIFIED transfer graph. Arms + k-sweep + bootstrap CI +
paired tests + per-year yield. Truth = extracted transfer edges (no inferred)."""
import json, pathlib, random, collections, numpy as np
import embed, llm
from graph import Graph
from p3_navigate import navigate

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = Graph.load(ROOT / "data" / "graph_transfer_resolved.json")
T0S, TEND, KMAX, KS, TAU = [2018, 2019, 2020], 2024, 100, [5, 10, 20, 50], 0.60
random.seed(0); rng = np.random.default_rng(0)
ALL = list(G.nodes)
EMB = {i: v for i, v in zip(ALL, embed.embed([G.nodes[i]["text"] for i in ALL]))}
PAP2N = collections.defaultdict(set)
for i, n in G.nodes.items():
    for p in n["paper_refs"]: PAP2N[p].add(i)


def known_nodes(y): return {i: n for i, n in G.nodes.items() if n["first_year"] and n["first_year"] <= y}
def known_edges(y): return {(s, d) for (s, d), e in G.edges.items() if e["first_year"] and e["first_year"] <= y}
def emerged(y): return [(s, d) for (s, d), e in G.edges.items() if e["first_year"] == y]


def cooccur_pred(nodes, kmax):
    adj = collections.defaultdict(set)
    for p, ns in PAP2N.items():
        ns = [i for i in ns if i in nodes]
        for a in ns:
            for b in ns:
                if a != b: adj[a].add(b)
    tasks = [i for i in nodes if nodes[i]["type"] == "task"]
    sc = []
    for a in tasks:
        for b in tasks:
            if a != b and b not in adj[a]:
                cn = len(adj[a] & adj[b])
                if cn: sc.append((cn, a, b))
    sc.sort(reverse=True)
    return [(a, b) for _, a, b in sc[:kmax]]


def nograph_llm_pred(nodes, kmax, y):
    names = [nodes[i]["text"] for i in nodes]; random.shuffle(names)
    usr = (f"Brain-imaging research tasks/methods known up to {y}:\n- " + "\n- ".join(names[:120]) +
           f"\n\nPropose the {min(kmax,50)} most promising NOT-YET-TRIED TRANSFER directions for {y+1} "
           "(knowledge/representation/pretraining from a source helps a target task). "
           'JSON: {"directions":[{"src":"...","dst":"..."}]}')
    try:
        import re
        txt = llm.call("You are a brain-imaging research strategist.", usr, max_tokens=3500)
        try: out = llm.extract_json(txt); return [(d["src"], d["dst"]) for d in out.get("directions", [])][:kmax]
        except Exception:
            return re.findall(r'"src"\s*:\s*"([^"]+)"\s*,\s*"dst"\s*:\s*"([^"]+)"', txt)[:kmax]
    except Exception:
        return []


def to_embs(pred, text=False):
    if text:
        return list(zip(embed.embed([s for s, d in pred]), embed.embed([d for s, d in pred]))) if pred else []
    return [(EMB[s], EMB[d]) for s, d in pred if s in EMB and d in EMB]


def score(pred_embs, truth):
    if not truth or not pred_embs: return {k: 0.0 for k in KS}
    Ts = np.stack([EMB[s] for s, d in truth]); Td = np.stack([EMB[d] for s, d in truth])
    r = {}
    for k in KS:
        hits = sum(1 for vs, vd in pred_embs[:k] if np.any((Ts @ vs >= TAU) & (Td @ vd >= TAU)))
        r[k] = hits / max(1, len(pred_embs[:k]))
    return r


def run(T0):
    arms = ["growing", "frozen", "cooccur", "nograph_llm", "random"]
    acc = {a: {k: [] for k in KS} for a in arms}
    grn, gre = known_nodes(T0), known_edges(T0); fzn, fze = known_nodes(T0), known_edges(T0)
    for Y in range(T0 + 1, TEND + 1):
        truth = emerged(Y)
        kn = list(grn); rc = []
        while len(rc) < KMAX and len(kn) > 1:
            s, d = random.choice(kn), random.choice(kn)
            if s != d and (s, d) not in gre: rc.append((s, d))
        preds = {"growing": to_embs([(s, d) for s, d, *_ in navigate(grn, gre, EMB, KMAX)]),
                 "frozen": to_embs([(s, d) for s, d, *_ in navigate(fzn, fze, EMB, KMAX)]),
                 "cooccur": to_embs(cooccur_pred(grn, KMAX)),
                 "nograph_llm": to_embs(nograph_llm_pred(grn, KMAX, Y), text=True),
                 "random": to_embs(rc)}
        for a in arms:
            r = score(preds[a], truth)
            for k in KS: acc[a][k].append(r[k])
        for i, n in G.nodes.items():
            if n["first_year"] == Y: grn[i] = n
        gre |= set(truth)
    return acc


def boot(vals, n=4000):
    vals = np.asarray(vals, float)
    if not len(vals): return (0, 0, 0)
    m = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return (vals.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5))


def main():
    print("PURIFIED transfer graph:", json.dumps(G.stats()))
    yr = collections.Counter(e["first_year"] for e in G.edges.values() if e["first_year"])
    print("edges by year:", {y: yr[y] for y in sorted(yr)})
    cells = {a: {k: [] for k in KS} for a in ["growing", "frozen", "cooccur", "nograph_llm", "random"]}
    for T0 in T0S:
        print("  run T0", T0); acc = run(T0)
        for a in cells:
            for k in KS: cells[a][k].extend(acc[a][k])
    print(f"\n=== precision@k, mean [95% CI], {len(cells['growing'][10])} cells ===")
    for k in (10, 20, 50):
        print(f"-- k={k} --")
        for a in ["growing", "frozen", "cooccur", "nograph_llm", "random"]:
            m, lo, hi = boot(cells[a][k]); print(f"   {a:<12} {m:.3f} [{lo:.3f},{hi:.3f}]")
    print("\n=== paired tests ===")
    for k in (10, 20, 50):
        m, lo, hi = boot(np.array(cells["growing"][k]) - np.array(cells["frozen"][k]))
        print(f" growing-frozen   k={k}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] {'REAL' if lo>0 else 'noise'}")
        m, lo, hi = boot(np.array(cells["growing"][k]) - np.array(cells["nograph_llm"][k]))
        print(f" growing-nographL k={k}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] {'REAL' if lo>0 else 'noise'}")


if __name__ == "__main__":
    main()
