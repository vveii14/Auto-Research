"""P5: hardened temporal eval — sweep k, precision@k + recall@k, 5 arms,
mean +/- std across prediction years. Semantic matching (TAU). Pilot-scale.
Arms: growing | frozen | cooccur(static co-occurrence + common-neighbor) |
      nograph_llm | random.
Navigation graph excludes method->method edges (kills transitive inflation).
"""
import sys, json, pathlib, random, collections, numpy as np
import embed, llm
from graph import Graph
from p3_navigate import navigate

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = Graph.load(ROOT / "data" / "graph_resolved.json")
TEND, KMAX, KS, TAU = 2024, 100, [5, 10, 20, 50, 100], 0.60
random.seed(0)

ALL = list(G.nodes)
EMB = {i: v for i, v in zip(ALL, embed.embed([G.nodes[i]["text"] for i in ALL]))}

# paper -> nodes (for co-occurrence baseline)
PAP2N = collections.defaultdict(set)
for i, n in G.nodes.items():
    for p in n["paper_refs"]:
        PAP2N[p].add(i)


def known_nodes(y):
    return {i: n for i, n in G.nodes.items() if n["first_year"] is not None and n["first_year"] <= y}


def known_edges(y):  # exclude method->method (inflation source)
    return {(s, d) for (s, d), e in G.edges.items()
            if e["first_year"] is not None and e["first_year"] <= y
            and not (G.nodes[s]["type"] == "method" and G.nodes[d]["type"] == "method")}


def emerged(y):
    return [(s, d) for (s, d), e in G.edges.items() if e["first_year"] == y]


def cooccur_pred(nodes, kmax, y):
    """static co-occurrence + common-neighbor link prediction (Deep Ideation analog)."""
    adj = collections.defaultdict(set)
    for p, ns in PAP2N.items():
        ns = [i for i in ns if i in nodes]
        for a in ns:
            for b in ns:
                if a != b:
                    adj[a].add(b)
    tasks = [i for i in nodes if nodes[i]["type"] == "task"]
    scores = []
    for a in tasks:
        for b in tasks:
            if a != b and b not in adj[a]:
                cn = len(adj[a] & adj[b])
                if cn > 0:
                    scores.append((cn, a, b))
    scores.sort(reverse=True)
    return [(a, b) for _, a, b in scores[:kmax]]


def nograph_llm_pred(nodes, kmax, y):
    names = [nodes[i]["text"] for i in nodes]
    random.shuffle(names)
    sample = names[:120]
    sys_p = "You are a brain medical-imaging research strategist."
    usr = (f"Given research tasks/methods known up to {y}:\n- " + "\n- ".join(sample) +
           f"\n\nPropose the {kmax} most promising NOT-YET-TRIED transfer directions for {y+1} "
           "(form: source --> target task). Output STRICT JSON: "
           '{"directions":[{"src":"...","dst":"..."}]}')
    try:
        out = llm.extract_json(llm.call(sys_p, usr, max_tokens=4000))
        return [(d["src"], d["dst"]) for d in out.get("directions", [])][:kmax]
    except Exception as e:
        print("  nograph_llm failed:", e); return []


def to_embs(pred, text=False):
    if text:
        srcs = embed.embed([s for s, d in pred]); dsts = embed.embed([d for s, d in pred])
        return list(zip(srcs, dsts))
    return [(EMB[s], EMB[d]) for s, d in pred if s in EMB and d in EMB]


def score(pred_embs, truth):
    if not truth or not pred_embs:
        return {k: (0.0, 0.0) for k in KS}
    Ts = np.stack([EMB[s] for s, d in truth]); Td = np.stack([EMB[d] for s, d in truth])
    res = {}
    for k in KS:
        top = pred_embs[:k]
        hits = 0; cov = set()
        for vs, vd in top:
            m = np.where((Ts @ vs >= TAU) & (Td @ vd >= TAU))[0]
            if len(m): hits += 1
            cov.update(m.tolist())
        res[k] = (hits / max(1, len(top)), len(cov) / len(truth))
    return res


def run(T0):
    arms = ["growing", "frozen", "cooccur", "nograph_llm", "random"]
    acc = {a: {k: {"p": [], "r": []} for k in KS} for a in arms}
    gr_nodes, gr_edges = known_nodes(T0), known_edges(T0)
    fz_nodes, fz_edges = known_nodes(T0), known_edges(T0)
    for Y in range(T0 + 1, TEND + 1):
        truth = emerged(Y)
        preds = {
            "growing": to_embs([(s, d) for s, d, *_ in navigate(gr_nodes, gr_edges, EMB, KMAX)]),
            "frozen":  to_embs([(s, d) for s, d, *_ in navigate(fz_nodes, fz_edges, EMB, KMAX)]),
            "cooccur": to_embs(cooccur_pred(gr_nodes, KMAX, Y)),
            "nograph_llm": to_embs(nograph_llm_pred(gr_nodes, KMAX, Y), text=True),
            "random":  to_embs(_rand(gr_nodes, gr_edges, KMAX)),
        }
        for a in arms:
            r = score(preds[a], truth)
            for k in KS:
                acc[a][k]["p"].append(r[k][0]); acc[a][k]["r"].append(r[k][1])
        # grow
        for i, n in G.nodes.items():
            if n["first_year"] == Y: gr_nodes[i] = n
        gr_edges |= {(s, d) for (s, d) in truth
                     if not (G.nodes[s]["type"] == "method" and G.nodes[d]["type"] == "method")}
    return acc


def _rand(nodes, edges, kmax):
    kn = list(nodes); out = []
    while len(out) < kmax and len(kn) > 1:
        s, d = random.choice(kn), random.choice(kn)
        if s != d and (s, d) not in edges: out.append((s, d))
    return out


def report(T0):
    print(f"\n================ T0={T0} (mean±std over years {T0+1}..{TEND}) ================")
    acc = run(T0)
    print(f"{'arm':<12} | " + "  ".join(f"P@{k:<3} R@{k:<3}" for k in KS))
    for a in ["growing", "frozen", "cooccur", "nograph_llm", "random"]:
        cells = []
        for k in KS:
            p = np.mean(acc[a][k]["p"]); r = np.mean(acc[a][k]["r"])
            cells.append(f"{p:.2f}     {r:.2f}  ")
        print(f"{a:<12} | " + "".join(cells))


if __name__ == "__main__":
    for t in (int(x) for x in (sys.argv[1:] or [2019, 2020])):
        report(t)
