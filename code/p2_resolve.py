#!/usr/bin/env python3
"""P2-resolve: entity resolution (merge near-duplicate nodes) + transitive
inference -> resolved graph. Reports edge-type distribution (the thesis check:
how many real task->task transfer edges?). Keeps first_year for temporal slicing.
"""
import json, pathlib, numpy as np
import embed
from graph import Graph

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "graph_raw.json"
OUT = ROOT / "data" / "graph_resolved.json"
SIM = 0.86  # cosine threshold to merge same-type nodes


def resolve(g):
    # cluster near-duplicate nodes within each type
    canon = {}  # old id -> canonical id
    for ntype in ("task", "method"):
        ns = [n for n in g.nodes.values() if n["type"] == ntype]
        if not ns:
            continue
        E = embed.embed([n["text"] for n in ns])
        S = E @ E.T
        order = sorted(range(len(ns)), key=lambda i: -ns[i]["freq"])  # high-freq = canonical
        assigned = {}
        for i in order:
            if ns[i]["id"] in assigned:
                continue
            assigned[ns[i]["id"]] = ns[i]["id"]
            for j in range(len(ns)):
                if ns[j]["id"] not in assigned and S[i, j] >= SIM:
                    assigned[ns[j]["id"]] = ns[i]["id"]
        canon.update(assigned)

    ng = Graph()
    # rebuild nodes
    for nid, n in g.nodes.items():
        c = canon.get(nid, nid)
        cn = g.nodes[c]
        m = ng.nodes.get(c)
        if m is None:
            m = dict(cn); m["paper_refs"] = list(cn["paper_refs"]); ng.nodes[c] = m
        if c != nid:
            m["freq"] += n["freq"]
            for pr in n["paper_refs"]:
                if pr not in m["paper_refs"]:
                    m["paper_refs"].append(pr)
            if n["first_year"] is not None:
                m["first_year"] = min(m["first_year"] or n["first_year"], n["first_year"])
    # remap edges
    for (s, d), e in g.edges.items():
        cs, cd = canon.get(s, s), canon.get(d, d)
        if cs == cd:
            continue
        ng.add_edge(cs, cd, e["sign"], e["strength"], e["state"], e["evidence"],
                    e["first_year"], e["measured"])
    return ng


def transitive(g):
    pos = {}
    for (s, d), e in g.edges.items():
        if e["sign"] == "+":
            pos.setdefault(s, []).append((d, e["first_year"]))
    added = 0
    for a, bs in list(pos.items()):
        for b, yab in bs:
            for c, ybc in pos.get(b, []):
                if a != c and (a, c) not in g.edges:
                    yr = max(x for x in (yab, ybc) if x is not None) if (yab or ybc) else None
                    g.add_edge(a, c, "+", 0.3, "inferred", "", yr)
                    added += 1
    return added


def edge_type_dist(g):
    d = {}
    for (s, dd), e in g.edges.items():
        t = f'{g.nodes[s]["type"]}->{g.nodes[dd]["type"]}'
        d[t] = d.get(t, 0) + 1
    return d


def main():
    g = Graph.load(RAW)
    print("raw:", json.dumps(g.stats()))
    g = resolve(g)
    print("after entity resolution:", json.dumps(g.stats()))
    print("edge types (extracted):", json.dumps(edge_type_dist(g)))
    n_inf = transitive(g)
    print(f"transitive inference added {n_inf} edges")
    print("after inference:", json.dumps(g.stats()))
    print("edge types (with inferred):", json.dumps(edge_type_dist(g)))
    g.save(OUT)
    # THE thesis check
    tt = sum(1 for (s, d) in g.edges if g.nodes[s]["type"] == "task" and g.nodes[d]["type"] == "task")
    print(f"\n*** task->task transfer edges (who-helps-whom signal): {tt} ***")


if __name__ == "__main__":
    main()
