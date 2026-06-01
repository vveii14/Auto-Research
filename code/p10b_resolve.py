#!/usr/bin/env python3
"""P10b: entity-resolve the transfer-only graph. NO transitive inference
(truth must stay = extracted transfer edges). Reports per-year yield."""
import json, pathlib, collections
import p2_resolve
from graph import Graph

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "graph_transfer_raw.json"
OUT = ROOT / "data" / "graph_transfer_resolved.json"


def main():
    g = Graph.load(RAW)
    print("raw transfer graph:", json.dumps(g.stats()))
    g = p2_resolve.resolve(g)            # entity resolution only
    print("after resolution:", json.dumps(g.stats()))
    print("edge types:", json.dumps(p2_resolve.edge_type_dist(g)))
    # per-year emerged transfer-edge counts (the yield watch)
    yr = collections.Counter(e["first_year"] for e in g.edges.values() if e["first_year"])
    print("transfer edges by first_year:", {y: yr[y] for y in sorted(yr)})
    tt = sum(1 for (s, d) in g.edges if g.nodes[s]["type"] == "task" and g.nodes[d]["type"] == "task")
    print(f"task->task transfer edges: {tt} / {len(g.edges)} ({100*tt/max(1,len(g.edges)):.0f}%)")
    g.save(OUT); print("saved", OUT)


if __name__ == "__main__":
    main()
