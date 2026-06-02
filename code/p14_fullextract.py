#!/usr/bin/env python3
"""P14: scale the transfer-purified extraction to ALL 8,000 papers (resumes from
the 886 already done). Builds the full transfer graph for a tighter hub/compounding test."""
import json, re, pathlib, time
import llm
from graph import Graph
from p10_reextract import SYS, norm, ntype   # reuse the validated transfer prompt

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPERS = ROOT / "data" / "brain_papers.jsonl"
GOUT = ROOT / "data" / "graph_transfer_raw.json"
DONE = ROOT / "data" / "processed_transfer_ids.json"
CKPT = 50


def main():
    papers = [json.loads(l) for l in PAPERS.read_text().splitlines()]
    g = Graph.load(GOUT) if GOUT.exists() else Graph()
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    todo = [p for p in papers if p["paperId"] not in done]
    print(f"total {len(papers)} | already done {len(done)} | todo {len(todo)}")
    kept = dropped = 0
    for i, p in enumerate(todo):
        pid = p["paperId"]
        if not p.get("abstract"):
            done.add(pid); continue
        src = norm(p["title"] + " " + p["abstract"])
        try:
            out = llm.extract_json(llm.call(SYS, f"Title:\n{p['title']}\n\nAbstract:\n{p['abstract']}", max_tokens=2000))
            edges = out.get("transfer_edges", []) or []
        except Exception as e:
            print(f"  fail {pid[:8]}: {e}"); continue
        yr = p["year"]
        for e in edges:
            if norm(e.get("evidence", "")) not in src:
                dropped += 1; continue
            s = g.add_node(e["source"], ntype(e.get("source_type", "method")), None, yr, pid)
            d = g.add_node(e["target"], ntype(e.get("target_type", "task")), None, yr, pid)
            if s == d:
                dropped += 1; continue
            g.add_edge(s, d, e.get("sign", "+"), 0.7, "verified", e["evidence"], yr, True, pid)
            kept += 1
        done.add(pid)
        if len(done) % CKPT == 0:
            g.save(GOUT); DONE.write_text(json.dumps(sorted(done)))
            print(f"  ckpt {len(done)} | new edges {kept}, dropped {dropped}, nodes {len(g.nodes)}, total edges {len(g.edges)}")
        time.sleep(0.25)
    g.save(GOUT); DONE.write_text(json.dumps(sorted(done)))
    print(f"DONE {len(done)} | {json.dumps(g.stats())}")


if __name__ == "__main__":
    main()
