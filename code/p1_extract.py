#!/usr/bin/env python3
"""P1: extract task/method nodes + directed transfer edges from paper abstracts.

Sanity slice: run on N papers, apply a deterministic post-check (evidence span
must appear verbatim in the source), and print samples for manual inspection.
"""
import sys, json, re, pathlib, time
import llm
from graph import Graph

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPERS = ROOT / "data" / "brain_papers.jsonl"

SYSTEM = """You build a TASK-TRANSFER graph for brain medical-image-analysis research.
From a single paper's title+abstract, extract:
1. TASK nodes  — the research tasks/problems it addresses (e.g., "brain tumor segmentation", "Alzheimer's classification from MRI").
2. METHOD nodes — the methods/approaches it uses (e.g., "contrastive self-supervised pretraining", "U-Net", "vision transformer").
3. TRANSFER edges A->B — a DIRECTED relation meaning "A helps B" (knowledge/representation/pretraining from A benefits task B), ONLY when the abstract states or clearly implies it.
   Examples of a transfer edge: "ImageNet pretraining improves tumor segmentation" (method->task, +); "multi-task training with registration hurts segmentation" (task->task, -); "self-supervised pretraining on unlabeled MRI boosts few-shot classification" (method->task, +).

For each transfer edge give:
- src, dst (use the exact node names you listed)
- src_type, dst_type ("task" or "method")
- sign: "+" (helps), "-" (hurts/conflicts), "0" (neutral/no effect)
- strength: 0..1 (how strong the stated benefit is)
- measured: true if the abstract reports a concrete quantitative improvement from this transfer, else false
- evidence: a SHORT VERBATIM span copied exactly from the title+abstract that supports the edge

Rules:
- Be conservative: if no transfer relation is stated, return "transfers": []. Many papers contribute only nodes.
- evidence MUST be copied verbatim (character-for-character) from the provided text.
- Output STRICT JSON only, no commentary."""

USER_TMPL = """TITLE: {title}

ABSTRACT: {abstract}

Return JSON:
{{"tasks":[{{"name":"...","domain":"..."}}],
 "methods":[{{"name":"..."}}],
 "transfers":[{{"src":"...","dst":"...","src_type":"task|method","dst_type":"task|method","sign":"+|-|0","strength":0.0,"measured":false,"evidence":"verbatim span"}}]}}"""


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    max_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2020
    rows = [json.loads(l) for l in PAPERS.read_text().splitlines()]
    rows = [r for r in rows if r.get("year", 9999) <= max_year][:n]
    print(f"extracting from {len(rows)} papers (year<= {max_year})\n")

    g = Graph()
    kept_edges = dropped = n_calls = 0
    samples = []
    for i, p in enumerate(rows):
        src_text = norm(p["title"] + " " + p["abstract"])
        try:
            out = llm.extract_json(llm.call(SYSTEM, USER_TMPL.format(
                title=p["title"], abstract=p["abstract"]), max_tokens=1500))
            n_calls += 1
        except Exception as e:
            print(f"  [{i}] extract failed: {e}"); continue

        yr = p["year"]; pid = p["paperId"]
        for t in out.get("tasks", []):
            g.add_node(t["name"], "task", t.get("domain"), yr, pid)
        for m in out.get("methods", []):
            g.add_node(m["name"], "method", None, yr, pid)
        for e in out.get("transfers", []):
            if norm(e.get("evidence", "")) not in src_text:  # deterministic post-check
                dropped += 1; continue
            s = g.add_node(e["src"], e.get("src_type", "method"), None, yr, pid)
            d = g.add_node(e["dst"], e.get("dst_type", "task"), None, yr, pid)
            if s == d:
                dropped += 1; continue
            g.add_edge(s, d, e.get("sign", "+"), float(e.get("strength", 0.5)),
                       "verified" if e.get("measured") else "prior",
                       e["evidence"], yr, bool(e.get("measured")), pid)
            kept_edges += 1
            if len(samples) < 12:
                samples.append((p["title"][:55], e["src"], e.get("sign"), e["dst"], e["evidence"][:90]))
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1} papers, {kept_edges} edges kept, {dropped} dropped")
        time.sleep(0.3)

    g.save(ROOT / "data" / "graph_p1.json")
    print("\n=== extraction stats ===")
    print(json.dumps(g.stats(), indent=1))
    print(f"transfer edges: kept {kept_edges}, dropped by post-check {dropped}, LLM calls {n_calls}")
    print("\n=== sample transfer edges (eyeball these) ===")
    for title, s, sign, d, ev in samples:
        print(f"\n[{title}]\n  {s}  --({sign})-->  {d}\n  evidence: \"{ev}\"")


if __name__ == "__main__":
    main()
