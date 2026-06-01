#!/usr/bin/env python3
"""P2: extract a year-stratified subsample into one raw task-transfer graph.

- <=2020: up to N0 papers (build G0); each of 2021..2024: up to NY (targets/ingest).
- deterministic subsample by hash(paperId) (no citation look-ahead).
- checkpoints every CKPT papers; resumable (skips already-processed paperIds).
Edges/nodes carry first_year so the eval harness can slice by year.
"""
import sys, json, re, hashlib, pathlib, time
import llm
from graph import Graph

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPERS = ROOT / "data" / "brain_papers.jsonl"
GRAPH_OUT = ROOT / "data" / "graph_raw.json"
DONE_OUT = ROOT / "data" / "processed_ids.json"
N0, NY, CKPT = 500, 100, 25

SYSTEM = """You build a TASK-TRANSFER graph for brain medical-image-analysis research.
From a single paper's title+abstract, extract:
1. TASK nodes  — research tasks/problems addressed (e.g., "brain tumor segmentation", "Alzheimer's classification from MRI").
2. METHOD nodes — methods/approaches used (e.g., "contrastive self-supervised pretraining", "U-Net", "vision transformer").
3. TRANSFER edges A->B — DIRECTED "A helps B" (knowledge/representation/pretraining from A benefits task B), ONLY when stated or clearly implied.
   e.g. "ImageNet pretraining improves tumor segmentation" (method->task,+); "joint training with registration hurts segmentation" (task->task,-); "SSL pretraining on unlabeled MRI boosts few-shot classification" (method->task,+).
Prefer canonical, reusable node names (e.g. "U-Net", not "our modified 3D U-Net with attention").
For each transfer edge: src, dst, src_type/dst_type ("task"|"method"), sign ("+"/"-"/"0"), strength 0..1, measured (true if a concrete quantitative gain is reported), evidence (SHORT VERBATIM span copied exactly from the text).
Rules: be conservative (no relation -> "transfers":[]); evidence must be verbatim; output STRICT JSON only."""

USER_TMPL = """TITLE: {title}

ABSTRACT: {abstract}

Return JSON:
{{"tasks":[{{"name":"...","domain":"..."}}],"methods":[{{"name":"..."}}],"transfers":[{{"src":"...","dst":"...","src_type":"task|method","dst_type":"task|method","sign":"+|-|0","strength":0.0,"measured":false,"evidence":"verbatim span"}}]}}"""


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def h(pid):
    return int(hashlib.md5(pid.encode()).hexdigest(), 16)


def subsample():
    rows = [json.loads(l) for l in PAPERS.read_text().splitlines()]
    strata = {"<=2020": [], 2021: [], 2022: [], 2023: [], 2024: []}
    for r in rows:
        y = r.get("year", 9999)
        k = "<=2020" if y <= 2020 else y
        if k in strata:
            strata[k].append(r)
    pick = []
    for k, lim in [("<=2020", N0), (2021, NY), (2022, NY), (2023, NY), (2024, NY)]:
        s = sorted(strata[k], key=lambda r: h(r["paperId"]))[:lim]
        pick += s
        print(f"  stratum {k}: {len(s)} (of {len(strata[k])})")
    return pick


def main():
    papers = subsample()
    print(f"total subsample: {len(papers)} papers")
    g = Graph.load(GRAPH_OUT) if GRAPH_OUT.exists() else Graph()
    done = set(json.loads(DONE_OUT.read_text())) if DONE_OUT.exists() else set()
    print(f"resuming: {len(done)} already processed\n")

    kept = dropped = 0
    for i, p in enumerate(papers):
        pid = p["paperId"]
        if pid in done:
            continue
        src_text = norm(p["title"] + " " + p["abstract"])
        try:
            out = llm.extract_json(llm.call(SYSTEM, USER_TMPL.format(
                title=p["title"], abstract=p["abstract"]), max_tokens=1500))
        except Exception as e:
            print(f"  [{i}] {pid[:8]} failed: {e}"); continue
        yr = p["year"]
        for t in out.get("tasks", []):
            g.add_node(t["name"], "task", t.get("domain"), yr, pid)
        for m in out.get("methods", []):
            g.add_node(m["name"], "method", None, yr, pid)
        for e in out.get("transfers", []):
            if norm(e.get("evidence", "")) not in src_text:
                dropped += 1; continue
            s = g.add_node(e["src"], e.get("src_type", "method"), None, yr, pid)
            d = g.add_node(e["dst"], e.get("dst_type", "task"), None, yr, pid)
            if s == d:
                dropped += 1; continue
            g.add_edge(s, d, e.get("sign", "+"), float(e.get("strength", 0.5)),
                       "verified" if e.get("measured") else "prior",
                       e["evidence"], yr, bool(e.get("measured")), pid)
            kept += 1
        done.add(pid)
        if len(done) % CKPT == 0:
            g.save(GRAPH_OUT); DONE_OUT.write_text(json.dumps(sorted(done)))
            print(f"  ckpt @ {len(done)} done | edges kept {kept}, dropped {dropped} | nodes {len(g.nodes)}")
        time.sleep(0.3)

    g.save(GRAPH_OUT); DONE_OUT.write_text(json.dumps(sorted(done)))
    print(f"\nDONE. processed {len(done)} papers. {json.dumps(g.stats())}")
    print(f"edges kept {kept}, dropped by post-check {dropped}")


if __name__ == "__main__":
    main()
