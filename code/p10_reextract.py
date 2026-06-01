#!/usr/bin/env python3
"""P10 (Step 2): full re-extraction of the 886-paper subsample with the
transfer-purified prompt. Transfer edges ONLY, with node types. Resumable.
"""
import json, re, hashlib, pathlib, time
import llm
from graph import Graph

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPERS = ROOT / "data" / "brain_papers.jsonl"
GOUT = ROOT / "data" / "graph_transfer_raw.json"
DONE = ROOT / "data" / "processed_transfer_ids.json"
N0, NY, CKPT = 500, 100, 25

SYS = """You are extracting a TRUE RESEARCH TRANSFER GRAPH from a paper title and abstract.
Extract only relations where knowledge, representations, models, methods, datasets, features, pretraining, objectives, or insights from one research problem/domain/task/method are reused, adapted, transferred, or shown to help another.
Do NOT extract a relation just because a method is used for a task.
TRUE_TRANSFER = source A contributes reusable knowledge/representation/model/dataset/features/training-signal/inductive-bias/objective/insight to target B (A helps/improves/enables/adapts-to/pretrains-then-benefits/bridges B).
USAGE_ONLY (DO NOT OUTPUT) = the paper merely applies method A to task B ("we use CNN for tumor segmentation", "U-Net for lesion detection").
For each TRUE_TRANSFER (or NEGATIVE_TRANSFER) relation give source, target, their types, sign, mechanism, and a verbatim evidence span.
Return JSON only:
{"transfer_edges":[{"source":"...","source_type":"task|method|dataset|representation|objective|domain","target":"...","target_type":"task|method|dataset|representation|objective|domain","sign":"+|-","mechanism":"pretraining|representation_reuse|feature_transfer|domain_adaptation|multi_task_learning|knowledge_distillation|dataset_transfer|objective_transfer|methodological_insight|other","evidence":"verbatim short span"}]}
Rules: evidence must be an exact substring of title+abstract. Only TRUE_TRANSFER/NEGATIVE_TRANSFER. If the paper only applies a method to a task, return {"transfer_edges":[]}. Be conservative: better empty than a usage edge."""


def norm(s): return re.sub(r"\s+", " ", (s or "").lower()).strip()
def h(pid): return int(hashlib.md5(pid.encode()).hexdigest(), 16)
def ntype(t): return "task" if t == "task" else "method"  # collapse non-task sources to method-like


def subsample():
    rows = [json.loads(l) for l in PAPERS.read_text().splitlines()]
    strata = {"<=2020": [], 2021: [], 2022: [], 2023: [], 2024: []}
    for r in rows:
        y = r.get("year", 9999); k = "<=2020" if y <= 2020 else y
        if k in strata: strata[k].append(r)
    pick = []
    for k, lim in [("<=2020", N0), (2021, NY), (2022, NY), (2023, NY), (2024, NY)]:
        pick += sorted(strata[k], key=lambda r: h(r["paperId"]))[:lim]
    return pick


def main():
    papers = subsample()
    g = Graph.load(GOUT) if GOUT.exists() else Graph()
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    print(f"subsample {len(papers)}; resuming {len(done)} done")
    kept = dropped = 0
    for i, p in enumerate(papers):
        pid = p["paperId"]
        if pid in done: continue
        src = norm(p["title"] + " " + p["abstract"])
        try:
            out = llm.extract_json(llm.call(SYS, f"Title:\n{p['title']}\n\nAbstract:\n{p['abstract']}", max_tokens=2000))
            edges = out.get("transfer_edges", []) or []
        except Exception as e:
            print(f"  fail {pid[:8]}: {e}"); continue
        yr = p["year"]
        for e in edges:
            if norm(e.get("evidence", "")) not in src: dropped += 1; continue
            s = g.add_node(e["source"], ntype(e.get("source_type", "method")), None, yr, pid)
            d = g.add_node(e["target"], ntype(e.get("target_type", "task")), None, yr, pid)
            if s == d: dropped += 1; continue
            g.add_edge(s, d, e.get("sign", "+"), float(e.get("confidence", 0.7)) if e.get("confidence") else 0.7,
                       "verified", e["evidence"], yr, True, pid)
            kept += 1
        done.add(pid)
        if len(done) % CKPT == 0:
            g.save(GOUT); DONE.write_text(json.dumps(sorted(done)))
            print(f"  ckpt {len(done)} | transfer edges {kept}, dropped {dropped}, nodes {len(g.nodes)}")
        time.sleep(0.3)
    g.save(GOUT); DONE.write_text(json.dumps(sorted(done)))
    print(f"DONE {len(done)} papers | {json.dumps(g.stats())} | dropped {dropped}")


if __name__ == "__main__":
    main()
