#!/usr/bin/env python3
"""P9: TRANSFER-PROMPT VALIDATION GATE (Step 1).
Diagnostic 30-paper sample; two-stage transfer-vs-usage extraction prompt;
INDEPENDENT judge audit; precision-first pass/fail gate. Does NOT full re-extract.
"""
import sys, json, re, time, pathlib, hashlib, collections
import llm
from graph import Graph

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPERS = {json.loads(l)["paperId"]: json.loads(l)
          for l in (ROOT / "data" / "brain_papers.jsonl").read_text().splitlines()}
RAW = Graph.load(ROOT / "data" / "graph_raw.json")

# ---- diagnostic sample selection (NOT random) ----
def paper_edge_counts():
    c = collections.defaultdict(lambda: {"mt": 0, "tt": 0, "year": None})
    for (s, d), e in RAW.edges.items():
        t = (RAW.nodes[s]["type"], RAW.nodes[d]["type"])
        for pid in e["paper_refs"]:
            if t == ("method", "task"): c[pid]["mt"] += 1
            elif t == ("task", "task"): c[pid]["tt"] += 1
            c[pid]["year"] = e["first_year"]
    return c

def select():
    c = paper_edge_counts()
    have = lambda pid: pid in PAPERS and PAPERS[pid].get("abstract")
    A = [p for p, _ in sorted(c.items(), key=lambda kv: -kv[1]["mt"]) if have(p)][:10]
    B = [p for p, _ in sorted(c.items(), key=lambda kv: -kv[1]["tt"]) if have(p) and p not in A][:10]
    recent = [p for p in PAPERS if PAPERS[p].get("year", 0) >= 2022 and have(p)]
    recent.sort(key=lambda p: int(hashlib.md5(p.encode()).hexdigest(), 16))
    C = [p for p in recent if p not in A and p not in B][:10]
    return [("A:method-heavy", A), ("B:task->task-heavy", B), ("C:recent22-24", C)]

# ---- transfer-purified extraction prompt (per spec) ----
EXTRACT_SYS = """You are extracting a TRUE RESEARCH TRANSFER GRAPH from a paper title and abstract.
Extract only relations where knowledge, representations, models, methods, datasets, features, pretraining, objectives, or insights from one research problem/domain/task/method are reused, adapted, transferred, or shown to help another.
Do NOT extract a relation just because a method is used for a task.
TRUE_TRANSFER = source A contributes reusable knowledge/representation/model/dataset/features/training-signal/inductive-bias/objective/insight to target B (A helps/improves/enables/adapts-to/pretrains-then-benefits/bridges B).
USAGE_ONLY = the paper merely applies method A to task B ("we use CNN for tumor segmentation", "U-Net for lesion detection") with no reuse/adaptation/transfer from another source.
Return JSON only:
{"transfer_edges":[{"source":"...","target":"...","sign":"+|-","mechanism":"pretraining|representation_reuse|feature_transfer|domain_adaptation|multi_task_learning|knowledge_distillation|dataset_transfer|objective_transfer|methodological_insight|other","evidence":"verbatim short span","rationale":"one sentence"}]}
Rules: every evidence must be an exact substring of title+abstract. Only TRUE_TRANSFER or NEGATIVE_TRANSFER go in transfer_edges. If the paper only says a method is used for a task, transfer_edges must be []. Be conservative: better to output nothing than a usage-only edge."""

# ---- independent judge (separate rubric) ----
JUDGE_SYS = """You audit candidate research-transfer edges. For each "SRC --> DST" with its evidence, label it:
- "true_transfer": the evidence shows knowledge/representation/pretraining/features/multi-task signal from SRC actually helps/transfers to DST.
- "usage_only": SRC is merely a method applied to task DST (no transfer/reuse from elsewhere).
- "too_vague": evidence is only a buzzword (e.g. "transfer learning") without a concrete source->target benefit.
- "wrong_direction": a real transfer exists but the direction SRC->DST is reversed.
Be strict. Output JSON: {"labels":[{"i":0,"label":"..."}]} in order."""

def norm(s): return re.sub(r"\s+", " ", (s or "").lower()).strip()


def main():
    groups = select()
    per_paper = []          # (group, pid, title, n_edges, edges[list])
    all_edges = []          # (pid, src, dst, mech, evidence, evidence_valid)
    for gname, pids in groups:
        for pid in pids:
            p = PAPERS[pid]
            src = norm(p["title"] + " " + p["abstract"])
            usr = f"Title:\n{p['title']}\n\nAbstract:\n{p['abstract']}"
            try:
                out = llm.extract_json(llm.call(EXTRACT_SYS, usr, max_tokens=2500))
                edges = out.get("transfer_edges", []) or []
            except Exception as e:
                edges = []; print(f"  extract fail {pid[:8]}: {e}")
            valid = []
            for e in edges:
                ok = norm(e.get("evidence", "")) in src
                all_edges.append((pid, e.get("source"), e.get("target"),
                                  e.get("mechanism"), e.get("evidence", ""), ok))
                if ok: valid.append(e)
            per_paper.append((gname, pid, p["title"][:50], len(valid), valid))
            time.sleep(0.3)

    # evidence validity
    n_total = len(all_edges); n_valid = sum(1 for *_, ok in all_edges if ok)
    valid_edges = [(pid, s, d, m, ev) for (pid, s, d, m, ev, ok) in all_edges if ok]

    # independent judge audit (batched)
    labels = []
    for b in range(0, len(valid_edges), 8):
        chunk = valid_edges[b:b+8]
        lines = [f'{i}. SRC="{s}" DST="{d}" EVIDENCE="{ev[:160]}"' for i, (_, s, d, m, ev) in enumerate(chunk)]
        try:
            out = llm.extract_json(llm.call(JUDGE_SYS, "Label each:\n" + "\n".join(lines), max_tokens=700))
            mp = {x["i"]: x["label"] for x in out.get("labels", [])}
        except Exception as e:
            print("  judge fail:", e); mp = {}
        for i, edge in enumerate(chunk):
            labels.append((edge, mp.get(i, "too_vague")))

    cnt = collections.Counter(l for _, l in labels)
    n = len(labels) or 1
    empty = sum(1 for *_, ln, _ in per_paper for _ in [0] if ln == 0)  # placeholder
    empty_papers = sum(1 for (_, _, _, ne, _) in per_paper if ne == 0)
    n_papers = len(per_paper)

    print("\n================ TRANSFER-PROMPT VALIDATION GATE ================")
    print(f"papers: {n_papers}  | extracted transfer_edges (valid evidence): {len(valid_edges)}")
    print(f"\n-- per-paper --")
    for gname, pid, title, ne, _ in per_paper:
        print(f"  [{gname:18}] {ne} edges | {title}")
    print(f"\n-- independent judge labels (n={len(labels)}) --")
    for k in ("true_transfer", "usage_only", "too_vague", "wrong_direction"):
        print(f"   {k:<16}: {cnt.get(k,0):2d}  ({100*cnt.get(k,0)/n:.0f}%)")

    prec = cnt.get("true_transfer", 0) / n
    usage_leak = cnt.get("usage_only", 0) / n
    ev_valid = n_valid / (n_total or 1)
    empty_rate = empty_papers / n_papers
    print(f"\n-- GATE METRICS --")
    print(f"   true_transfer precision : {prec:.0%}")
    print(f"   usage leakage           : {usage_leak:.0%}")
    print(f"   evidence validity       : {ev_valid:.0%}  ({n_valid}/{n_total})")
    print(f"   empty_rate (papers w/0) : {empty_rate:.0%}")
    verdict = ("PASS -> proceed to full re-extraction" if prec >= 0.70 and usage_leak <= 0.25 and ev_valid >= 0.95
               else "TUNE PROMPT, retry" if prec >= 0.50
               else "DO NOT full re-extract (model can't separate use vs transfer)")
    print(f"\n   >>> VERDICT: {verdict}")

    print("\n-- sample true_transfer edges --")
    shown = 0
    for (pid, s, d, m, ev), lab in labels:
        if lab == "true_transfer" and shown < 8:
            print(f"   {s} --[{m}]--> {d}\n      \"{ev[:80]}\""); shown += 1
    print("\n-- sample edges the judge REJECTED (usage/vague/wrong) --")
    shown = 0
    for (pid, s, d, m, ev), lab in labels:
        if lab != "true_transfer" and shown < 6:
            print(f"   [{lab}] {s} --> {d}"); shown += 1


if __name__ == "__main__":
    main()
