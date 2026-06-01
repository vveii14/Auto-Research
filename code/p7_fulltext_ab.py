#!/usr/bin/env python3
"""P7-test: does FULL TEXT thicken task->task transfer edges vs abstract-only?
A/B on the same papers (those with a PubMedCentral id). Transfer-targeted prompt.
Measures task->task edge count per paper under (A) abstract-only, (B) abstract+fulltext.
"""
import sys, json, re, time, pathlib, urllib.request
import xml.etree.ElementTree as ET
import llm

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPERS = ROOT / "data" / "brain_papers.jsonl"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&rettype=xml&id="

SYSTEM = """You extract a TASK-TRANSFER graph for brain medical-image-analysis research.
Identify TASK nodes and METHOD nodes, and especially TRANSFER edges A->B meaning "A helps B".
PAY SPECIAL ATTENTION to transfer relations that papers state in their methods/experiments:
 - pretraining / self-supervised pretraining on one task or dataset helping another task
 - transfer learning / fine-tuning / domain adaptation across tasks or modalities
 - multi-task / joint / auxiliary-task training where one task helps (or hurts) another
 - representations learned for task A reused to improve task B
For each transfer edge: src, dst, src_type/dst_type ("task"|"method"), sign ("+"/"-"/"0"),
strength 0..1, measured (true if a concrete gain is reported), evidence (SHORT VERBATIM span).
Be conservative; evidence must be copied verbatim; output STRICT JSON only:
{"tasks":[{"name":"..."}],"methods":[{"name":"..."}],"transfers":[{"src":"...","dst":"...","src_type":"task|method","dst_type":"task|method","sign":"+|-|0","strength":0.0,"measured":false,"evidence":"verbatim"}]}"""


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def fetch_fulltext(pmcid):
    pmcid = pmcid.replace("PMC", "")
    try:
        with urllib.request.urlopen(EFETCH + pmcid, timeout=60) as r:
            xml = r.read().decode("utf-8", "ignore")
        root = ET.fromstring(xml)
        body = root.find(".//body")
        if body is None:
            return None
        txt = " ".join(body.itertext())
        return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        return None


def extract(text):
    try:
        out = llm.extract_json(llm.call(SYSTEM, text[:14000], max_tokens=2500))
        return out
    except Exception:
        return {"tasks": [], "methods": [], "transfers": []}


def count_tt(out, src_text):
    """count valid task->task transfer edges (evidence verbatim)."""
    tt = total = 0
    for e in out.get("transfers", []):
        if norm(e.get("evidence", "")) not in norm(src_text):
            continue
        total += 1
        if e.get("src_type") == "task" and e.get("dst_type") == "task":
            tt += 1
    return tt, total


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    rows = [json.loads(l) for l in PAPERS.read_text().splitlines()]
    rows = [r for r in rows if (r.get("externalIds") or {}).get("PubMedCentral")]
    rows = rows[:200]  # pool; we'll take first n with retrievable full text

    done = 0
    agg = {"abs": {"tt": 0, "tot": 0}, "full": {"tt": 0, "tot": 0}}
    for r in rows:
        if done >= n:
            break
        pmc = (r["externalIds"]["PubMedCentral"])
        ft = fetch_fulltext(pmc); time.sleep(0.4)
        if not ft or len(ft) < 800:
            continue
        abs_text = f"TITLE: {r['title']}\n\nABSTRACT: {r['abstract']}"
        full_text = abs_text + "\n\nFULL TEXT (excerpt):\n" + ft
        oa = extract(abs_text); of = extract(full_text)
        a_tt, a_tot = count_tt(oa, abs_text)
        f_tt, f_tot = count_tt(of, full_text)
        agg["abs"]["tt"] += a_tt; agg["abs"]["tot"] += a_tot
        agg["full"]["tt"] += f_tt; agg["full"]["tot"] += f_tot
        done += 1
        print(f"[{done}] {r['title'][:45]:<45} | abstract: {a_tt} t->t /{a_tot}  | fulltext: {f_tt} t->t /{f_tot}")

    print(f"\n==== A/B over {done} papers ====")
    for k in ("abs", "full"):
        tt, tot = agg[k]["tt"], agg[k]["tot"]
        pct = 100 * tt / tot if tot else 0
        label = "abstract-only" if k == "abs" else "abstract+FULLTEXT"
        print(f"  {label:<18}: {tt} task->task edges / {tot} total  ({pct:.0f}% task->task), "
              f"{tt/done:.2f} t->t per paper")


if __name__ == "__main__":
    main()
