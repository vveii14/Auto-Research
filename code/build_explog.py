#!/usr/bin/env python3
"""Complete step-by-step experiment log PDF (P0 -> P11)."""
import pathlib, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, KeepTogether)

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "data" / "figs"
OUT = ROOT / "Complete_Experiment_Log.pdf"
SERIF, SANS = "Times-Roman", "Helvetica-Bold"

def st(name, **kw):
    b = dict(fontName=SERIF, fontSize=10, leading=14, spaceAfter=5); b.update(kw)
    return ParagraphStyle(name, **b)
TITLE = st("T", fontName=SANS, fontSize=20, leading=24, textColor=colors.HexColor("#0b3d91"))
SUB = st("S", fontSize=12, textColor=colors.HexColor("#57606a"))
H = st("H", fontName=SANS, fontSize=13, leading=16, textColor=colors.HexColor("#0b3d91"), spaceBefore=10, spaceAfter=5)
LBL = st("L", fontName=SANS, fontSize=9.5, textColor=colors.HexColor("#1f6feb"), spaceAfter=1)
BODY = st("B")
SMALL = st("SM", fontSize=8.5, leading=11, textColor=colors.HexColor("#57606a"))

story = []
def P(t, s=BODY): story.append(Paragraph(t, s))
def sp(h=5): story.append(Spacer(1, h))
def fig(name, w=14):
    p = FIG / name
    if not p.exists(): return
    iw, ih = ImageReader(str(p)).getSize()
    story.append(Image(str(p), width=w*cm, height=w*cm*ih/iw)); sp(3)
def tbl(data, widths, fs=8.5):
    t = Table(data, colWidths=[w*cm for w in widths])
    t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),SERIF),("FONTNAME",(0,0),(-1,0),SANS),
        ("FONTSIZE",(0,0),(-1,-1),fs),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0b3d91")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c9d1d9")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f6f8fa")]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(1,0),(-1,-1),"CENTER")]))
    story.append(t); sp(6)

def exp(tag, title, goal, did, result_rows, result_widths, takeaway, figname=None, figw=14, fs=8.5):
    P(f"{tag} &mdash; {title}", H)
    P("<b>Goal.</b> " + goal, BODY)
    P("<b>What we did.</b> " + did, BODY)
    if result_rows: tbl(result_rows, result_widths, fs)
    if figname: fig(figname, figw)
    P("<b>Takeaway.</b> " + takeaway, BODY)
    sp(6)

# ---------- title ----------
P("Complete Experiment Log", TITLE)
P("A Living Task-Transfer Graph for Auto-Research &mdash; pilot, brain medical imaging", SUB)
P("2026-06-01 &middot; github.com/vveii14/Auto-Research", SMALL); sp(8)
fig("f8_journey.png", 17)
P("Figure. The whole arc in three regimes: (A) on the usage-heavy graph the graph navigation wins; "
  "(B) after purifying to genuine-transfer edges only, the graph collapses to 0; "
  "(C) under a region-level reformulation the LLM works but the graph does not help.", SMALL)
story.append(PageBreak())

# ---------- overview ----------
P("Overview of all experiments", H)
tbl([["Exp", "Question", "Headline result"],
     ["P0", "Can we get a timestamped corpus?", "8,000 CS brain-imaging papers, 2015-2024"],
     ["P1", "Does LLM extraction work?", "yes - 40-paper sanity passes"],
     ["P2", "Build the graph", "2,907 nodes / 1,748 edges; task->task only 8%"],
     ["P4", "Temporal compounding eval", "exact-match metric -> all zeros (too strict)"],
     ["P4b", "Fix metric (semantic)", "nonzero; compounding at small base"],
     ["P5", "k-sweep + 5 baselines", "growing >= frozen > cooccur > no-graph > random"],
     ["P6", "Bootstrap CIs + significance", "graph>baselines SIGNIFICANT; compounding NOT"],
     ["P7", "Full text vs abstract", "prompt is the lever (8%->32%), not full text"],
     ["P8", "Edge quality audit", "method->task 78% usage; only ~23% genuine transfer"],
     ["P9", "Transfer-prompt gate (30)", "100% precision, 0% usage, but 83% empty"],
     ["P10", "Purify + re-eval (886)", "1,748->253 edges; graph arms -> 0 (collapse)"],
     ["P11", "Region reformulation", "tractable; LLM beats random; graph HURTS"]],
    [1.3, 7, 8.4], fs=8.5)
story.append(PageBreak())

# ---------- P0 ----------
exp("P0", "Corpus acquisition",
    "Obtain a timestamped brain medical-imaging corpus (corpus scope is a tunable parameter).",
    "Semantic Scholar bulk search. The first query returned mostly clinical neuro-oncology (only 1,013/8,000 were Computer Science); re-queried with fieldsOfStudy=Computer Science + method terms.",
    [["", "papers", "venues found"], ["result", "8,000 (abstracts, 2015-2024)", "arXiv, NeuroImage, ISBI, MICCAI, TMI, MedIA, SPIE"]],
    [2, 5, 7.7],
    "Corpus is a knob; with the CS filter we get the methods community, not clinical papers.",
    "f1_years.png", 13)

# ---------- P1 ----------
exp("P1", "Extraction sanity (40 papers)",
    "Validate the most uncertain component (LLM edge extraction) cheaply before scaling.",
    "Run extraction on 40 papers (<=2020): task/method nodes + signed transfer edges + a deterministic post-check that the evidence span is a verbatim substring of the source.",
    [["task nodes", "method nodes", "transfer edges", "dropped by post-check"],
     ["75", "131", "26", "0"]],
    [3.5, 3.5, 3.5, 4.2],
    "Extraction produces grounded, sensible edges (e.g. 'mixture-density network -> tumor growth estimation'). PASS.",
    None)

# ---------- P2 ----------
exp("P2", "Build the graph (886-paper subsample)",
    "Construct the initial pilot graph.",
    "Extract an 886-paper subsample (500 from <=2020 + 100/yr 2021-24), entity-resolve near-duplicate nodes (MiniLM, cos>=0.86), then transitive inference.",
    [["nodes", "edges", "method->task", "task->task", "method->method", "+inferred"],
     ["2,907", "1,748", "1,581", "131 (8%)", "36", "+518"]],
    [2.2, 2.2, 3, 2.6, 3, 2.3], fs=8,
    takeaway="Graph builds cleanly, but task->task (the who-helps-whom edge) is only ~8% - first warning sign.",
    figname="f2_graph.png", figw=14)
story.append(PageBreak())

# ---------- P4 ----------
exp("P4", "First temporal eval (exact match)",
    "Test compounding: build graph <=T0, predict edges emerging in T0+1, score, then ingest.",
    "Require an exact (source,target) node-id match among 2,907 nodes for a hit.",
    [["growing", "frozen", "random", "diagnosis"],
     ["0.000", "0.000", "0.000", "only 5-13% of future edges are even predictable; 90% add a new node"]],
    [2.2, 2.2, 2.2, 8.4],
    "All zeros - but this is a metric problem (exact-id match is hopeless), not absence of signal.",
    None)

# ---------- P4b ----------
exp("P4b", "Semantic-matching fix + T0 sweep",
    "Make the metric measurable: a prediction hits if it is embedding-close to an emerged edge.",
    "Semantic match (cos>=0.60 on both endpoints). Sweep the start year T0 from 2017 to 2020.",
    [["T0", "2017", "2018", "2019", "2020"],
     ["growing", "0.080", "0.090", "0.100", "0.065"],
     ["frozen", "0.000", "0.107", "0.072", "0.085"]],
    [2.8, 2.8, 2.8, 2.8, 2.8],
    "Nonzero now. Compounding (growing>frozen) shows at small base (T0=2017, 2019), ties at large base.",
    "f4_t0sweep.png", 13)
story.append(PageBreak())

# ---------- P5 ----------
exp("P5", "Hardened eval: k-sweep + 5 arms",
    "Compare all baselines across many k on the same temporal-forecast task.",
    "Arms: growing / frozen / co-occurrence / no-graph-LLM / random; semantic precision@k for k in {5,10,20,50,100}.",
    [["P@10", "growing", "frozen", "cooccur", "no-graph LLM", "random"],
     ["mean", "0.127", "0.091", "0.027", "0.023", "0.000"]],
    [2, 2.4, 2.2, 2.2, 3, 2.2], fs=8,
    takeaway="Consistent ordering: growing >= frozen > co-occurrence > no-graph-LLM > random. The graph beats everything; the LLM alone is near-random.",
    figname="f3_precision.png", figw=14.5)

# ---------- P6 ----------
exp("P6", "Bootstrap CIs + decisive paired tests",
    "Decide whether the differences are real (CI excludes 0).",
    "Pool 4 T0 settings x years; bootstrap 95% CI; paired tests for growing-frozen and growing-(no-graph-LLM).",
    [["paired test", "k=10", "verdict"],
     ["growing - no-graph-LLM", "+0.105 [+0.041, +0.164]", "REAL (significant)"],
     ["growing - frozen (compounding)", "+0.036 [-0.050, +0.123]", "NOT significant"]],
    [5.5, 5, 4.2],
    "Graph > baselines is SIGNIFICANT. Compounding (growing>frozen) is directionally positive but NOT significant.",
    "f5_paired.png", 13.5)
story.append(PageBreak())

# ---------- P7 ----------
exp("P7", "Full-text vs abstract A/B",
    "Would reading full text thicken the thin task->task signal?",
    "On 25 papers with PubMedCentral full text, extract from abstract-only vs abstract+full-text with a transfer-targeted prompt; count task->task share.",
    [["condition", "task->task share", "note"],
     ["abstract (generic prompt)", "8%", "the original P2 graph"],
     ["abstract (transfer prompt)", "32%", "prompt is the lever"],
     ["abstract + full text", "19%", "full text dilutes (adds method->task)"]],
    [6, 3.5, 5.2],
    "The extraction PROMPT, not full text, is the lever for who-helps-whom density.",
    "f7_prompt.png", 12)

# ---------- P8 ----------
exp("P8", "Edge-quality audit (transfer vs usage)",
    "Are the edges genuine transfer, or merely 'method used for task'?",
    "An independent LLM judge classifies 40 sampled edges per type as transfer / usage / unclear.",
    [["edge type", "genuine transfer", "mere usage"],
     ["method->task", "20%", "78%"],
     ["task->task", "65%", "35%"]],
    [4.5, 5, 5.2],
    "The graph is largely a method-task USAGE table; genuine transfer is only ~23% of all edges.",
    "f6_quality.png", 12)
story.append(PageBreak())

# ---------- P9 ----------
exp("P9", "Transfer-prompt validation gate (30 papers)",
    "Before spending on a full re-extraction, prove a new prompt can purify usage out (precision-first gate).",
    "Diagnostic 30 = 10 method-heavy + 10 task->task-heavy + 10 recent (2022-24). Two-stage transfer-vs-usage prompt; independent judge; thresholds: precision>=70%, usage<=25%, evidence>=95%.",
    [["true-transfer precision", "usage leakage", "evidence valid", "empty rate"],
     ["100% (12/12)", "0%", "100%", "83%"]],
    [4.5, 3.2, 3.2, 3.3],
    "Prompt cleanly separates transfer from usage (PASS on precision), but genuine transfer is very sparse (83% of papers yield none).",
    None)

# ---------- P10 ----------
exp("P10", "Purify + re-evaluate (full 886 papers)",
    "Re-extract keeping only genuine transfer, then re-run the full evaluation on the clean graph (truth excludes inferred edges).",
    "Re-extract 886 papers with the validated transfer prompt; entity-resolve; NO transitive inference; re-run all 5 arms with bootstrap CIs.",
    [["", "edges", "growing", "frozen", "cooccur", "no-graph LLM", "random"],
     ["usage graph", "1,748", "0.127", "0.091", "0.027", "0.023", "0.000"],
     ["transfer graph", "253", "0.000", "0.000", "0.000", "0.027", "0.000"]],
    [2.6, 1.6, 2, 1.8, 1.8, 2.6, 1.8], fs=7.5,
    takeaway="Purifying to genuine transfer (1,748->253 edges) makes ALL graph arms collapse to 0; only the no-graph LLM stays nonzero. Diagnostic: navigation produces 0 candidates (1-2 'rich' nodes) and predictable-truth=0 (every emerged transfer edge introduces a NEW node). => The earlier win was carried by usage edges, not genuine transfer.",
    figname=None)
story.append(PageBreak())

# ---------- P11 ----------
exp("P11", "Region-level reformulation",
    "Edge-completion is structurally impossible on a sparse, new-node-driven transfer graph. Reframe to predicting the semantic REGION of next-year transfer (new-node-tolerant).",
    "Arms: LLM generates target areas (llm_raw); graph re-ranks the LLM list (llm_rerank); most-frequent past targets (popularity); random. Region-tolerant precision@k.",
    [["P@10", "llm_raw", "graph-rerank", "popularity", "random"],
     ["mean", "0.389", "0.222", "0.278", "0.144"],
     ["vs", "rawLLM-random = +0.244 REAL", "", "rerank-rawLLM = -0.167 (graph HURTS)", ""]],
    [1.6, 3.2, 3.4, 3.2, 2.8], fs=7.5,
    takeaway="The reformulation is tractable (0.3-0.5 vs 0.0 for edge-completion) and the LLM beats random - but the graph does NOT help (re-rank and popularity are below the raw LLM). Confounds: the LLM is leakage-inflated (recent cutoff), and the re-rank crudely replaced the LLM's ordering. A clean 'does the graph add value' test (LLM held constant, graph features ADDED not substituted) remains to be run.")

# ---------- summary ----------
P("Bottom line: what is established vs not", H)
P("<b>Established (real results):</b>", LBL)
for t in ["A complete, automated pipeline: fetch -> extract -> graph -> navigate -> temporal-replay evaluation.",
          "On the research-activity / usage graph, structure-driven navigation significantly beats no-graph-LLM, co-occurrence, and random (P6, with CIs).",
          "A history-as-label temporal-replay evaluation methodology (which the static-graph competitors cannot run).",
          "An honest edge-quality audit exposing the usage-vs-transfer distinction."]:
    P("&bull;&nbsp; " + t, st("x", leftIndent=12, spaceAfter=2))
P("<b>Not established:</b>", LBL)
for t in ["'A genuine-transfer graph compounds (growing>frozen)' - after purification the transfer graph is too sparse and the graph collapses.",
          "'The graph beats an LLM at discovering transfer' - not shown; the LLM is currently stronger."]:
    P("&bull;&nbsp; " + t, st("y", leftIndent=12, spaceAfter=2))
sp(4)
P("<b>Net.</b> We built the system and obtained a real, significant result - but at the research-activity level. The moment we insist on genuine transfer, the signal is too sparse and the graph stops helping. The journey is an honest de-risking: a plausible compounding story, verified layer by layer down to where it actually holds.", BODY)

doc = SimpleDocTemplate(str(OUT), pagesize=A4, topMargin=1.6*cm, bottomMargin=1.6*cm,
                        leftMargin=1.8*cm, rightMargin=1.8*cm, title="Complete Experiment Log")
doc.build(story)
print("PDF:", OUT, "|", os.path.getsize(OUT), "bytes")
