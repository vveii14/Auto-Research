#!/usr/bin/env python3
"""Assemble the complete pilot findings PDF in academic English."""
import pathlib, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "data" / "figs"
OUT = ROOT / "Pilot_Findings_and_Outlook.pdf"

SERIF, SANS = "Times-Roman", "Helvetica-Bold"
def style(name, **kw):
    base = dict(fontName=SERIF, fontSize=10.5, leading=15, spaceAfter=6)
    base.update(kw); return ParagraphStyle(name, **base)
H1 = style("H1", fontName=SANS, fontSize=15, leading=19, textColor=colors.HexColor("#0b3d91"), spaceBefore=10, spaceAfter=7)
H2 = style("H2", fontName=SANS, fontSize=11.5, leading=15, textColor=colors.HexColor("#1f6feb"), spaceBefore=8, spaceAfter=4)
BODY = style("BODY", alignment=4)
BULLET = style("BULLET", leftIndent=14, spaceAfter=3)
SMALL = style("SMALL", fontSize=8.5, leading=11, textColor=colors.HexColor("#57606a"))
TITLE = style("TITLE", fontName=SANS, fontSize=21, leading=26, textColor=colors.HexColor("#0b3d91"))
SUB = style("SUB", fontSize=12.5, leading=16, textColor=colors.HexColor("#57606a"))

story = []
def P(t, s=BODY): story.append(Paragraph(t, s))
def B(t): story.append(Paragraph("&bull;&nbsp; " + t, BULLET))
def sp(h=6): story.append(Spacer(1, h))
def fig(name, w_cm=15.5):
    p = FIG / name
    iw, ih = ImageReader(str(p)).getSize()
    w = w_cm * cm; h = w * ih / iw
    story.append(Image(str(p), width=w, height=h)); sp(4)
def cap(t): story.append(Paragraph(t, SMALL)); sp(8)
def tbl(data, widths):
    t = Table(data, colWidths=[w*cm for w in widths])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), SERIF), ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (-1,0), SANS),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#c9d1d9")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f6f8fa")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (1,0), (-1,-1), "CENTER")]))
    story.append(t); sp(8)

# ---------------- Title ----------------
P("A Living Task-Transfer Graph<br/>for Compounding Auto-Research", TITLE); sp(4)
P("Pilot Findings and Full-Scale Outlook", SUB); sp(2)
P("Brain medical-imaging domain &middot; 886-paper subsample &middot; 2026-06-01 &middot; github.com/vveii14/Auto-Research", SMALL)
sp(10); fig("../../figure1_overview.png", 16)
cap("Figure 0. System overview: corpus -&gt; LLM extraction (task/method nodes + directed transfer edges) -&gt; navigation (rich-node projection + structural holes) -&gt; explore-and-fold-back (graph grows) -&gt; governance. Bottom: the history-as-label temporal-replay evaluation.")
story.append(PageBreak())

# ---------------- 1 ----------------
P("1. Executive Summary", H1)
P("We cast automatic research-direction discovery as <b>candidate transfer-edge prediction on a temporal graph</b>. From timestamped papers we extract task and method nodes together with <b>directed, signed &ldquo;does A help B&rdquo; transfer edges</b>, forming a graph that grows as exploration proceeds and, in turn, navigates it; a <b>history-as-label</b> temporal replay tests whether it becomes more accurate over time. This report presents an end-to-end pilot on 886 brain medical-imaging abstracts.")
sp(2); P("Key findings (stated honestly):", H2)
B("<b>Significant.</b> Graph-structured navigation predicts future research directions <b>significantly better than a no-graph LLM, a co-occurrence graph, and random</b> (paired bootstrap CI excludes 0). This also shows the gain comes from graph structure, not from the LLM memorizing future papers.")
B("<b>Not yet significant.</b> The headline <b>compounding</b> effect (growing &gt; frozen) is consistently positive in the point estimate, but its CI still spans 0 at the 900-paper scale.")
B("<b>Real weakness.</b> 78% of method-&gt;task edges are mere <i>usage</i> rather than genuine <i>transfer</i>; only ~23% of all edges are genuine transfer. The main cause is the extraction prompt (fixable).")
B("<b>Verdict.</b> The infrastructure and the &ldquo;graph &gt; baselines&rdquo; result are established (a safety-net contribution); the compounding claim warrants one scale-up with a clear stop-loss.")
story.append(PageBreak())

# ---------------- 2 ----------------
P("2. Core Idea and Workflow", H1)
P("<b>Positioning.</b> The first system to turn a research-relation graph from a <i>static, read-only substrate</i> into a structure that <b>grows with exploration and navigates it</b>. Closest prior work: Intern-Atlas (static, read-only), Deep Ideation (fixed substrate), AC/DC (no relational graph). The novelty is the <b>combination of directed transfer semantics + living growth + graph navigation + compounding</b>, applied equally to method-&gt;task and task-&gt;task edges.")
P("Workflow (8 stages):", H2)
for t in ["Define corpus scope (a tunable parameter: venues / years / sample size).",
          "Ingest timestamped papers indiscriminately.",
          "LLM extraction: task/method nodes + directed transfer edges, each with a verbatim evidence span.",
          "Transitive inference: A-&gt;B and B-&gt;C imply A-&gt;C, filling sparse regions.",
          "Navigation (non-random, the core): rich-node projection + structural holes + topological scoring; the LLM only performs caged local filtering, yielding top-k directions.",
          "Explore and fold back: a direction is sent to auto-research; success and failure alike return as verified edges, so the graph grows.",
          "Governance: deduplication / conflict resolution / decay, to prevent the graph from degrading.",
          "Continuous ingestion of new papers, so the graph grows with the field."]:
    B(t)
P("<b>Evaluation.</b> Build the graph up to year T, predict the transfer edges that actually emerge in T+1, score first and only then ingest T+1, and roll forward year by year; compare growing / frozen / co-occurrence / no-graph-LLM / random. Real publication history provides <b>free, objective labels</b>, which avoids circular reasoning.")
story.append(PageBreak())

# ---------------- 3 ----------------
P("3. Pilot Setup and Corpus", H1)
P("Source: Semantic Scholar (abstracts only); filtered to fieldsOfStudy = Computer Science plus brain-imaging / method terms; years 2015-2024. 8,000 papers were fetched; the pilot uses an 886-paper subsample (500 from &le;2020 to build the graph, and 100 per year for 2021-2024 as prediction targets / ingestion material). Extraction uses an Azure-hosted Opus model; node embeddings use all-MiniLM-L6-v2.")
fig("f1_years.png", 14)
cap("Figure 1. Corpus year distribution. The steady 2015-2024 growth suits temporal hold-out; the red line marks the T0=2020 build/predict split.")
story.append(PageBreak())

# ---------------- 4 ----------------
P("4. The Constructed Transfer Graph", H1)
P("886 papers yield, after entity resolution, <b>2,907 nodes (1,064 task / 1,843 method) and 1,748 extracted edges</b>; transitive inference adds about 518 more. Edges are dominated by method-&gt;task (1,581); task-&gt;task accounts for 131 and method-&gt;method for 36.")
fig("f2_graph.png", 15)
cap("Figure 2. Graph composition: node types (left) and edge types (right, stacked to show extracted vs. transitively inferred).")
P("Representative genuine task-&gt;task transfer edges (who-helps-whom):", H2)
tbl([["Source task -&gt; Target task", "Evidence (excerpt)"],
     ["age prediction (structural MRI) -&gt; Alzheimer's diagnosis", "sensitive to deviance from normal aging..."],
     ["hippocampus segmentation -&gt; Alzheimer's classification", "multi-task CNN jointly learning hippocampus..."],
     ["virtual connectome inference -&gt; AD classification", "trained on virtual connectomes can be used..."],
     ["PET segmentation <-> denoising <-> partial-volume corr.", "segmentation can help in denoising and PVC..."]],
    [7.6, 7.9])
story.append(PageBreak())

# ---------------- 5 ----------------
P("5. Pilot Results", H1)
P("5.1 Navigation accuracy (semantic matching, bootstrap 95% CI)", H2)
P("On the same temporal-forecast task, our arms (growing / frozen) consistently outperform the co-occurrence graph, the no-graph LLM, and random across all k. The no-graph LLM is near-random, indicating that the LLM alone (with its memorized future) is weak and that the gain comes from the graph.")
fig("f3_precision.png", 15)
cap("Figure 3. Semantic precision@k by arm with 95% confidence intervals (pooled over 4 T0 settings x years).")
tbl([["Arm", "P@10", "P@20", "P@50"],
     ["growing", "0.127", "0.093", "0.079"],
     ["frozen", "0.091", "0.075", "0.056"],
     ["co-occurrence", "0.027", "0.057", "0.047"],
     ["no-graph LLM", "0.023", "0.016", "0.013"],
     ["random", "0.000", "0.000", "0.001"]], [5, 3.5, 3.5, 3.5])

P("5.2 Compounding probe: growing vs. frozen", H2)
P("Sweeping the start year T0: when the base is small and growth is proportionally large (T0=2017, 2019), growing clearly exceeds frozen; when the base is large and yearly increments are tiny (T0=2020), they converge. This is consistent with &ldquo;growth must be large relative to the base for compounding to surface,&rdquo; and reflects a limitation of the small subsample at late T0.")
fig("f4_t0sweep.png", 14)
cap("Figure 4. growing (graph grows each year) vs. frozen (graph fixed at T0) across start years T0.")

P("5.3 Decisive paired tests", H2)
P("growing - (no-graph LLM) has a CI above 0 at every k (<b>significant</b>); growing - frozen has a consistently positive mean but a CI that still spans 0 (<b>not yet significant</b>).")
fig("f5_paired.png", 14.5)
cap("Figure 5. Paired mean differences with 95% CIs; green = CI excludes 0 (significant).")

P("5.4 Edge-quality audit: transfer vs. usage", H2)
P("Classifying edges as genuine <i>transfer</i> (pretraining / representation reuse / multi-task / measured gain) vs. mere <i>usage</i> (a method simply applied to a task): only 20% of method-&gt;task edges are genuine transfer (78% usage), whereas 65% of task-&gt;task edges are transfer. Genuine-transfer edges are estimated at only ~23% of the graph.")
fig("f6_quality.png", 13)
cap("Figure 6. Edge-quality audit (40 sampled edges per type). The current graph is largely a method-task usage table; genuine transfer is the minority.")

P("5.5 The extraction prompt is the key lever", H2)
P("A full-text vs. abstract A/B test shows that the main lever for thickening task-&gt;task is the <b>extraction prompt, not reading full text</b>. On the same abstracts, a transfer-targeted prompt raises the task-&gt;task share from 8% to 32%; adding full text dilutes the proportion (it contributes still more method-&gt;task edges).")
fig("f7_prompt.png", 13)
cap("Figure 7. task-&gt;task edge share: generic prompt vs. transfer-targeted prompt vs. + full text.")
story.append(PageBreak())

# ---------------- 6 ----------------
P("6. Established / Open", H1)
P("Established (significant):", H2)
B("Graph-structured navigation &gt; no-graph LLM, &gt; co-occurrence graph, &gt;&gt; random (paired CIs exclude 0).")
B("The gain is attributable to the graph, not to LLM leakage (the no-graph LLM is near-random).")
P("Open:", H2)
B("Compounding (growing &gt; frozen) is directionally positive but not significant -- needs scale-up (more years / data points to tighten the CI).")
B("Edge quality: genuine transfer is only ~23% -- needs a transfer-vs-usage re-extraction.")
B("Scale and noise: 900-paper subsample, abstracts only, a single embedding model; values are small and noisy.")

# ---------------- 7 ----------------
P("7. Full-Scale Outlook and Roadmap", H1)
P("Three highest-leverage steps (the rest is engineering scale-up):", H2)
B("<b>Purifying re-extraction.</b> A prompt that distinguishes genuine transfer from usage (treating both source types equally); cheaply re-verify graph quality and whether &ldquo;graph &gt; baselines&rdquo; survives purification.")
B("<b>Scale to resolve compounding.</b> ~3,000 papers plus more start years T0 to push the growing - frozen CI off zero; add full text to further thicken genuine transfer.")
B("<b>Full system.</b> Local 72B extraction (reproducible, leakage-controlled) + Neo4j / vector index + governance always on + multiple domains (brain -&gt; retina -&gt; cardiac) + a live auto-research closed-loop demo.")
P("<b>Stop-loss.</b> If compounding remains non-significant after scale-up, fall back to the (already significant) &ldquo;structure-driven navigation &gt; no-graph LLM / co-occurrence / random&rdquo; paper.")
sp(6)
P("Implementation progress", H2)
tbl([["Stage", "Content", "Status"],
     ["P0", "Data acquisition + graph scaffold", "done (8,000 papers)"],
     ["P1", "Extraction validation", "done"],
     ["P2", "Graph build + resolution + inference", "done (886 papers)"],
     ["P3", "Graph navigation", "done"],
     ["P4-P6", "Temporal evaluation + confidence intervals", "done"],
     ["P7-P8", "Full-text A/B + edge-quality audit", "done"],
     ["--", "Purifying re-extraction / scale-up / full system", "to do"]], [2.4, 9.1, 4])

doc = SimpleDocTemplate(str(OUT), pagesize=A4, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        leftMargin=2*cm, rightMargin=2*cm, title="Pilot Findings and Outlook")
doc.build(story)
print("PDF written:", OUT, "|", os.path.getsize(OUT), "bytes")
