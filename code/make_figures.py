#!/usr/bin/env python3
"""Generate all figures for the pilot findings report (real data + run results)."""
import json, pathlib, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from graph import Graph

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "data" / "figs"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spider": False} if False else {"figure.dpi": 150, "font.size": 11})
C = {"growing": "#1f6feb", "frozen": "#8b949e", "cooccur": "#a371f7", "nograph_llm": "#f0883e", "random": "#6e7681"}

# ---------- data from files ----------
papers = [json.loads(l) for l in (ROOT / "data" / "brain_papers.jsonl").read_text().splitlines()]
G = Graph.load(ROOT / "data" / "graph_resolved.json")

# ---------- Fig 1: corpus year distribution ----------
yrs = collections.Counter(p["year"] for p in papers)
xs = sorted(yrs)
plt.figure(figsize=(7, 3.2))
plt.bar(xs, [yrs[y] for y in xs], color="#1f6feb")
plt.axvline(2020.5, color="crimson", ls="--", lw=1.5)
plt.text(2020.6, max(yrs.values())*0.9, "T0=2020\n(build / predict split)", color="crimson", fontsize=9)
plt.title("Corpus: 8,000 brain medical-imaging papers (2015–2024)")
plt.xlabel("year"); plt.ylabel("# papers"); plt.tight_layout()
plt.savefig(FIG / "f1_years.png"); plt.close()

# ---------- Fig 2: graph composition ----------
ntype = collections.Counter(n["type"] for n in G.nodes.values())
etype_ext = collections.Counter(); etype_inf = collections.Counter()
for (s, d), e in G.edges.items():
    t = f'{G.nodes[s]["type"]}→{G.nodes[d]["type"]}'
    (etype_inf if e["state"] == "inferred" else etype_ext)[t] += 1
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].bar(["task", "method"], [ntype["task"], ntype["method"]], color=["#1f6feb", "#a371f7"])
ax[0].set_title(f"Nodes (n={len(G.nodes)})"); ax[0].set_ylabel("count")
types = ["method→task", "task→task", "method→method"]
ext = [etype_ext.get(t, 0) for t in types]; inf = [etype_inf.get(t, 0) for t in types]
ax[1].bar(types, ext, color="#1f6feb", label="extracted")
ax[1].bar(types, inf, bottom=ext, color="#9ecbff", label="transitive-inferred")
ax[1].set_title(f"Edges (n={len(G.edges)})"); ax[1].legend(); ax[1].tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig(FIG / "f2_graph.png"); plt.close()

# ---------- Fig 3: precision@k by arm, with 95% CI (P6 results) ----------
P6 = {  # arm -> k -> (mean, lo, hi)
 "growing": {10: (.127, .068, .191), 20: (.093, .059, .132), 50: (.079, .055, .103)},
 "frozen": {10: (.091, .036, .168), 20: (.075, .036, .123), 50: (.056, .030, .089)},
 "cooccur": {10: (.027, .000, .055), 20: (.057, .025, .091), 50: (.047, .027, .069)},
 "nograph_llm": {10: (.023, .005, .041), 20: (.016, .007, .025), 50: (.013, .006, .020)},
 "random": {10: (.000, 0, 0), 20: (.000, 0, 0), 50: (.001, 0, .003)},
}
ks = [10, 20, 50]; arms = list(P6); w = 0.16
plt.figure(figsize=(8, 4))
for i, a in enumerate(arms):
    means = [P6[a][k][0] for k in ks]
    lo = [P6[a][k][0]-P6[a][k][1] for k in ks]; hi = [P6[a][k][2]-P6[a][k][0] for k in ks]
    x = np.arange(len(ks)) + (i-2)*w
    plt.bar(x, means, w, yerr=[lo, hi], capsize=3, color=C[a], label=a)
plt.xticks(np.arange(len(ks)), [f"P@{k}" for k in ks])
plt.ylabel("semantic precision@k"); plt.legend(fontsize=8, ncol=2)
plt.title("Navigation accuracy by arm (bootstrap 95% CI, 4 T0 × years)")
plt.tight_layout(); plt.savefig(FIG / "f3_precision.png"); plt.close()

# ---------- Fig 4: T0 sweep growing vs frozen (compounding signal) ----------
T0 = [2017, 2018, 2019, 2020]
gw = [.080, .090, .100, .065]; fz = [.000, .107, .072, .085]
plt.figure(figsize=(7, 3.6))
plt.plot(T0, gw, "o-", color=C["growing"], lw=2, label="growing (graph grows yearly)")
plt.plot(T0, fz, "s--", color=C["frozen"], lw=2, label="frozen (graph fixed at T0)")
plt.xticks(T0); plt.xlabel("start year T0 (smaller T0 = smaller base, more growth)")
plt.ylabel("mean semantic precision@50")
plt.title("Compounding probe: growing vs frozen across T0")
plt.legend(); plt.tight_layout(); plt.savefig(FIG / "f4_t0sweep.png"); plt.close()

# ---------- Fig 5: paired difference forest plot ----------
pairs = [("growing − frozen\n(compounding)", [(.036, -.050, .123), (.018, -.027, .066), (.023, -.008, .052)]),
         ("growing − no-graph-LLM\n(value of graph)", [(.105, .041, .164), (.077, .041, .116), (.066, .042, .088)])]
plt.figure(figsize=(7.5, 3.6)); ypos = 0; yticks = []; ylabels = []
for name, vals in pairs:
    for j, (m, lo, hi) in enumerate(vals):
        col = "#2ea043" if lo > 0 else "#d29922"
        plt.errorbar(m, ypos, xerr=[[m-lo], [hi-m]], fmt="o", color=col, capsize=4)
        yticks.append(ypos); ylabels.append(f"{name.splitlines()[0] if j==1 else ''}  k={[10,20,50][j]}")
        ypos += 1
    ypos += 0.6
plt.axvline(0, color="k", lw=1)
plt.yticks(yticks, ylabels, fontsize=8)
plt.xlabel("mean paired difference (95% CI); green = CI excludes 0 (significant)")
plt.title("Decisive tests"); plt.tight_layout(); plt.savefig(FIG / "f5_paired.png"); plt.close()

# ---------- Fig 6: edge quality (transfer vs usage) ----------
labels = ["method→task", "task→task"]
transfer = [20, 65]; usage = [78, 35]; unclear = [2, 0]
x = np.arange(2)
plt.figure(figsize=(6.5, 3.6))
plt.bar(x, transfer, color="#2ea043", label="genuine transfer")
plt.bar(x, usage, bottom=transfer, color="#d29922", label="mere usage")
plt.bar(x, unclear, bottom=[t+u for t, u in zip(transfer, usage)], color="#6e7681", label="unclear")
plt.xticks(x, labels); plt.ylabel("% of sampled edges")
plt.title("Edge-quality audit: transfer vs usage (40 edges/type)")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / "f6_quality.png"); plt.close()

# ---------- Fig 7: prompt lever (task->task density) ----------
conds = ["abstract\n(generic prompt)", "abstract\n(transfer prompt)", "abstract+fulltext\n(transfer prompt)"]
pct = [8, 32, 19]
plt.figure(figsize=(6.5, 3.4))
b = plt.bar(conds, pct, color=["#8b949e", "#2ea043", "#1f6feb"])
plt.bar_label(b, fmt="%d%%")
plt.ylabel("% task→task edges"); plt.title("Prompt is the lever for who-helps-whom density")
plt.tight_layout(); plt.savefig(FIG / "f7_prompt.png"); plt.close()

print("figures written to", FIG)
for f in sorted(FIG.glob("*.png")):
    print("  ", f.name)
