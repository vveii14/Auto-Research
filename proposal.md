# A Living Task-Transfer Map for Compounding Auto-Research
### Project Proposal (for collaborators) · 项目提案(合作者对齐版)

> **One-line thesis / 一句话主张.** Build a *living* "who-helps-whom" task-transfer graph that **grows as exploration proceeds** and, in turn, **navigates exploration toward the next most valuable research direction** — turning auto-research from a memoryless, re-start-every-round process into a **compounding** one.
>
> 〔中文〕维护一张**会动的"任务互助迁移图"**:它随探索生长、并反过来导航下一个最值得做的方向,把"每轮从零重推"的自动研究变成**复利**过程。

---

## 1. TL;DR

- **Problem.** Current auto-research / idea-generation systems do not *accumulate*. Each round re-ingests papers and re-prompts an LLM from scratch; discovered relationships ("approach A turns out to help task B") are never retained. Exploration is **flat, not compounding**.
- **Idea.** A directed, signed **task-transfer graph** (nodes = tasks/methods; edges = "A helps/hurts B") that is **continuously grown** by folding exploration results back in, and that **drives navigation** of where to explore next.
- **Why it can win.** We are the first to make the research graph **dynamic + self-navigating**. Closest systems are either static read-only (Intern-Atlas), fixed-substrate (Deep Ideation), or graph-less diversity search (AC/DC).
- **Why it is credible (not circular).** We validate against **external ground truth from real history**: (i) edges vs. *measured* transfer numbers reported in papers; (ii) navigation vs. *what the field actually did next* (temporal hold-out). The headline evidence is a **compounding curve** that a **growing** graph produces but a **frozen** graph cannot.
- **Training-free.** The LLM is never fine-tuned; intelligence accumulates in the **graph (external memory)**, not in weights. Corpus scope is a **tunable parameter** (a few thousand → millions of papers, same pipeline).

〔中文〕病:不积累。招:活体迁移图 + 复利闭环。赢点:图会动且自导航。可信:拿真实历史当外部真值,主证据是"生长 vs 冻结"的复利曲线。不训练,scale 是旋钮。

---

![System overview](./figure1_overview.png)

**Figure 1. System overview.** Time-stamped corpus → LLM extraction of **task** and **method** nodes with signed transfer edges → transitive inference → **graph-guided navigation** (rich-node projection + structural-hole discovery; *topology-driven scoring, with the LLM restricted to lightweight feasibility filtering*) → auto-research → results fold back as **verified positive/negative edges** → governance → continuous ingest. **Bottom:** *Historical Replay* evaluation — build graph ≤ T, **predict T+1, compare to real T+1 papers, and only then ingest T+1** (strict temporal hold-out), repeating year by year; a **Growing** graph is compared against a **Frozen** one and against no-graph / co-occurrence / random baselines.
*The hit-rate curve is **illustrative of the expected trend**, not measured results. "No leakage" refers to corpus-level hold-out; residual LLM-pretraining leakage is **controlled and quantified by the no-graph-LLM baseline** (§7.4), not eliminated.*
〔中文〕图中折线为**预期趋势示意**,非真实数据;"无泄漏"指语料层留出,LLM 权重层的残余泄漏由 no-graph 基线**度量并扣除**,而非消除。

---

## 2. Motivation & Problem / 动机与问题

Auto-research pipelines (idea generation, autonomous experimentation) treat each query independently. They can broadcast many ideas but **do not get smarter across rounds**: the structure they uncover (which directions transfer to which) evaporates. This is the gap we target — **persistent, structured, self-improving research memory that also decides where to look next.**

The key unmet need is not *scale of reading* (Intern-Atlas already reads ~1M papers) nor *retrieval speed* (HNSW/GraphRAG are mature). It is a **living relational structure that compounds**: each exploration both *uses* and *improves* the map.

〔中文〕缺口不是"读得多"(Intern-Atlas 已百万级),也不是"检索快"(成熟),而是**会复利的活体关系结构**——每次探索既用图、又改进图。

---

## 3. Core Idea / 核心想法

A map where:
- **Each node = a research task / direction / method** (e.g., "contrastive pretraining for brain-MRI segmentation").
- **Each edge = a directed, signed transfer relation A→B**: does mastering A *help* (positive transfer), *hurt* (negative transfer), or not affect B?

Unlike a citation graph (Intern-Atlas: method-evolution edges) or a co-occurrence graph (Deep Ideation), our edge encodes **where knowledge can flow** — the question a researcher actually asks: *"Given what I can do, which new direction is most likely to pay off?"*

**The flywheel (闭环):**
```
read graph → navigate to a "rich" node / structural hole
→ incubate the next direction (auto-research)
→ fold result back as new node/edges (graph grows & self-corrects)
→ govern (keep it clean) → read graph …
```
Bigger, truer graph → smarter navigation → more valuable discoveries → bigger graph. **This compounding IS the thesis.**

---

## 4. Positioning vs. Related Work / 定位

| System | Has graph? | Graph grows? | Graph navigates exploration? | Edge semantics |
|---|---|---|---|---|
| **Intern-Atlas** (2604.28158) | ✅ ~1M papers / 9.4M edges | ❌ static, read-only, no result feedback | ❌ | method evolution (extends/improves) |
| **Deep Ideation** (2511.02238) | ✅ concept co-occurrence | ❌ underlying graph fixed | partial | keyword co-occurrence |
| **AC/DC** (2604.14969) | ❌ no relational graph | — | ❌ diversity/novelty broadcast | n/a (discovers *LLM experts* via model merging) |
| **Ours** | ✅ | ✅ **living, results fold back** | ✅ **structure-directed** | **task transfer (directed, signed)** |

> **Note on AC/DC** 〔重要〕: AC/DC is an *LLM-expert-discovery* framework (model merging + synthetic-task coevolution), **not** an idea-generation competitor. We cite it as an **open-endedness paradigm reference** (diversity-driven discovery), and use a diversity-navigation variant as a **baseline**, not as the same-task rival.

**Clean delta vs. Deep Ideation** 〔与最近对手的锋利区别〕: Deep Ideation's *Idea Stack* accumulates a *linear history*; our contribution is folding results into a **structured transfer graph** where **transitivity** lets one new edge update estimates for *many* unexplored directions — compounding the Idea Stack cannot give.

---

## 5. Framework / 框架

### 5.1 Graph data model / 数据模型

**Node** `{ id, text, type(task|method), domain, method_summary, embedding, first_year, paper_refs, aliases, freq }`

**Edge A→B** `{ sign(±1/0), strength[0,1], transfer_value, state, confidence, contradicts_prior, evidence(paper, span), first_year, access_count, last_used }`

- `state ∈ {prior, proxy, inferred, verified}` — credibility tier: LLM-guessed → geometry-backed → transitivity-inferred → measured/really-run.
- **`first_year` on both nodes and edges is the linchpin** of temporal validation. 〔时序验证全靠它〕
- `evidence` makes every edge traceable (anti-hallucination, à la Intern-Atlas).

### 5.2 Pipeline (8 stages) / 八步流程

| # | Stage | In → Out |
|---|---|---|
| 0 | **Scope (only human input)** | venue/year/sample-size parameter → timestamped paper stream 〔唯一人工设定,可控旋钮〕 |
| 1 | **Bulk ingest** | papers ≤T, full or random sample (no cherry-picking; tasks emerge bottom-up) |
| 2 | **Extraction** | LLM reads each paper → task nodes + signed transfer edges + evidence; entity-resolve nodes → initial graph G₀ |
| 3 | **Transitive inference** | A→B, B→C ⇒ A→C (`inferred`, low conf) — extract the skeleton, infer the rest |
| 4 | **Navigation (NON-random, core)** | graph → ranked "next directions" |
| 5 | **Explore + fold-back (growth)** | direction → auto-research → result (success/contradiction) folded back as new edges |
| 6 | **Governance** | dedup / conflict-resolution / decay / deprecate — keep graph clean |
| 7 | **Continuous ingest (living)** | new papers added incrementally, no full re-read 〔对手做不到〕 |
| 8 | **Evaluation harness** | inject external truth, score the system (see §7) |

### 5.3 Navigation — the heart / 导航(灵魂)

A "direction" = a **plausible edge that does not yet exist** but is implied by structure:
- **(a) Rich-node projection.** A node with many positive out-edges (a "rich vein") is projected onto similar, not-yet-connected targets.
- **(b) Structural holes.** High-similarity / high-shared-neighbor task pairs that lack an edge ("should-be-connected-but-isn't").
- **Scoring is graph-topological** (out-degree, similarity, path strength, transitive support). The **LLM is caged**: it only does light, *local*, future-knowledge-free filtering ("is this learnable/interesting?"), never free-form generation of the answer.

> **Why caged?** 〔关键:防泄漏〕 A modern LLM has *already read* future papers during pretraining. If navigation relied on the LLM's free generation, "predicting" a 2023 direction from a 2020 graph could be **memory, not navigation**. Keeping the decision graph-driven makes any hit **attributable to the graph**. See §7.4.

### 5.4 Growth & governance / 生长与治理

- **Fold-back:** success → `verified` positive edge; failure/contradiction → negative edge + `contradicts_prior` flag (a "dead-end" is equally valuable — it stops the navigator from going there and calibrates priors).
- **Governance** (mature engineering; cf. Mem0/MemOS/FadeMem/SSGM): merge aliases, resolve conflicting edges via self-edit (not append), decay & deprecate low-confidence, long-unused edges.
- 〔注意张力〕Decay must not delete an under-explored *opportunity*; in a discovery graph "long unused" ≠ "wrong". We will tune decay to be conservative on high-`confidence`/`verified` edges.

---

## 6. Key Innovations / 关键创新

1. **Living graph** — results fold back, graph grows; no competitor does this.
2. **Structure-directed navigation** — exploration steered by graph topology, not diversity broadcast.
3. **History-as-label evaluation** — converts open-ended ideation into a **falsifiable forecasting task** with free, objective labels; a method only a *growing temporal* graph can even set up.

---

## 7. Validation Plan / 验证方案 (the make-or-break)

**Principle:** never let the system grade its own homework. Truth comes from **real history**, external and pre-existing. 〔历史当标签,不循环〕

### 7.1 Three verifiers — do not conflate / 三个裁判别混

| | Role | Ground truth |
|---|---|---|
| **Stage 5 (fold-back)** | NOT a verifier — the in-loop **growth/teacher** mechanism (the thing being judged) | auto-research result (real run) / in validation: that year's real papers |
| **V-A: edge quality** | JUDGE of whether edges are correct | *measured* transfer deltas reported in papers |
| **V-B: navigation/compounding** | JUDGE of the main thesis | *what the field actually did next* |

### 7.2 V-A — Are the edges true? / 图准不准
Harvest a **measured transfer matrix M\*** from papers reporting "pretrain on A → finetune on B → +x%". Compare the system's `prior`/`inferred` edges against M\* (sign accuracy / precision / recall). **Anti-leakage:** hold out the measurement papers so the edge is *predicted*, not *looked up*.

### 7.3 V-B — Does navigation predict value? + Compounding / 复利曲线
**The year-by-year temporal loop** (history replaces expensive auto-research):

```
G ← build_initial_graph(corpus ≤ T0)
for year Y in T0+1 … T_end:
    pred  ← navigate(G, k)                  # predict using graph that has NOT seen Y
    truth ← real edges/directions in year Y
    hit[Y] ← hit_rate(pred, truth)          # SCORE FIRST  (V-B judge)
    G.ingest(papers of year Y); govern(G)   # THEN GROW    (stage-5 teacher)
```
- **Compounding curve = hit[Y] over years.** Rising ⇒ the graph gets smarter as it grows.
- **Make-or-break ablation — Growing vs Frozen:** run a parallel **Frozen** arm that never folds back (always predicts from G_{T0}). **Growing pulling away from Frozen = compounding, measured.**
- **Strict order (anti-leakage):** SCORE year Y *before* ingesting Y. Reversing it invalidates the experiment.
- **"Value" via real signals:** weight hits by citations / follow-up papers / top-venue acceptance — not LLM opinion.

### 7.4 Baselines & leakage control / 基线与泄漏控制
On the same temporal-forecast task:
- **`no-graph LLM`** — ask the LLM directly for next directions. **This doubles as the leakage yardstick:** it quantifies how much the LLM's *memorized future* alone predicts. Our system must **significantly beat it**, so the delta is attributable to the graph (LLM memory held constant across arms). 〔把 LLM 记忆当常数扣除〕
- **`static co-occurrence graph`** (Deep Ideation analog), **`diversity navigation`** (AC/DC-spirit), **`random`** (lower bound).

### 7.5 Metrics / 指标
- V-A: sign-accuracy, precision/recall of edges vs M\*.
- V-B: hit@k (value-weighted) vs year; **slope** of compounding curve; Growing − Frozen gap; margin over `no-graph LLM`.
- Efficiency: high-value directions discovered per unit LLM/compute.

### 7.6 Optional live demo / 真实系统演示
On the top predicted directions, **actually run auto-research once** as qualitative evidence the loop is deployable. (The compounding curve does **not** depend on this.)

---

## 8. Risks & Mitigations / 风险与应对

| Risk | Mitigation |
|---|---|
| **LLM pretraining leakage** (model "knows" the future) | structure-driven navigation (§5.3) + `no-graph LLM` baseline as leakage yardstick (§7.4); optionally time-frozen models |
| **Circularity** (LLM judges itself) | external ground truth only (measured transfer + real history); growing-vs-frozen |
| **Hit-rate rewards only what happened** (mainstream bias; good-but-unpursued = "miss") | recall-oriented framing + citation/value weighting; state as a lower bound |
| **Navigation collapse** (endless drilling into rich veins) | novelty/interestingness quota forces a fraction of divergent exploration |
| **Distance to Deep Ideation** | hammer the three deltas: living + structure-directed + temporal eval; lead the intro with transitive-reuse vs linear Idea-Stack |
| **Graph dirtiness at scale** | governance suite (dedup/conflict/decay/deprecate), conservative on verified edges |

---

## 9. Work Plan, Milestones & Division of Labor / 分工与里程碑

**Suggested workstreams (assignable to collaborators):**

- **WS1 — Corpus & extraction (Stages 0–2).** Crawl scoped venues with timestamps; LLM extraction of task nodes + signed edges + evidence; entity resolution. *Deliverable: G₀ + extraction quality report.*
- **WS2 — Graph engine (Stages 3, 6, 7).** Transitive inference, governance (dedup/conflict/decay), incremental ingest; graph DB + embedding index. *Deliverable: living-graph service.*
- **WS3 — Navigation (Stage 4).** Rich-node projection, structural holes, topological scoring, caged-LLM filter. *Deliverable: `navigate()` + ablation hooks.*
- **WS4 — Ground truth & evaluation (Stage 8).** Mine measured-transfer matrix M\* (V-A); temporal harness, growing/frozen, baselines, leakage control, value weighting (V-B). *Deliverable: the compounding-curve experiment.*
- **WS5 — Live auto-research demo (optional).** Plug one auto-research backend into fold-back for the qualitative demo.

**Milestones / 里程碑:**
1. **M1 — Pilot graph + temporal harness on a narrow sub-domain** (e.g., brain/retina): prove the *compounding curve trends positive*. 〔最高优先级:小规模先坐实复利趋势,这是命脉〕
2. **M2 — Growing-vs-frozen + baselines** complete; leakage control verified (beat `no-graph LLM`).
3. **M3 — Edge-quality validation (V-A)** vs measured transfer matrix.
4. **M4 — Scale up corpus** (turn the scope knob), governance hardened.
5. **M5 — Optional live auto-research demo; paper writing.**

> **Why pilot first** 〔为什么先做 pilot〕: the single biggest experimental risk is whether the compounding curve is *actually* positive. Validate the *trend* small before scaling — do not build full-domain first.

---

## 10. Decisions to Confirm Together / 待共同拍板

1. **Node granularity:** task-only vs task+method (recommend **both**, via `type`).
2. **V-A ground truth:** mine paper-reported deltas / use existing benchmarks (MedMNIST, nnU-Net cross-task) / **both** (recommended).
3. **Leakage strategy:** time-frozen LLM vs **structure-driven + no-graph baseline** (recommended).
4. **First corpus scope:** focused sub-domain (brain/retina) vs full medical imaging (recommend **focused first**, scale later).

---

## 11. References / 文献

*(All arXiv IDs verified 2026-05.)*

- **Edge semantics / transfer:** TaskWeb (2305.13256) — pairwise transfer + transitivity; TaskShop improves source **ranking +10% / top-k precision +38%**. Vu et al. (2005.00770) — task embeddings predict transferability; gains largest when target data scarce.
- **Large-graph extraction:** Intern-Atlas (2604.28158) — LLM extraction at ~1M-paper scale, training-free, evidence-grounded edges.
- **Navigation / open-endedness:** OMNI (2306.01711) — foundation model as model-of-interestingness. AC/DC (2604.14969) — open-ended LLM-expert discovery (paradigm reference / baseline). ACD (2502.07577) — automated capability discovery, human-survey validated.
- **Growth / closed loop:** SkillGraph (2605.12039) — insert/merge/split/deprecate operators, graph–policy co-evolution. AI-Supervisor (2603.24402) — persistent Research World Model, consensus before commit.
- **Governance:** MemOS (2507.03724); Mem0 (2504.19413) — self-edit over append, p95 latency −91% / tokens −90%; FadeMem (2601.18642) — adaptive decay, storage −45%; SSGM (2603.11768) — hallucination vs temporal obsolescence, evidence-gated updates.
- **Retrieval:** Microsoft GraphRAG (Edge et al., 2024) — Leiden community hierarchy; LazyGraphRAG (2025) — indexing at 0.1% of GraphRAG cost; HNSW (Malkov & Yashunin) — million-scale ANN.
- **Idea-generation rivals:** Intern-Atlas; Deep Ideation (2511.02238) — review-trained critic, idea quality **+10.67%**, "surpassing top-conference acceptance"; SciAgents (2409.05556).
