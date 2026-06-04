#!/usr/bin/env python3
"""新方案 STEP 1 — 纯规则版组合式迁移边预测(无训练,按时间切片防泄漏).

实现总纲的 §1-§7 + §9 的三组对照(LLM 对照见 §note):
- 组合式边界:只在"已有节点对、尚未连边"之间生成候选(不造新节点).
- 打分两条腿:语义(MiniLM 余弦)+ 结构(persistence/momentum/共同邻居/出度),全部从"倒带到 Y-1"的图算.
- 命中=三维连续验证分:① 语义命中(对未来真实边,卡严阈值)② 兑现速度(当年高,次年低)③ 兑现强度(几篇在做).
- 评测=排序式:对全部候选排序,用 AUC / 平均验证分@前段 / Spearman,看高验证分是否被排前;只看相对值.
- 权重:开发期(早期年)客观校准后冻死;测试期不动.
- 三组对照:system(语义+结构) vs persistence(纯结构) vs random;growing vs frozen.
NOTE: 数据只有"年"粒度(S2 仅给发表年),故 period=年,非季度.
"""
import sys, pathlib, collections, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "code"))
import embed
from graph import Graph
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = Graph.load(ROOT / "data" / "graph_transfer_resolved.json")
rng = np.random.default_rng(0)
IDS = list(G.nodes)
EMB = embed.embed([G.nodes[i]["text"] for i in IDS])           # (N,384) normalized
EMB = {i: EMB[r] for r, i in enumerate(IDS)}
TASK = {i for i, n in G.nodes.items() if n["type"] == "task"}
FY = {i: G.nodes[i]["first_year"] for i in IDS}
EDGE_YEAR = {(s, d): e["first_year"] for (s, d), e in G.edges.items()}
INDEG = {y: collections.Counter(d for (s, d), yy in EDGE_YEAR.items() if yy == y) for y in range(2013, 2025)}
OUTDEG = {y: collections.Counter(s for (s, d), yy in EDGE_YEAR.items() if yy == y) for y in range(2013, 2025)}

DEV = [2017, 2018, 2019]            # 开发期:校准权重
TEST = [2020, 2021, 2022, 2023, 2024]
M_PER_TASK = 20                     # 每个目标任务取语义最近的 M 个未连源做候选
PREFILTER = 0.30                    # 语义预筛下限
TAU = 0.62                          # 验证时"语义边匹配"的严阈值


def rewind(t):
    nodes = [i for i in IDS if FY[i] and FY[i] <= t]
    edges = {(s, d) for (s, d), y in EDGE_YEAR.items() if y <= t}
    adj = collections.defaultdict(set)
    for (s, d) in edges:
        adj[s].add(d); adj[d].add(s)
    return nodes, edges, adj


def candidates(t_year):
    """预测 t_year:用 <= t_year-1 的图,在已有节点对间生成未连候选."""
    nodes, edges, adj = rewind(t_year - 1)
    nset = set(nodes)
    tasks = [i for i in nodes if i in TASK]
    if not tasks: return [], edges, adj, nodes
    Nmat = np.stack([EMB[i] for i in nodes])
    cand = []
    for t in tasks:
        sims = Nmat @ EMB[t]
        order = np.argsort(-sims)
        c = 0
        for j in order:
            s = nodes[j]
            if s == t or (s, t) in edges or sims[j] < PREFILTER:
                if sims[j] < PREFILTER: break
                continue
            cand.append((s, t, float(sims[j])))
            c += 1
            if c >= M_PER_TASK: break
    return cand, edges, adj, nodes


def features(cand, t_year, adj):
    """结构+语义特征,全部来自 <= t_year-1(无泄漏)."""
    Y = t_year
    persist = lambda n: sum(INDEG[y].get(n, 0) for y in range(2013, Y))
    mom = lambda n: INDEG[Y-1].get(n, 0) - INDEG[Y-2].get(n, 0)
    outd = lambda n: sum(OUTDEG[y].get(n, 0) for y in range(2013, Y))
    X = []
    for (s, t, sem) in cand:
        cn = len(adj[s] & adj[t])
        X.append([sem, np.log1p(persist(t)), mom(t), cn, np.log1p(outd(s)),
                  np.log1p(persist(s)), Y-1-FY[t]])
    return np.array(X, dtype=float)   # cols: sem, persist_t, mom_t, common_nbr, outdeg_s, persist_s, age_t


def verify_scores(cand, t_year):
    """三维连续验证分: 语义命中 × 速度 × 强度. 对未来真实边(Y..Y+2)打分."""
    if not cand: return np.zeros(0), np.zeros(0)
    Cs = np.stack([EMB[s] for s, t, _ in cand]); Ct = np.stack([EMB[t] for s, t, _ in cand])
    verify = np.zeros(len(cand)); binhit = np.zeros(len(cand))
    for off, w_speed in [(0, 1.0), (1, 0.6), (2, 0.35)]:
        Y = t_year + off
        em = [(s, d) for (s, d), y in EDGE_YEAR.items() if y == Y]
        if not em: continue
        Es = np.stack([EMB[s] for s, d in em]); Et = np.stack([EMB[d] for s, d in em])
        simS = Cs @ Es.T; simT = Ct @ Et.T          # (cand, em)
        match = np.minimum(simS, simT)              # 边匹配强度
        best = match.max(1)                          # 语义命中度
        strength = (match >= TAU).sum(1)             # 兑现强度:多少真实边匹配
        sc = np.where(best >= TAU, best, 0.0) * w_speed * (1 + np.log1p(strength))
        verify = np.maximum(verify, sc)
        binhit = np.maximum(binhit, (best >= TAU).astype(float))
    return verify, binhit


def auc(score, binhit):
    if binhit.sum() == 0 or binhit.sum() == len(binhit): return np.nan
    return roc_auc_score(binhit, score)


def main():
    # ---- 开发期:校准 system 权重(逻辑回归当线性校准器,非神经网络) ----
    Xdev, ydev = [], []
    for Y in DEV:
        cand, edges, adj, nodes = candidates(Y)
        if not cand: continue
        X = features(cand, Y, adj); _, bh = verify_scores(cand, Y)
        Xdev.append(X); ydev.append(bh)
    Xdev = np.vstack(Xdev); ydev = np.concatenate(ydev)
    sc = StandardScaler().fit(Xdev)
    cal = LogisticRegression(max_iter=2000).fit(sc.transform(Xdev), ydev)
    print(f"dev candidates={len(ydev)}  positive rate={ydev.mean():.3f}")
    print("calibrated weights:", {f: round(w, 3) for f, w in zip(
        ["sem","persist_t","mom_t","commonNbr","outdeg_s","persist_s","age_t"], cal.coef_[0])})

    # ---- 测试期:滚动评测 + 三组对照 ----
    res = collections.defaultdict(list)
    for Y in TEST:
        cand, edges, adj, nodes = candidates(Y)
        if not cand: continue
        X = features(cand, Y, adj); verify, binhit = verify_scores(cand, Y)
        # arms (排序依据)
        sys_score = cal.predict_proba(sc.transform(X))[:, 1]    # 语义+结构(校准)
        persist_score = X[:, 1]                                  # 纯结构 persistence(目标)
        sem_score = X[:, 0]                                      # 纯语义
        rand_score = rng.random(len(cand))
        # frozen 结构:persistence 只数到 DEV 末(2019)
        frozen_persist = np.array([sum(INDEG[y].get(t, 0) for y in range(2013, 2020)) for (s, t, _) in cand], float)
        for name, s in [("system(sem+struct)", sys_score), ("persistence", persist_score),
                        ("semantic_only", sem_score), ("random", rand_score),
                        ("frozen_persist", frozen_persist)]:
            res[name].append((Y, auc(s, binhit), len(cand), int(binhit.sum())))
        # growing(system 用 <=Y-1)已是默认; 复利对照在 system vs frozen_persist 体现
        print(f"  Y={Y}: candidates={len(cand)}  verified(binary)={int(binhit.sum())} "
              f"({100*binhit.mean():.1f}%)  AUC system={auc(sys_score,binhit):.3f} "
              f"persist={auc(persist_score,binhit):.3f} sem={auc(sem_score,binhit):.3f}")

    print("\n==== AUC over test years (mean), higher=better ====")
    for name in ["system(sem+struct)", "persistence", "semantic_only", "frozen_persist", "random"]:
        aucs = [a for (_, a, _, _) in res[name] if not np.isnan(a)]
        m = np.mean(aucs) if aucs else float("nan")
        print(f"  {name:<20} AUC={m:.3f}")

    print("\n==== 命门: system - persistence (paired over test years) ====")
    sysa = np.array([a for (_, a, _, _) in res["system(sem+struct)"]])
    pera = np.array([a for (_, a, _, _) in res["persistence"]])
    d = sysa - pera
    bs = [rng.choice(d, len(d), True).mean() for _ in range(4000)]
    print(f"  mean diff={d.mean():+.3f}  95% CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] "
          f"{'语义有增量' if np.percentile(bs,2.5)>0 else '语义无显著增量'}")
    print("  per-year (system, persistence):", [(round(s,3),round(p,3)) for s,p in zip(sysa,pera)])

    print("\n==== 复利: growing(system) - frozen_persist ====")
    fza = np.array([a for (_, a, _, _) in res["frozen_persist"]])
    print(f"  per-year diff: {[round(x,3) for x in (sysa-fza)]}  mean={np.mean(sysa-fza):+.3f}")


if __name__ == "__main__":
    main()
