#!/usr/bin/env python3
"""新方案 STEP 2 + 缺失的 LLM 对照(用 HARDENED 验证,严谨).
A) 训练挑战者(expanding-window,防泄漏):学一个小预测器,看能否超过纯 semantic / persistence.
B) vs 直接问 LLM:让 LLM 给候选边打可信度分,比 AUC —— 凑齐纲领 §9 的三组对照.
"""
import sys, pathlib, numpy as np, collections, json, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_step1 as R
from run_step1_hardened import hard_verify
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "code"))
import llm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def auc(sc, b): return roc_auc_score(b, sc) if 0 < b.sum() < len(b) else float("nan")

# ---------- precompute candidates/features/labels per year ----------
DATA = {}
for Y in [2017, 2018, 2019] + R.TEST:
    cand, edges, adj, nodes = R.candidates(Y)
    X = R.features(cand, Y, adj); bh = hard_verify(cand, Y)
    DATA[Y] = (cand, X, bh)

# ================= A) trained challenger (expanding window) =================
print("==== A) 训练挑战者 vs 纯 semantic / persistence (HARDENED 验证, AUC) ====")
ch_lr, ch_gb, sem, per = [], [], [], []
for Y in R.TEST:
    Xtr = np.vstack([DATA[ty][1] for ty in range(2017, Y)])
    ytr = np.concatenate([DATA[ty][2] for ty in range(2017, Y)])
    cand, Xte, bh = DATA[Y]
    if bh.sum() == 0: continue
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3).fit(Xtr, ytr)
    ch_lr.append(auc(lr.predict_proba(scaler.transform(Xte))[:, 1], bh))
    ch_gb.append(auc(gb.predict_proba(Xte)[:, 1], bh))
    sem.append(auc(Xte[:, 0], bh)); per.append(auc(Xte[:, 1], bh))
    print(f"  Y={Y}: challenger_LR={ch_lr[-1]:.3f} challenger_GB={ch_gb[-1]:.3f} | semantic={sem[-1]:.3f} persistence={per[-1]:.3f}")
print(f"  MEAN: challenger_LR={np.nanmean(ch_lr):.3f}  challenger_GB={np.nanmean(ch_gb):.3f}  "
      f"semantic={np.nanmean(sem):.3f}  persistence={np.nanmean(per):.3f}")
d = np.array(ch_lr) - np.array(sem)
print(f"  challenger_LR - semantic mean = {np.nanmean(d):+.3f}  => {'训练翻盘' if np.nanmean(d)>0 else '训练没翻盘(纯 semantic 更优/持平)'}")

# ================= B) vs direct LLM (sampled, batched) =================
print("\n==== B) vs 直接问 LLM (sampled, HARDENED 验证, AUC) ====")
SYS = "You rate research-transfer plausibility for brain medical imaging."
def llm_score(pairs):
    out = []
    for b in range(0, len(pairs), 10):
        chunk = pairs[b:b+10]
        lines = [f'{i}. source=\"{s}\"  target_task=\"{t}\"' for i,(s,t) in enumerate(chunk)]
        usr = ("For each, rate 0..1 how plausible that knowledge/representations from SOURCE will "
               "transfer to / help TARGET_TASK in the near future.\n" + "\n".join(lines) +
               '\nJSON only: {\"scores\":[0.x, ...]}  (same order)')
        try:
            o = llm.extract_json(llm.call(SYS, usr, max_tokens=600)); sc = o.get("scores", [])
        except Exception:
            sc = [float(x) for x in re.findall(r'[01]\.\d+', llm.call(SYS, usr, max_tokens=600))][:len(chunk)]
        sc = (sc + [0.5]*len(chunk))[:len(chunk)]
        out += [float(x) for x in sc]
    return np.array(out)

llm_a, sem_b = [], []
for Y in R.TEST:
    cand, Xte, bh = DATA[Y]
    pos = [i for i in range(len(cand)) if bh[i] > 0][:40]
    neg = list(R.rng.choice([i for i in range(len(cand)) if bh[i] == 0], min(40, max(1,len(pos))), replace=False))
    idx = pos + neg
    if len(set(bh[idx])) < 2: continue
    pairs = [(R.G.nodes[cand[i][0]]["text"], R.G.nodes[cand[i][1]]["text"]) for i in idx]
    ls = llm_score(pairs); b = bh[idx]
    llm_a.append(auc(ls, b)); sem_b.append(auc(Xte[idx, 0], b))
    print(f"  Y={Y}: sample={len(idx)} (pos={len(pos)})  AUC LLM={llm_a[-1]:.3f}  semantic={sem_b[-1]:.3f}")
print(f"  MEAN: LLM={np.nanmean(llm_a):.3f}  semantic={np.nanmean(sem_b):.3f}  "
      f"=> {'语义(图)优于直接问LLM' if np.nanmean(sem_b)>np.nanmean(llm_a) else 'LLM更强'}")
