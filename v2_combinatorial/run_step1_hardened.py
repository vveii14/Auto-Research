#!/usr/bin/env python3
"""Step-1 robustness: HARDENED verification removing semantic 'home advantage'.
A candidate (s,t) is a hit ONLY if a future edge with the EXACT target t (years Y..Y+2)
has a source semantically near s (cos>=TAU). Re-checks 命门 (semantic vs persistence)."""
import numpy as np
import run_step1 as R
from sklearn.metrics import roc_auc_score

def hard_verify(cand, Y):
    bh = np.zeros(len(cand)); fut = {}
    for (s, d), y in R.EDGE_YEAR.items():
        if Y <= y <= Y + 2: fut.setdefault(d, []).append(R.EMB[s])
    for i, (s, t, _) in enumerate(cand):
        if t in fut and (np.stack(fut[t]) @ R.EMB[s]).max() >= R.TAU: bh[i] = 1.0
    return bh

def auc(sc, b): return roc_auc_score(b, sc) if 0 < b.sum() < len(b) else float("nan")

def main():
    sem_a, per_a = [], []
    for Y in R.TEST:
        cand, edges, adj, nodes = R.candidates(Y)
        X = R.features(cand, Y, adj); bh = hard_verify(cand, Y)
        s, p = auc(X[:, 0], bh), auc(X[:, 1], bh)
        sem_a.append(s); per_a.append(p)
        print(f"  Y={Y}: cand={len(cand)} hit={int(bh.sum())} AUC sem={s:.3f} persist={p:.3f}")
    d = np.array(sem_a) - np.array(per_a)
    bs = [R.rng.choice(d, len(d), True).mean() for _ in range(4000)]
    print(f"\nHARDENED semantic-persistence = {d.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
    print(f"mean AUC semantic={np.mean(sem_a):.3f} persistence={np.mean(per_a):.3f}")

if __name__ == "__main__":
    main()
