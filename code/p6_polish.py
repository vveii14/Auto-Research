"""P6 (free polish): aggregate p5 eval across many T0, with bootstrap 95% CIs,
and the decisive PAIRED test growing-frozen (CI excluding 0 => real signal).
Still 900-paper pilot — this only judges whether the signal is real vs noise.
"""
import numpy as np
import p5_eval as p5

T0S = [2017, 2018, 2019, 2020]
ARMS = ["growing", "frozen", "cooccur", "nograph_llm", "random"]
FOCUS = [10, 20, 50]
rng = np.random.default_rng(0)


def boot_ci(vals, n=4000):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (0, 0, 0)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return (vals.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5))


def main():
    # collect per-(T0,year) precision@k cells, keeping growing/frozen aligned
    cells = {a: {k: [] for k in p5.KS} for a in ARMS}
    for T0 in T0S:
        print(f"... running T0={T0}")
        acc = p5.run(T0)
        for a in ARMS:
            for k in p5.KS:
                cells[a][k].extend(acc[a][k]["p"])

    n = len(cells["growing"][10])
    print(f"\n==== precision@k across {n} (T0,year) cells — mean [95% CI] ====")
    for k in FOCUS:
        print(f"\n-- k={k} --")
        for a in ARMS:
            m, lo, hi = boot_ci(cells[a][k])
            print(f"  {a:<12} {m:.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\n==== DECISIVE TEST: growing - frozen (paired) ====")
    for k in FOCUS:
        g = np.array(cells["growing"][k]); f = np.array(cells["frozen"][k])
        diff = g - f
        m, lo, hi = boot_ci(diff)
        verdict = "REAL (CI>0)" if lo > 0 else ("noise (CI spans 0)" if hi > 0 else "negative")
        print(f"  k={k:<3} mean diff={m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")

    print("\n==== growing vs no-graph-LLM (paired) — is the graph worth it? ====")
    for k in FOCUS:
        g = np.array(cells["growing"][k]); nl = np.array(cells["nograph_llm"][k])
        m, lo, hi = boot_ci(g - nl)
        print(f"  k={k:<3} mean diff={m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  -> {'REAL (CI>0)' if lo>0 else 'noise'}")


if __name__ == "__main__":
    main()
