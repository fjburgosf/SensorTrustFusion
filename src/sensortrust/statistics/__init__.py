r"""Statistical analysis of repeated experiments.

* :func:`describe` - mean, median, standard deviation, minimum, maximum,
  percentiles (5, 25, 75, 95) and the confidence interval of the mean
  (Student t: ``mean +- t_{1-a/2, n-1} s / sqrt(n)``) for a sample.
* :func:`summarize` - :func:`describe` for every (method, metric) pair of a
  tidy Monte Carlo table.
* :func:`paired_comparison` - paired comparison of two methods evaluated on
  the *same* scenarios (common random numbers): median of differences,
  win rate, Wilcoxon signed-rank test and bootstrap CI of the mean difference.
* :func:`pairwise_tests` - all pairs vs. a reference method with Holm-Bonferroni
  correction of the p-values.
* :func:`ecdf` - empirical cumulative distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def describe(x, confidence: float = 0.95) -> dict:
    a = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None, "p05": None,
                "p25": None, "p75": None, "p95": None, "ci_low": None, "ci_high": None}
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        h = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * sd / np.sqrt(n))
    else:
        h = float("nan")
    p = np.percentile(a, [5, 25, 75, 95])
    return {"n": int(n), "mean": mean, "median": float(np.median(a)), "std": sd, "min": float(a.min()),
            "max": float(a.max()), "p05": float(p[0]), "p25": float(p[1]), "p75": float(p[2]),
            "p95": float(p[3]), "ci_low": mean - h if n > 1 else None, "ci_high": mean + h if n > 1 else None}


def summarize(df: pd.DataFrame, metrics: list[str], by: str = "method", confidence: float = 0.95) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=False):
        for m in metrics:
            if m not in g:
                continue
            d = describe(g[m], confidence)
            rows.append({by: key, "metric": m, **d})
    return pd.DataFrame(rows)


def ecdf(x):
    a = np.sort(np.asarray(x, dtype=float)[np.isfinite(np.asarray(x, dtype=float))])
    return a, np.arange(1, a.size + 1) / max(a.size, 1)


def paired_comparison(a, b, n_boot: int = 2000, seed: int = 0, lower_is_better: bool = True) -> dict:
    """Compare paired samples ``a`` (method A) and ``b`` (method B) run on identical scenarios."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n == 0:
        return {"n": 0}
    d = a - b
    wins = np.mean(d < 0) if lower_is_better else np.mean(d > 0)
    res = {"n": int(n), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
           "win_rate_a": float(wins), "wilcoxon_statistic": None, "p_value": None}
    if n >= 6 and np.any(d != 0):
        w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        res["wilcoxon_statistic"] = float(w.statistic)
        res["p_value"] = float(w.pvalue)
    rng = np.random.default_rng(seed)
    if n > 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        bm = d[idx].mean(axis=1)
        res["ci_low"], res["ci_high"] = (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))
    return res


def holm(pvalues: list[float | None]) -> list[float | None]:
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvalues[i])
    adj: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for r, i in enumerate(order):
        v = min(1.0, (m - r) * pvalues[i])
        running = max(running, v)
        adj[i] = running
    return adj


def pairwise_tests(df: pd.DataFrame, metric: str, reference: str, run_col: str = "run",
                   method_col: str = "method") -> pd.DataFrame:
    wide = df.pivot_table(index=run_col, columns=method_col, values=metric, aggfunc="first")
    if reference not in wide:
        raise ValueError(f"Reference method {reference!r} not present.")
    rows = []
    for m in wide.columns:
        if m == reference:
            continue
        r = paired_comparison(wide[reference].values, wide[m].values)
        rows.append({"reference": reference, "method": m, "metric": metric, **r})
    out = pd.DataFrame(rows)
    if not out.empty and "p_value" in out:
        out["p_value_holm"] = holm(list(out["p_value"].where(out["p_value"].notna(), None)))
    return out
