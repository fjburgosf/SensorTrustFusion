r"""Filter consistency metrics (NIS and NEES).

* ``NIS_k = nu_k^T S_k^{-1} nu_k`` (stacked innovation, effective ``S_k``),
  ``~ chi2(m_k)`` for a consistent filter.  Only defined for methods that
  provide innovations and their covariance.
* ``NEES_k = (x_k - hat x_k)^T P_k^{-1} (x_k - hat x_k)``, ``~ chi2(n)``.  Only
  defined for methods that provide ``P_k`` and estimate the full state.

Summaries: mean normalised value (``mean(NIS_k / m_k)``, ``mean(NEES_k)/n``;
both ~1 when consistent) and the fraction of steps inside the two-sided 95 %
chi-square interval.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def nees_series(x_true: np.ndarray, x_hat: np.ndarray, P: np.ndarray) -> np.ndarray:
    e = x_true - x_hat
    ok = np.all(np.isfinite(e), axis=1) & np.all(np.isfinite(P.reshape(P.shape[0], -1)), axis=1)
    out = np.full(x_true.shape[0], np.nan)
    if ok.any():
        sol = np.linalg.solve(P[ok], e[ok][..., None])[..., 0]
        out[ok] = np.einsum("ki,ki->k", e[ok], sol)
    return out


def consistency_summary(values: np.ndarray, dof, mask: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    d = np.broadcast_to(np.asarray(dof, dtype=float), v.shape)
    ok = mask & np.isfinite(v) & (d > 0)
    if not ok.any():
        return {"mean": None, "mean_normalized": None, "fraction_in_95": None}
    vv, dd = v[ok], d[ok]
    lo = chi2.ppf(0.025, dd)
    hi = chi2.ppf(0.975, dd)
    return {
        "mean": float(np.mean(vv)),
        "mean_normalized": float(np.mean(vv / dd)),
        "fraction_in_95": float(np.mean((vv >= lo) & (vv <= hi))),
    }
