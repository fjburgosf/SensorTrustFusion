r"""Estimation-error metrics.

For the evaluated state component ``j`` and the evaluation set ``I`` (all
steps after the warm-up, optionally restricted to a phase), with
``e_k = x_{k,j} - hat x_{k,j}``:

==============  =============================================================
RMSE            sqrt( (1/|I|) sum e_k^2 )
MSE             (1/|I|) sum e_k^2
MAE             (1/|I|) sum |e_k|
NRMSE           RMSE / (max_k x_{k,j} - min_k x_{k,j})   (range-normalised)
max_error       max |e_k|
bias            (1/|I|) sum e_k       (mean error)
error_variance  (1/(|I|-1)) sum (e_k - bias)^2
final_error     |e_K|
==============  =============================================================

Phases (relative to the fault ground truth of *all* sensors): ``pre_fault``
(before the first fault onset), ``during_fault`` (any sensor degraded) and
``post_fault`` (after the last fault has ended, if it ends).
"""

from __future__ import annotations

import numpy as np


def error_stats(e: np.ndarray, truth: np.ndarray | None = None) -> dict:
    e = np.asarray(e, dtype=float)
    ok = np.isfinite(e)
    n = int(ok.sum())
    if n == 0:
        return {k: None for k in ("rmse", "mse", "mae", "nrmse", "max_error", "bias", "error_variance", "n")}
    ee = e[ok]
    mse = float(np.mean(ee**2))
    out = {
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "mae": float(np.mean(np.abs(ee))),
        "max_error": float(np.max(np.abs(ee))),
        "bias": float(np.mean(ee)),
        "error_variance": float(np.var(ee, ddof=1)) if n > 1 else 0.0,
        "n": n,
    }
    if truth is not None:
        tr = np.asarray(truth)[ok]
        rng = float(np.max(tr) - np.min(tr)) if tr.size else 0.0
        out["nrmse"] = out["rmse"] / rng if rng > 0 else None
    else:
        out["nrmse"] = None
    return out


def phase_masks(t: np.ndarray, fault_any: np.ndarray, warmup: float) -> dict[str, np.ndarray]:
    valid = t >= t[0] + warmup
    idx = np.flatnonzero(fault_any)
    if idx.size == 0:
        return {"all": valid, "pre_fault": valid, "during_fault": np.zeros_like(valid),
                "post_fault": np.zeros_like(valid)}
    first, last = idx[0], idx[-1]
    k = np.arange(t.size)
    pre = valid & (k < first)
    during = valid & fault_any
    post = valid & (k > last) if last < t.size - 1 else np.zeros_like(valid)
    return {"all": valid, "pre_fault": pre, "during_fault": during, "post_fault": post}


def estimation_metrics(t, x_true, x_hat, state_index: int, fault_any, warmup: float) -> dict:
    j = state_index
    e = x_hat[:, j] - x_true[:, j]
    masks = phase_masks(t, fault_any, warmup)
    out = error_stats(e[masks["all"]], x_true[masks["all"], j])
    fin = np.flatnonzero(np.isfinite(e))
    out["final_error"] = float(abs(e[fin[-1]])) if fin.size else None
    for ph in ("pre_fault", "during_fault", "post_fault"):
        m = masks[ph]
        out[f"rmse_{ph}"] = error_stats(e[m])["rmse"] if m.any() else None
    return out


def full_state_rmse(x_true, x_hat, warmup_mask) -> list[float | None]:
    out = []
    for j in range(x_true.shape[1]):
        e = x_hat[warmup_mask, j] - x_true[warmup_mask, j]
        out.append(float(np.sqrt(np.nanmean(e**2))) if np.any(np.isfinite(e)) else None)
    return out
