r"""Per-sensor metrics: detection, trust and weights.

Detection metrics (only for methods that declare alarms ``D_i(k)``), computed
sample-wise against the fault ground truth ``F_i(k)`` after the warm-up:

* TP = #{D=1, F=1}, FP = #{D=1, F=0}, FN = #{D=0, F=1}, TN = #{D=0, F=0}
* precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R)
* false-alarm rate = FP/(FP+TN), missed-detection rate = FN/(TP+FN)
* detection delay of each degradation: ``T_delay = t_detection - t_fault`` where
  ``t_fault`` is the configured onset and ``t_detection`` the first alarm at or
  after the onset while the degradation is active (``None`` = missed);
* false-alarm events: rising edges of ``D_i`` while ``F_i = 0``.

Trust metrics (only for methods that estimate ``T_i``):

* nominal mean / variance: over samples before the first fault of the sensor
  (all samples if the sensor is never degraded);
* degraded mean / minimum: over samples where ``F_i = 1``;
* response time: first time after the onset with ``T_i < trust_threshold``;
* recovery time: first time after the end of the fault with
  ``T_i >= recovery_fraction * nominal mean``;
* reduction ratio: degraded mean / nominal mean.

Weight metrics (methods with weights ``w_i``): nominal mean, minimum,
mean share during the fault (percentage of the total weight), reduction time
(``w_i < weight_fraction * nominal mean``) and recovery time (back to
``>= recovery_fraction * nominal mean``).
"""

from __future__ import annotations

import numpy as np


def _first_time(t, cond, start_k):
    idx = np.flatnonzero(cond[start_k:])
    return float(t[start_k + idx[0]]) if idx.size else None


def _nanmean(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if np.any(np.isfinite(a)) else None


def _nanvar(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanvar(a)) if np.sum(np.isfinite(a)) > 1 else None


def _fault_bounds(t, fault, metas):
    """Onset (configured) and end (last active sample) of the sensor's fault period."""
    idx = np.flatnonzero(fault)
    if idx.size == 0:
        return None, None, None
    onsets = [m["onset_time"] for m in metas if m.get("first_active_time") is not None]
    onset = min(onsets) if onsets else float(t[idx[0]])
    k_on = int(np.searchsorted(t, onset))
    end_k = int(idx[-1])
    k_end = end_k + 1 if end_k < t.size - 1 else None
    return onset, k_on, k_end


def detection_metrics(t, alarms_i, fault_i, metas, valid) -> dict:
    D = alarms_i.astype(bool) & valid
    F = fault_i.astype(bool) & valid
    tp = int(np.sum(D & F))
    fp = int(np.sum(D & ~F))
    fn = int(np.sum(~D & F))
    tn = int(np.sum(~D & ~F & valid))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    rising = np.flatnonzero(np.diff(alarms_i.astype(int), prepend=0) == 1)
    fa_events = int(np.sum(valid[rising] & ~fault_i[rising]))
    delays = []
    for m in metas:
        if m.get("first_active_time") is None:
            delays.append(None)
            continue
        k0 = int(np.searchsorted(t, m["onset_time"]))
        k1 = int(np.searchsorted(t, m["last_active_time"], side="right"))
        hit = np.flatnonzero(alarms_i[k0:k1])
        delays.append(float(t[k0 + hit[0]] - m["onset_time"]) if hit.size else None)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
        "missed_detection_rate": fn / (tp + fn) if tp + fn else None,
        "false_alarm_events": fa_events,
        "detection_delays": delays,
        "detection_delay": delays[0] if delays else None,
        "detected": (delays[0] is not None) if delays else None,
    }


def trust_metrics(t, T_i, fault_i, metas, valid, trust_threshold=0.5, recovery_fraction=0.9) -> dict:
    onset, k_on, k_end = _fault_bounds(t, fault_i, metas)
    nominal = valid.copy()
    if k_on is not None:
        nominal[k_on:] = False
    nom_mean = _nanmean(T_i[nominal])
    out = {
        "trust_nominal_mean": nom_mean,
        "trust_nominal_variance": _nanvar(T_i[nominal]),
        "trust_min": _nanmean([np.nanmin(T_i[valid])]) if np.any(np.isfinite(T_i[valid])) else None,
        "trust_variance": _nanvar(T_i[valid]),
        "trust_degraded_mean": None, "trust_degraded_min": None, "trust_reduction_ratio": None,
        "trust_response_time": None, "trust_recovery_time": None,
    }
    if k_on is None:
        return out
    F = fault_i & valid
    deg = _nanmean(T_i[F])
    out["trust_degraded_mean"] = deg
    out["trust_degraded_min"] = float(np.nanmin(T_i[F])) if np.any(np.isfinite(T_i[F])) else None
    if deg is not None and nom_mean:
        out["trust_reduction_ratio"] = deg / nom_mean
    tr = _first_time(t, np.nan_to_num(T_i, nan=1.0) < trust_threshold, k_on)
    out["trust_response_time"] = tr - onset if tr is not None else None
    if k_end is not None and nom_mean:
        rt = _first_time(t, np.nan_to_num(T_i, nan=0.0) >= recovery_fraction * nom_mean, k_end)
        out["trust_recovery_time"] = rt - float(t[k_end]) if rt is not None else None
    return out


def weight_metrics(t, w_i, fault_i, metas, valid, weight_fraction=0.5, recovery_fraction=0.9) -> dict:
    onset, k_on, k_end = _fault_bounds(t, fault_i, metas)
    nominal = valid.copy()
    if k_on is not None:
        nominal[k_on:] = False
    nom = _nanmean(w_i[nominal])
    out = {
        "weight_nominal_mean": nom,
        "weight_mean": _nanmean(w_i[valid]),
        "weight_min": float(np.nanmin(w_i[valid])) if np.any(np.isfinite(w_i[valid])) else None,
        "weight_share_during_fault_pct": None, "weight_reduction_time": None, "weight_recovery_time": None,
    }
    if k_on is None or nom is None:
        return out
    F = fault_i & valid
    m = _nanmean(w_i[F])
    out["weight_share_during_fault_pct"] = 100.0 * m if m is not None else None
    tr = _first_time(t, np.nan_to_num(w_i, nan=np.inf) < weight_fraction * nom, k_on)
    out["weight_reduction_time"] = tr - onset if tr is not None else None
    if k_end is not None:
        rt = _first_time(t, np.nan_to_num(w_i, nan=-np.inf) >= recovery_fraction * nom, k_end)
        out["weight_recovery_time"] = rt - float(t[k_end]) if rt is not None else None
    return out
