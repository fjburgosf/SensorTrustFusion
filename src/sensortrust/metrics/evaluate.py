"""Evaluation of a method result against the scenario ground truth.

Metrics that a method cannot support are *not* computed (reported as
``None``): NIS needs innovations and their covariance, NEES needs ``P_k``
and a full-state estimate, detection metrics need declared alarms, trust and
weight metrics need trust / weights.

Robustness criteria (explicit):

* ``diverged``: non-finite estimate, or ``max |e_k| > divergence_threshold``
  (default ``1000 x`` the largest nominal sensor standard deviation);
* ``success``: not diverged and, if ``rmse_max`` is given, ``RMSE <= rmse_max``;
* ``performance_loss``: ``RMSE / RMSE_nominal - 1`` where ``RMSE_nominal`` is
  obtained by the same method on the nominal twin scenario (same seed, same
  noise, no degradation);
* ``recovery_quality``: ``RMSE_post_fault / RMSE_pre_fault`` (1 = full recovery).
"""

from __future__ import annotations

import numpy as np

from ..fusion.base import MethodResult
from ..simulation.scenario import Scenario
from .consistency import consistency_summary, nees_series
from .estimation import estimation_metrics, full_state_rmse, phase_masks
from .sensor_metrics import detection_metrics, trust_metrics, weight_metrics


def target_index(scenario: Scenario, metrics_cfg: dict) -> int:
    return scenario.model.state_index(metrics_cfg.get("target_state", 0))


def evaluate(scenario: Scenario, res: MethodResult, metrics_cfg: dict, nominal_rmse: float | None = None) -> dict:
    t = scenario.t
    xt = scenario.truth.x
    warm = float(metrics_cfg.get("warmup", 0.0))
    valid = t >= t[0] + warm
    j = target_index(scenario, metrics_cfg)
    fault_any = scenario.fault_windows()
    out: dict = {
        "method": res.label, "key": res.key, "category": res.category, "oracle": res.is_oracle,
        "target_state": scenario.model.state_names[j], "target_unit": scenario.model.state_units[j],
        "runtime_s": res.runtime_s, "diverged_numerically": res.diverged,
    }
    has_truth = bool(getattr(scenario, "has_truth", True))
    if j not in res.estimated_states:
        out["estimation"] = None
        out["notes"] = ["target state not estimated by this method"]
        return out
    if has_truth:
        est = estimation_metrics(t, xt, res.x, j, fault_any, warm)
    else:
        est = None
        out["notes"] = ["no ground truth available: error, NEES and robustness metrics disabled"]
    out["estimation"] = est
    if has_truth and len(res.estimated_states) == scenario.model.n:
        vals = full_state_rmse(xt, res.x, valid)
        out["state_rmse"] = dict(zip(scenario.model.state_names, vals))
    # consistency
    cons = {}
    if res.nis is not None:
        cons["nis"] = consistency_summary(res.nis, res.nis_dof, valid)
    if has_truth and res.P is not None and len(res.estimated_states) == scenario.model.n:
        nees = nees_series(xt, res.x, res.P)
        cons["nees"] = consistency_summary(nees, scenario.model.n, valid)
    out["consistency"] = cons or None
    # per-sensor metrics
    sensors = {}
    delays, f1s, fars, degraded_trust_min = [], [], [], []
    for i, (s, d) in enumerate(zip(scenario.sensors, scenario.data)):
        sm: dict = {"degraded": bool(d.fault_active.any())}
        if res.alarms is not None:
            dm = detection_metrics(t, res.alarms[:, i], d.fault_active, d.fault_meta, valid)
            sm["detection"] = dm
            if sm["degraded"]:
                delays.extend([x for x in dm["detection_delays"]])
                if dm["f1"] is not None:
                    f1s.append(dm["f1"])
            fars.append(dm["false_alarm_rate"])
        if res.trust is not None:
            tm = trust_metrics(t, res.trust[:, i], d.fault_active, d.fault_meta, valid,
                               metrics_cfg.get("trust_threshold", 0.5), metrics_cfg.get("recovery_fraction", 0.9))
            sm["trust"] = tm
            if sm["degraded"] and tm["trust_degraded_min"] is not None:
                degraded_trust_min.append(tm["trust_degraded_min"])
        if res.weights is not None:
            sm["weights"] = weight_metrics(t, res.weights[:, i], d.fault_active, d.fault_meta, valid,
                                           metrics_cfg.get("weight_fraction", 0.5),
                                           metrics_cfg.get("recovery_fraction", 0.9))
        sensors[s.name] = sm
    out["sensors"] = sensors
    if res.alarms is not None:
        found = [x for x in delays if x is not None]
        out["detection_summary"] = {
            "n_faults": len(delays),
            "n_detected": len(found),
            "mean_detection_delay": float(np.mean(found)) if found else None,
            "mean_f1": float(np.mean(f1s)) if f1s else None,
            "mean_false_alarm_rate": float(np.mean([x for x in fars if x is not None])) if fars else None,
        }
    else:
        out["detection_summary"] = None
    out["degraded_sensor_trust_min"] = min(degraded_trust_min) if degraded_trust_min else None
    if est is None:
        out["robustness"] = None
        return out
    # robustness
    thr = metrics_cfg.get("divergence_threshold")
    if thr is None:
        thr = 1000.0 * max(float(np.sqrt(np.max(np.diag(s.R_nominal)))) for s in scenario.sensors)
    diverged = bool(res.diverged or est["rmse"] is None or not np.isfinite(est["rmse"])
                    or (est["max_error"] is not None and est["max_error"] > thr))
    rmse_max = metrics_cfg.get("rmse_max")
    success = (not diverged) and (rmse_max is None or est["rmse"] <= float(rmse_max))
    rob = {"diverged": diverged, "success": bool(success), "divergence_threshold": thr, "rmse_max": rmse_max,
           "performance_loss": None, "recovery_quality": None}
    if nominal_rmse is not None and nominal_rmse > 0 and est["rmse"] is not None:
        rob["nominal_rmse"] = nominal_rmse
        rob["performance_loss"] = est["rmse"] / nominal_rmse - 1.0
    if est.get("rmse_post_fault") and est.get("rmse_pre_fault"):
        rob["recovery_quality"] = est["rmse_post_fault"] / est["rmse_pre_fault"]
    out["robustness"] = rob
    out["degraded_sensor_trust_min"] = min(degraded_trust_min) if degraded_trust_min else None
    return out


SUMMARY_COLUMNS = [
    "method", "category", "oracle", "rmse", "mae", "mse", "nrmse", "max_error", "bias", "error_variance",
    "final_error", "rmse_pre_fault", "rmse_during_fault", "rmse_post_fault", "nis_normalized", "nis_in_95",
    "nees_normalized", "nees_in_95", "mean_detection_delay", "n_detected", "n_faults", "mean_f1",
    "mean_false_alarm_rate", "degraded_sensor_trust_min", "trust_response_time", "trust_recovery_time",
    "weight_share_during_fault_pct", "performance_loss", "recovery_quality", "success",
    "diverged", "runtime_s",
]


def flatten(m: dict) -> dict:
    """One-row summary of an evaluation (used for tables and Monte Carlo)."""
    row = {k: None for k in SUMMARY_COLUMNS}
    row.update({"method": m["method"], "category": m["category"], "oracle": m["oracle"], "runtime_s": m["runtime_s"]})
    est = m.get("estimation")
    if est:
        for k in ("rmse", "mae", "mse", "nrmse", "max_error", "bias", "error_variance", "final_error",
                  "rmse_pre_fault", "rmse_during_fault", "rmse_post_fault"):
            row[k] = est.get(k)
    cons = m.get("consistency") or {}
    if "nis" in cons:
        row["nis_normalized"] = cons["nis"]["mean_normalized"]
        row["nis_in_95"] = cons["nis"]["fraction_in_95"]
    if "nees" in cons:
        row["nees_normalized"] = cons["nees"]["mean_normalized"]
        row["nees_in_95"] = cons["nees"]["fraction_in_95"]
    ds = m.get("detection_summary")
    if ds:
        row["mean_detection_delay"] = ds["mean_detection_delay"]
        row["n_detected"] = ds["n_detected"]
        row["n_faults"] = ds["n_faults"]
        row["mean_f1"] = ds["mean_f1"]
        row["mean_false_alarm_rate"] = ds["mean_false_alarm_rate"]
    row["degraded_sensor_trust_min"] = m.get("degraded_sensor_trust_min")
    for key, sect in (("trust_response_time", "trust"), ("trust_recovery_time", "trust"),
                      ("weight_share_during_fault_pct", "weights")):
        vals = [sm[sect][key] for sm in (m.get("sensors") or {}).values()
                if sm.get("degraded") and sm.get(sect) and sm[sect].get(key) is not None]
        row[key] = float(sum(vals) / len(vals)) if vals else None
    rob = m.get("robustness") or {}
    for k in ("performance_loss", "recovery_quality", "success", "diverged"):
        row[k] = rob.get(k)
    return row
