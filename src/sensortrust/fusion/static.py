r"""Static (memoryless) measurement-level fusion baselines.

These methods are only conceptually valid when every sensor measures the same
scalar state component ``x_j`` directly (``H_i = e_j^T``).  They estimate that
component only; the other states are reported as not estimated.  At each step
only the available samples are combined and the weights are renormalised:

* Simple average:          ``hat x = (1/N_k) sum_i z_i``
* Fixed weighted average:  ``hat x = sum_i w_i z_i / sum_i w_i`` (user weights)
* Inverse-variance:        ``w_i \propto 1 / sigma_i^2`` (nominal variances)
* Best-sensor oracle:      the single sensor with the lowest RMSE against the
  ground truth over the whole run (**oracle**, offline reference only).

If no sensor is available at step ``k``, the previous estimate is held.
"""

from __future__ import annotations

import time

import numpy as np

from ..utils.errors import ConfigurationError, MethodNotApplicableError
from .base import FusionMethod, MethodResult, OracleInfo, available_mask


def _common_target(meas) -> int:
    idx = {s.target_index for s in meas.specs}
    if None in idx or len(idx) != 1:
        raise MethodNotApplicableError(
            "Measurement-level averaging requires all sensors to measure the same state component directly "
            f"(sensors measure: {', '.join(s.measured for s in meas.specs)}).")
    return int(next(iter(idx)))


class _StaticAverage(FusionMethod):
    category = "baseline"

    def base_weights(self, meas) -> np.ndarray:
        raise NotImplementedError

    def run(self, meas, model, oracle=None):
        t0 = time.perf_counter()
        j = _common_target(meas)
        K, N = meas.K, meas.N
        Z = np.column_stack([z[:, 0] for z in meas.z])
        A = np.column_stack([available_mask(z) for z in meas.z])
        wb = self.base_weights(meas)
        W = np.where(A, wb[None, :], 0.0)
        s = W.sum(axis=1)
        est = np.full(K, np.nan)
        ok = s > 0
        est[ok] = np.nansum(np.where(A, Z, 0.0) * W, axis=1)[ok] / s[ok]
        # hold last value when nothing is available
        last = float(meas.x0[j])
        for k in range(K):
            if ok[k]:
                last = est[k]
            else:
                est[k] = last
        x = np.full((K, model.n), np.nan)
        x[:, j] = est
        Wn = np.where(ok[:, None], W / np.where(ok, s, 1.0)[:, None], np.nan)
        Wn[~A] = np.nan
        return MethodResult(key=self.key, label=self.label, category=self.category, is_oracle=self.is_oracle,
                            description=self.description, params=self.params, x=x, estimated_states=[j],
                            weights=Wn, runtime_s=time.perf_counter() - t0)


class SimpleAverage(_StaticAverage):
    key = "simple_average"
    label = "Simple average"
    description = "Unweighted mean of the available measurements."

    def base_weights(self, meas):
        return np.ones(meas.N)


class FixedWeightedAverage(_StaticAverage):
    key = "fixed_weights"
    label = "Fixed weighted average"
    description = "Weighted mean with user-defined constant weights."

    def base_weights(self, meas):
        w = self.params.get("weights")
        if w is None:
            raise ConfigurationError("fixed_weights requires 'weights' (one per sensor).")
        w = np.asarray(w, dtype=float)
        if w.size != meas.N or np.any(w < 0) or w.sum() <= 0:
            raise ConfigurationError(f"fixed_weights: need {meas.N} non-negative weights with positive sum.")
        return w


class InverseVarianceWeighting(_StaticAverage):
    key = "inverse_variance"
    label = "Inverse-variance weighting"
    description = "Weighted mean with w_i proportional to 1/sigma_i^2 (nominal variances)."

    def base_weights(self, meas):
        return np.array([1.0 / float(s.R[0, 0]) for s in meas.specs])


class BestSensorOracle(FusionMethod):
    key = "best_sensor_oracle"
    label = "Best-sensor oracle"
    category = "oracle"
    is_oracle = True
    description = ("ORACLE (offline reference, not realizable): output of the single sensor with the lowest "
                   "RMSE against the ground truth.")

    def run(self, meas, model, oracle: OracleInfo | None = None):
        if oracle is None:
            raise MethodNotApplicableError("best_sensor_oracle requires ground-truth (oracle) information.")
        t0 = time.perf_counter()
        j = _common_target(meas)
        truth = oracle.x_true[:, j]
        rmses = []
        for z in meas.z:
            e = z[:, 0] - truth
            rmses.append(np.sqrt(np.nanmean(e**2)) if np.any(np.isfinite(e)) else np.inf)
        b = int(np.argmin(rmses))
        est = meas.z[b][:, 0].copy()
        last = float(meas.x0[j])
        for k in range(meas.K):
            if np.isnan(est[k]):
                est[k] = last
            else:
                last = est[k]
        x = np.full((meas.K, model.n), np.nan)
        x[:, j] = est
        W = np.zeros((meas.K, meas.N))
        W[:, b] = 1.0
        return MethodResult(key=self.key, label=self.label, category=self.category, is_oracle=True,
                            description=self.description, params={**self.params, "selected_sensor": meas.specs[b].name},
                            x=x, estimated_states=[j], weights=W, runtime_s=time.perf_counter() - t0,
                            notes=[f"selected sensor: {meas.specs[b].name}"])
