r"""Estimate-level fusion with Covariance Intersection.

Each sensor ``i`` feeds its own local filter (same filter type, nominal
``R_i``), producing ``hat x_i`` and ``P_i``.  Because local estimates share the
same process noise, their errors are correlated with unknown
cross-covariances; the fused estimate therefore uses Covariance
Intersection (Julier & Uhlmann, 1997), which is consistent for any
cross-correlation:

.. math::

    P_f^{-1} = \sum_i \omega_i P_i^{-1},\qquad
    \hat x_f = P_f \sum_i \omega_i P_i^{-1}\hat x_i,\qquad \sum_i\omega_i = 1 .

Weighting:

* ``equal``: ``omega_i = 1/N``;
* ``trust``: ``omega_i = T_i / sum_j T_j`` (SensorTrust estimate-level fusion).

For trust estimation, the innovation of sensor ``i`` is computed against the
*fused* prediction ``hat x_f^-`` (propagated with ``f`` and its Jacobian),
``S_i = H_i P_f^- H_i^T + R_i``: this exposes inconsistencies of a sensor
with respect to the consensus, which the sensor's own local filter would
partly absorb (e.g. a slowly growing bias).
"""

from __future__ import annotations

import time

import numpy as np

from ..estimators.filters import create_filter
from ..trust.detection import create_detector
from ..trust.estimators import create_trust
from ..utils.errors import ConfigurationError
from ..utils.linalg import safe_inv_spd, symmetrize, whiten
from .base import FusionMethod, MethodResult


class EstimateLevelFusion(FusionMethod):
    key = "estimate_level"
    label = "Estimate-level CI fusion"
    category = "estimate_level"

    def __init__(self, label=None, filter="kf", weighting="equal", trust=None, detector=None, key=None,
                 description=None, **params):
        super().__init__(label, **params)
        if weighting not in ("equal", "trust"):
            raise ConfigurationError("estimate_level: weighting must be 'equal' or 'trust'.")
        self.filter_kind = filter
        self.weighting = weighting
        self.trust_spec = trust if trust is not None else ({"method": "first_two_moments"}
                                                           if weighting == "trust" else None)
        self.detector_spec = detector
        if key:
            self.key = key
        if description:
            self.description = description
        if self.trust_spec is not None:
            create_trust(self.trust_spec)
        create_detector(detector)
        if weighting == "trust":
            self.category = "trust"

    def describe(self):
        d = super().describe()
        d.update({"filter": self.filter_kind, "weighting": self.weighting, "trust": self.trust_spec,
                  "detector": self.detector_spec})
        return d

    def run(self, meas, model, oracle=None):
        t0 = time.perf_counter()
        K, N, n = meas.K, meas.N, model.n
        specs = meas.specs
        locs = [create_filter(self.filter_kind, model, [s], meas.x0, meas.P0) for s in specs]
        trust = create_trust(self.trust_spec) if self.trust_spec is not None else None
        if trust is not None:
            trust.reset(specs, K)
        det = create_detector(self.detector_spec)
        if det is not None:
            if trust is None:
                raise ConfigurationError("A detector requires a trust estimator.")
            det.reset(N, K)
        xf, Pf = meas.x0.copy(), meas.P0.copy()
        X = np.full((K, n), np.nan)
        PP = np.full((K, n, n), np.nan)
        ms = [s.m for s in specs]
        innov = [np.full((K, m), np.nan) for m in ms]
        istd = [np.full((K, m), np.nan) for m in ms]
        epsh = [np.full((K, m), np.nan) for m in ms]
        nis_s = np.full((K, N), np.nan)
        W = np.full((K, N), np.nan)
        avail_all = [~np.any(np.isnan(z), axis=1) for z in meas.z]
        u = meas.u
        diverged = False
        notes = []
        for k in range(K):
            if k > 0:
                for lf in locs:
                    lf.predict(u[k - 1])
                F = model.F(xf, u[k - 1])
                xf = model.f(xf, u[k - 1])
                Pf = symmetrize(F @ Pf @ F.T + model.Q)
            avail = [bool(a[k]) for a in avail_all]
            eps_list = [None] * N
            S_list = [None] * N
            for i, s in enumerate(specs):
                if not avail[i]:
                    continue
                z = meas.z[i][k]
                H = s.jacobian(xf)
                nu = z - s.h(xf)
                S = H @ Pf @ H.T + s.R
                e = whiten(nu, S)
                innov[i][k], istd[i][k], epsh[i][k] = nu, np.sqrt(np.diag(S)), e
                nis_s[k, i] = float(e @ e)
                eps_list[i], S_list[i] = e, S
                locs[i].update([(0, z, s.R)])
            if trust is not None:
                T = trust.update(k, eps_list, S_list).copy()
                if det is not None:
                    det.update(k, trust, avail)
            if self.weighting == "trust":
                om = T / T.sum()
            else:
                om = np.full(N, 1.0 / N)
            info = np.zeros((n, n))
            vec = np.zeros(n)
            for i, lf in enumerate(locs):
                Ii = safe_inv_spd(lf.P)
                info += om[i] * Ii
                vec += om[i] * (Ii @ lf.x)
            Pf = symmetrize(safe_inv_spd(info))
            xf = Pf @ vec
            if not (np.all(np.isfinite(xf)) and np.all(np.isfinite(Pf))):
                diverged = True
                notes.append(f"numerical divergence at t = {meas.t[k]:.3f} s")
                break
            X[k], PP[k], W[k] = xf, Pf, om
        res = MethodResult(
            key=self.key, label=self.label, category=self.category, is_oracle=False,
            description=self.description, params=self.describe(), x=X, estimated_states=list(range(n)), P=PP,
            innovation=innov, innovation_std=istd, eps=epsh, nis_sensor=nis_s, weights=W,
            runtime_s=time.perf_counter() - t0, diverged=diverged, notes=notes,
        )
        if trust is not None:
            res.trust = trust.hist_T
            res.moments = trust.diagnostics() or None
        if det is not None:
            res.alarms = det.hist
        return res
