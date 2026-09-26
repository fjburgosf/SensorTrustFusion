r"""Centralized (measurement-level) filter fusion.

All available measurements at step ``k`` are stacked

.. math:: z_k = [z_{1,k}; \dots; z_{N,k}],\quad H = [H_1; \dots; H_N],\quad
          R^{eff}_k = \mathrm{blkdiag}(R^{eff}_{1,k}, \dots, R^{eff}_{N,k})

and processed in one KF / EKF / UKF update.  Missing samples are removed
from the stack.  The covariance strategy defines ``R^eff_{i,k}``:

========  ====================================================================
case      strategy
========  ====================================================================
A         ``nominal``: ``R^eff_i = R_i`` (nominal, fixed)
B         ``oracle``: ``R^eff_{i,k} = diag(true Var[v_{i,k}])`` (**oracle**, knows the
          true noise variance, not the bias)
C         ``adaptive``: Sage-Husa covariance matching
          ``R_hat_{i,k} = (1-d_k) R_hat_{i,k-1} + d_k (nu nu^T - H P^- H^T)``,
          ``d_k = (1-b)/(1-b^{k+1})``, projected so that ``R_hat >= r_floor R_i``
D         ``trust``: ``R^eff_{i,k} = R_i / max(T_min, T_i(k) / max_j T_j(k))``
          (SensorTrust; relative trust, the most trusted sensor keeps ``R_i``)
========  ====================================================================

Innovation validation gate (strategy ``trust`` only, ``innovation_gate`` =
significance ``p_g``, default ``1e-6``; ``null`` disables it): a sample with
``eps_i^T eps_i > chi2_{m_i, 1-p_g}`` is excluded from the state update of
that step (it still enters the trust statistics).  Rationale: an abrupt
error of many standard deviations on a *dominant* (most precise) sensor would
otherwise pull the prior toward the faulty value before the moment
statistics can react, after which the healthy sensors look inconsistent.
If every available sample of a step exceeds the gate, no sample is gated
(prevents lock-out after long outages).

For every sensor the innovation at the prior and its *nominal* covariance
``S_i = H_i P^- H_i^T + R_i`` are recorded; trust estimators receive the
standardised innovation ``eps_i = L_i^{-1} nu_i`` (``S_i = L_i L_i^T``).
Using the nominal (not the effective) ``R_i`` keeps the trust statistic a
test of the hypothesis "the sensor behaves as specified".

Reported weights are the relative information of each available sensor,
``w_i = (1/r_i) / sum_j (1/r_j)`` with ``r_i = tr(R^eff_i)/m_i``; for sensors
measuring the same quantity they are the relative weights of the measurements
in the Kalman update and reduce to ``T_i / sum_j T_j`` when all nominal
variances are equal (case D).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import chi2

from ..estimators.filters import create_filter
from ..trust.detection import create_detector
from ..trust.estimators import create_trust
from ..utils.errors import ConfigurationError, MethodNotApplicableError
from ..utils.linalg import clip_spd, whiten
from .base import FusionMethod, MethodResult, OracleInfo

R_STRATEGIES = ("nominal", "oracle", "adaptive", "trust")


class CentralizedFilterFusion(FusionMethod):
    key = "centralized"
    label = "Centralized filter"
    category = "filter"

    def __init__(self, label=None, filter="kf", r_strategy="nominal", trust=None, detector=None,
                 adaptive=None, filter_params=None, key=None, description=None, use_sensors=None,
                 innovation_gate=1e-6, **params):
        super().__init__(label, **params)
        if r_strategy not in R_STRATEGIES:
            raise ConfigurationError(f"Unknown r_strategy {r_strategy!r}; use one of {R_STRATEGIES}.")
        self.filter_kind = str(filter)
        self.r_strategy = r_strategy
        self.trust_spec = trust
        self.detector_spec = detector
        self.adaptive = {"forgetting_factor": 0.99, "min_variance_ratio": 0.1, **(adaptive or {})}
        self.filter_params = dict(filter_params or {})
        self.innovation_gate = None if innovation_gate in (None, False, 0) else float(innovation_gate)
        if self.innovation_gate is not None and not 0.0 < self.innovation_gate < 1.0:
            raise ConfigurationError("innovation_gate must be a significance level in (0, 1) or null.")
        self.use_sensors = None if use_sensors is None else [str(x) for x in (
            [use_sensors] if isinstance(use_sensors, str) else use_sensors)]
        if key:
            self.key = key
        if description:
            self.description = description
        if r_strategy == "trust" and trust is None:
            self.trust_spec = {"method": "first_two_moments"}
        if r_strategy == "oracle":
            self.is_oracle = True
            self.category = "oracle"
        elif r_strategy == "adaptive":
            self.category = "adaptive"
        elif r_strategy == "trust":
            self.category = "trust"
        # validate trust / detector specs now (configuration errors early)
        if self.trust_spec is not None:
            create_trust(self.trust_spec)
        create_detector(self.detector_spec)
        b = float(self.adaptive["forgetting_factor"])
        if not 0.0 < b < 1.0:
            raise ConfigurationError("adaptive.forgetting_factor must be in (0, 1).")

    def describe(self):
        d = super().describe()
        d.update({"filter": self.filter_kind, "r_strategy": self.r_strategy, "trust": self.trust_spec,
                  "detector": self.detector_spec, "filter_params": self.filter_params,
                  "use_sensors": self.use_sensors})
        if self.r_strategy == "trust":
            d["innovation_gate"] = self.innovation_gate
        if self.r_strategy == "adaptive":
            d["adaptive"] = self.adaptive
        return d

    def run(self, meas, model, oracle: OracleInfo | None = None):
        if self.r_strategy == "oracle" and oracle is None:
            raise MethodNotApplicableError("The oracle-R filter requires ground-truth noise variances.")
        t0 = time.perf_counter()
        K, N, n = meas.K, meas.N, model.n
        specs = meas.specs
        filt = create_filter(self.filter_kind, model, specs, meas.x0, meas.P0, **self.filter_params)
        trust = create_trust(self.trust_spec) if self.trust_spec is not None else None
        if trust is not None:
            trust.reset(specs, K)
        det = create_detector(self.detector_spec)
        if det is not None:
            if trust is None:
                raise ConfigurationError("A detector requires a trust estimator (set 'trust').")
            det.reset(N, K)

        ms = [s.m for s in specs]
        Rnom = [s.R for s in specs]
        Rhat = [R.copy() for R in Rnom]
        n_upd = np.zeros(N, dtype=int)
        b = float(self.adaptive["forgetting_factor"])
        r_floor = float(self.adaptive["min_variance_ratio"])
        t_min = trust.trust_floor if trust is not None else 0.01

        X = np.full((K, n), np.nan)
        PP = np.full((K, n, n), np.nan)
        innov = [np.full((K, m), np.nan) for m in ms]
        istd = [np.full((K, m), np.nan) for m in ms]
        epsh = [np.full((K, m), np.nan) for m in ms]
        nis_s = np.full((K, N), np.nan)
        nis = np.full(K, np.nan)
        dof = np.zeros(K, dtype=int)
        W = np.full((K, N), np.nan)
        reff = np.full((K, N), np.nan)
        rhat_h = np.full((K, N), np.nan) if self.r_strategy == "adaptive" else None
        zs = meas.z
        avail_all = [~np.any(np.isnan(z), axis=1) for z in zs]
        if self.use_sensors is not None:
            names = [s.name for s in specs]
            unknown = [x for x in self.use_sensors if x not in names]
            if unknown:
                raise ConfigurationError(f"use_sensors: unknown sensor(s) {unknown}; available {names}.")
            avail_all = [a if s.name in self.use_sensors else np.zeros_like(a) for a, s in zip(avail_all, specs)]
        u = meas.u
        diverged = False
        notes: list[str] = []
        use_gate = self.r_strategy == "trust" and self.innovation_gate is not None
        gate_thr = {m: float(chi2.ppf(1.0 - self.innovation_gate, m)) for m in set(ms)} if use_gate else {}
        gated = np.zeros((K, N), dtype=bool) if use_gate else None

        for k in range(K):
            if k > 0:
                filt.predict(u[k - 1])
            avail = [bool(a[k]) for a in avail_all]
            eps_list: list = [None] * N
            S_list: list = [None] * N
            nus: list = [None] * N
            pzz: list = [None] * N
            for i in range(N):
                if not avail[i]:
                    continue
                nu, S, H, Pzz = filt.innovation(i, zs[i][k], Rnom[i])
                e = whiten(nu, S)
                innov[i][k] = nu
                istd[i][k] = np.sqrt(np.diag(S))
                epsh[i][k] = e
                nis_s[k, i] = float(e @ e)
                eps_list[i], nus[i], pzz[i], S_list[i] = e, nu, Pzz, S
            T = None
            if trust is not None:
                T = trust.update(k, eps_list, S_list)
                if det is not None:
                    det.update(k, trust, avail)
            items = []
            info = np.zeros(N)
            if self.r_strategy == "trust":
                Tav = [T[i] for i in range(N) if avail[i]]
                Tref = max(Tav) if Tav else 1.0
            skip = [False] * N
            if use_gate:
                over = [avail[i] and nis_s[k, i] > gate_thr[ms[i]] for i in range(N)]
                if any(over) and not all(over[i] for i in range(N) if avail[i]):
                    skip = over
                    gated[k] = over
            for i in range(N):
                if not avail[i] or skip[i]:
                    continue
                if self.r_strategy == "nominal":
                    Re = Rnom[i]
                elif self.r_strategy == "oracle":
                    Re = np.diag(np.maximum(oracle.true_var[i][k], 1e-12))
                elif self.r_strategy == "adaptive":
                    n_upd[i] += 1
                    d = (1.0 - b) / (1.0 - b ** (n_upd[i] + 1))
                    nu = nus[i]
                    Rn = (1.0 - d) * Rhat[i] + d * (np.outer(nu, nu) - pzz[i])
                    Rhat[i] = clip_spd(Rn, r_floor * Rnom[i])
                    Re = Rhat[i]
                    rhat_h[k, i] = float(np.trace(Re)) / ms[i]
                else:  # trust
                    rel = max(t_min, T[i] / Tref) if Tref > 0 else t_min
                    Re = Rnom[i] / rel
                items.append((i, zs[i][k], Re))
                r_i = float(np.trace(Re)) / ms[i]
                reff[k, i] = r_i
                info[i] = 1.0 / r_i
            if items:
                nis[k], dof[k] = filt.update(items)
                s = info.sum()
                for i in range(N):
                    if avail[i]:
                        W[k, i] = info[i] / s
            if not (np.all(np.isfinite(filt.x)) and np.all(np.isfinite(filt.P))):
                diverged = True
                notes.append(f"numerical divergence at t = {meas.t[k]:.3f} s")
                break
            X[k] = filt.x
            PP[k] = filt.P

        res = MethodResult(
            key=self.key, label=self.label, category=self.category, is_oracle=self.is_oracle,
            description=self.description, params=self.describe(), x=X, estimated_states=list(range(n)), P=PP,
            innovation=innov, innovation_std=istd, eps=epsh, nis_sensor=nis_s, nis=nis, nis_dof=dof,
            weights=W, r_eff=reff, r_hat=rhat_h, runtime_s=time.perf_counter() - t0, diverged=diverged,
            notes=notes,
        )
        if trust is not None:
            res.trust = trust.hist_T
            diag = trust.diagnostics()
            if diag:
                res.moments = diag
        if gated is not None:
            res.detector_stats = {"gated": gated}
        if det is not None:
            res.alarms = det.hist
            if hasattr(det, "stat1"):
                res.detector_stats = {**(res.detector_stats or {}), "mean_test": det.stat1,
                                      "dispersion_test": det.stat2}
        return res
