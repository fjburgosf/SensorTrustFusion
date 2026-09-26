r"""Sensor trust estimators.

Common API (:class:`TrustEstimator`): at each step the fusion layer passes,
for every sensor, its standardised innovation ``eps_i`` (``None`` when the
sensor produced no sample) and receives ``T_i(k) in [T_min, 1]``.

Available estimators
--------------------

``first_two_moments`` -- :class:`FirstTwoMomentsTrustEstimator` (main method)
``first_moment_only`` -- same estimator with the second-moment term disabled (ablation)
``second_moment_only`` -- same estimator with the first-moment term disabled (ablation)
``nis_gate``          -- :class:`NISGateTrust`, classical chi-square validation gate
``constant``          -- :class:`ConstantTrust`, T = nominal reliability (no adaptation)

First-Two-Moments trust (SensorTrust formulation, v1.0.0)
---------------------------------------------------------

The formulation below is the one implemented in SensorTrust Fusion v1.0.0 and
documented in the Technical Manual.  The mapping from moment excesses to trust
is isolated in :meth:`FirstTwoMomentsTrustEstimator.raw_trust`, so alternative
formulations can be studied by subclassing the estimator and overriding that
method, without modifying filters, models, simulation or interface.

With the moment estimates of :mod:`sensortrust.trust.moments`
(``mu_hat``, ``m2_hat``, ``s2_hat = m2_hat - ||mu_hat||^2/m``, ``N_eff``):

.. math::

    d_{1,i}(k) &= \max\!\left(0,\ \frac{\lVert\hat\mu_i(k)\rVert}{\sqrt m}
                  - \frac{\gamma_1}{\sqrt{N_{eff}}}\right)
                  \quad\text{(first-moment excess, in innovation std units)}\\
    d_{2,i}(k) &= \max\!\left(0,\ \hat s^2_i(k) - 1
                  - \gamma_2\sqrt{2/N_{eff}}\right)
                  \quad\text{(dispersion excess, in nominal-variance units)}\\
    \tilde T_i(k) &= \rho_i^0\,\exp\!\left(-\frac{d_{1,i}}{\beta_1}
                  - \frac{d_{2,i}}{\beta_2}\right)

``gamma_1/sqrt(N_eff)`` and ``gamma_2 sqrt(2/N_eff)`` are statistical dead
zones: under ``H0`` the standard deviation of the moment estimates is
``1/sqrt(N_eff)`` and ``sqrt(2/N_eff)`` respectively, so ``gamma`` is a
number of standard deviations tolerated before trust is reduced.
``beta_1`` is the first-moment excess (innovation std units) that reduces
trust by a factor ``e``; ``beta_2`` is the analogous variance excess.
``rho_i^0`` is the nominal reliability of the sensor.  With
``dispersion = "second_moment"`` the raw second moment ``m2_hat`` replaces
``s2_hat`` in ``d_2`` (bias then also contributes to ``d_2``).

Trust dynamics (asymmetric first-order smoothing, recovery-aware):

.. math::

    T_i(k) = \mathrm{clip}\Big(T_i(k-1) + \eta\,[\tilde T_i(k) - T_i(k-1)],\ T_{min},\ 1\Big),
    \qquad \eta = \begin{cases}\eta_\downarrow & \tilde T_i(k) < T_i(k-1)\\
                                \eta_\uparrow & \text{otherwise}\end{cases}

When a sensor produces no sample, its moments and trust are held.

Consensus reference of the first moment (``consensus = true``, default)
-----------------------------------------------------------------------
Innovations are computed against the fused prediction, which is shared by all
sensors.  If a degraded sensor carries most of the information (e.g. the most
precise sensor develops a bias), the fused estimate follows it and the
*healthy* sensors show the larger innovation means; judged one at a time,
the wrong sensors would be distrusted.  A common estimation error ``e``,
however, shifts the innovation mean of every sensor of a *redundancy group*
(sensors with the same measurement function) by the same amount ``-H e``,
whereas a sensor fault shifts only its own mean.  Therefore the physical
innovation means ``p_i = L_i mu_hat_i`` of a group ``G`` are referenced to the
component-wise median over its currently *trusted* members
``G_T = {j in G : T_j >= f_c max_{l in G} T_l}``:

.. math:: \hat\mu_i^{c} = L_i^{-1}\,\big(L_i\hat\mu_i - \mathrm{median}_{j\in G_T} L_j\hat\mu_j\big),

applied when the group has at least ``consensus_min_sensors`` (>= 3) sensors
with a sample.  If fewer than two members are currently trusted, the median is
taken over the members that were trusted within the last
``consensus_memory_samples`` samples, provided they are at least
``consensus_min_sensors``; otherwise no correction is applied.  This prevents a
lock-in where noise momentarily distrusts the healthy sensors at the fault
onset (the faulty sensor would become the only reference), while sensors that
have been persistently distrusted stay out of the reference (sequential and
majority faults).  ``mu_hat^c`` replaces ``mu_hat`` in ``d_1``.  The median is robust to a minority of faulty
sensors, and restricting it to trusted members prevents sensors that are
already isolated from corrupting the reference for later faults.  The second moment and the variance are not
modified.  Groups of two sensors are not corrected: a disagreement between two
sensors cannot be attributed without further redundancy.

Configuration names of the parameters (symbol -> key, default):

=================  ==========================  =========
symbol             configuration key           default
=================  ==========================  =========
moment estimator   ``estimator``               ewma
lambda (EWMA)      ``ewma_lambda``             0.98
N (window)         ``window_length``           100
gamma_1            ``mean_deadzone``           3.0
gamma_2            ``dispersion_deadzone``     3.0
beta_1             ``mean_scale``              0.5
beta_2             ``dispersion_scale``        1.0
eta_down           ``trust_decrease_rate``     0.5
eta_up             ``trust_recovery_rate``     0.02
T_min              ``trust_floor``             0.01
use of mu_hat      ``use_first_moment``        true
use of s2_hat      ``use_second_moment``       true
dispersion         ``dispersion``              variance
consensus          ``consensus``               true
minimum group      ``consensus_min_sensors``   3
f_c                ``consensus_trust_fraction``  0.5
memory window      ``consensus_memory_samples``  200
=================  ==========================  =========
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import chi2

from ..sensors.sensor import SensorSpec
from ..utils.errors import ConfigurationError
from .moments import MomentTracker


class TrustEstimator(ABC):
    key = "abstract"
    label = "Abstract trust estimator"
    uses_moments = False

    def __init__(self, **params):
        self.params = dict(params)
        self.trust_floor = float(params.get("trust_floor", 0.01))
        if not 0.0 < self.trust_floor < 1.0:
            raise ConfigurationError("Trust parameter trust_floor must be in (0, 1).")

    def reset(self, specs: list[SensorSpec], K: int) -> None:
        self.specs = specs
        self.N = len(specs)
        self.K = K
        self.T = np.array([s.nominal_reliability for s in specs], dtype=float)
        self.hist_T = np.full((K, self.N), np.nan)

    @abstractmethod
    def update(self, k: int, eps: list[np.ndarray | None], S: list[np.ndarray | None] | None = None) -> np.ndarray:
        """Return trust of all sensors at step ``k``.

        ``eps[i]``: standardised innovation of sensor ``i`` (``None`` if no sample);
        ``S[i]``: its nominal innovation covariance (optional).
        """

    def diagnostics(self) -> dict[str, np.ndarray]:
        return {}

    def describe(self) -> dict:
        return {"method": self.key, **self.params}


class ConstantTrust(TrustEstimator):
    key = "constant"
    label = "Constant (nominal reliability)"

    def update(self, k, eps, S=None):
        self.hist_T[k] = self.T
        return self.T


class NISGateTrust(TrustEstimator):
    r"""Chi-square validation gate: ``T = rho0`` if ``eps^T eps <= chi2_{m,1-alpha}`` else ``T_min``."""

    key = "nis_gate"
    label = "NIS validation gate"

    def __init__(self, **params):
        super().__init__(**params)
        self.alpha = float(params.get("gate_significance", 0.01))
        if not 0.0 < self.alpha < 1.0:
            raise ConfigurationError("nis_gate: gate_significance must be in (0, 1).")

    def reset(self, specs, K):
        super().reset(specs, K)
        self.gate = np.array([chi2.ppf(1 - self.alpha, s.m) for s in specs])
        self.rho0 = self.T.copy()
        self.hist_nis = np.full((K, self.N), np.nan)

    def update(self, k, eps, S=None):
        for i, e in enumerate(eps):
            if e is None:
                continue
            q = float(e @ e)
            self.hist_nis[k, i] = q
            self.T[i] = self.rho0[i] if q <= self.gate[i] else self.trust_floor
        self.hist_T[k] = self.T
        return self.T

    def diagnostics(self):
        return {"nis_sensor": self.hist_nis}


class FirstTwoMomentsTrustEstimator(TrustEstimator):
    """Trust from the first two moments of the standardised innovation (see module docstring)."""

    key = "first_two_moments"
    label = "First Two Moments of the innovation"
    uses_moments = True
    DEFAULTS = dict(estimator="ewma", ewma_lambda=0.98, window_length=100, mean_deadzone=3.0, dispersion_deadzone=3.0, mean_scale=0.5, dispersion_scale=1.0,
                    trust_decrease_rate=0.5, trust_recovery_rate=0.02, use_first_moment=True, use_second_moment=True,
                    dispersion="variance", trust_floor=0.01, consensus=True, consensus_min_sensors=3,
                    consensus_trust_fraction=0.5, consensus_memory_samples=200)

    def __init__(self, **params):
        p = dict(self.DEFAULTS)
        p.update(params)
        super().__init__(**p)
        self.estimator = str(p["estimator"])
        self.ewma_lambda = float(p["ewma_lambda"])
        self.window = int(p["window_length"])
        self.mean_deadzone, self.dispersion_deadzone = float(p["mean_deadzone"]), float(p["dispersion_deadzone"])
        self.mean_scale, self.dispersion_scale = float(p["mean_scale"]), float(p["dispersion_scale"])
        self.trust_decrease_rate, self.trust_recovery_rate = float(p["trust_decrease_rate"]), float(p["trust_recovery_rate"])
        self.use_first_moment, self.use_second_moment = bool(p["use_first_moment"]), bool(p["use_second_moment"])
        self.dispersion = str(p["dispersion"])
        if self.dispersion not in ("variance", "second_moment"):
            raise ConfigurationError("first_two_moments: dispersion must be 'variance' or 'second_moment'.")
        for nm in ("mean_deadzone", "dispersion_deadzone"):
            if getattr(self, nm) < 0:
                raise ConfigurationError(f"first_two_moments: {nm} must be >= 0.")
        for nm in ("mean_scale", "dispersion_scale"):
            if getattr(self, nm) <= 0:
                raise ConfigurationError(f"first_two_moments: {nm} must be > 0.")
        for nm in ("trust_decrease_rate", "trust_recovery_rate"):
            if not 0.0 < getattr(self, nm) <= 1.0:
                raise ConfigurationError(f"first_two_moments: {nm} must be in (0, 1].")
        if not (self.use_first_moment or self.use_second_moment):
            raise ConfigurationError("first_two_moments: at least one moment must be used.")
        self.consensus = bool(p["consensus"])
        self.consensus_min_sensors = int(p["consensus_min_sensors"])
        self.consensus_trust_fraction = float(p["consensus_trust_fraction"])
        self.consensus_memory_samples = int(p["consensus_memory_samples"])
        if self.consensus_memory_samples < 0:
            raise ConfigurationError("first_two_moments: consensus_memory_samples must be >= 0.")
        if not 0.0 < self.consensus_trust_fraction <= 1.0:
            raise ConfigurationError("first_two_moments: consensus_trust_fraction must be in (0, 1].")
        if self.consensus_min_sensors < 3:
            raise ConfigurationError("first_two_moments: consensus_min_sensors must be >= 3 "
                                     "(with two sensors a disagreement cannot be attributed).")
        # validate the moment estimator parameters early
        MomentTracker(1, self.estimator, self.ewma_lambda, self.window)

    def reset(self, specs, K):
        super().reset(specs, K)
        self.rho0 = self.T.copy()
        self.trackers = [MomentTracker(s.m, self.estimator, self.ewma_lambda, self.window) for s in specs]
        self.L: list = [None] * self.N
        self.mu_eff = [np.zeros(s.m) for s in specs]
        self.last_trusted = np.zeros(self.N, dtype=int)
        self.groups = redundancy_groups(specs, self.consensus_min_sensors) if self.consensus else []
        shape = (K, self.N)
        self.hist = {k: np.full(shape, np.nan) for k in
                     ("mean", "mean_consensus", "second_moment", "variance", "rms", "n_eff", "d1", "d2",
                      "raw_trust")}

    def raw_trust(self, i: int, tr: MomentTracker) -> tuple[float, float, float]:
        """Return ``(T_raw, d1, d2)`` of sensor ``i`` from its moment tracker."""
        ne = tr.n_eff
        mu = self.mu_eff[i]
        mean_norm = float(np.sqrt(mu @ mu / tr.m))
        d1 = max(0.0, mean_norm - self.mean_deadzone / np.sqrt(ne)) if self.use_first_moment else 0.0
        disp = tr.variance if self.dispersion == "variance" else tr.m2
        d2 = max(0.0, disp - 1.0 - self.dispersion_deadzone * np.sqrt(2.0 / ne)) if self.use_second_moment else 0.0
        return float(self.rho0[i] * np.exp(-d1 / self.mean_scale - d2 / self.dispersion_scale)), d1, d2

    def _consensus(self, k: int = 0) -> None:
        """Reference the first moment of each redundancy group to its median (see module docstring)."""
        for i, tr in enumerate(self.trackers):
            self.mu_eff[i] = tr.mu
        for grp in self.groups:
            members = [i for i in grp if self.trackers[i].ready and self.L[i] is not None]
            if len(members) < self.consensus_min_sensors:
                continue
            tmax = max(self.T[i] for i in members)
            trusted = [i for i in members if self.T[i] >= self.consensus_trust_fraction * tmax]
            for i in trusted:
                self.last_trusted[i] = k
            if len(trusted) < 2:
                # no trusted pair: fall back to the members trusted within the recent memory window,
                # provided they still form a group where a majority can exist
                recent = [i for i in members if k - self.last_trusted[i] <= self.consensus_memory_samples]
                if len(recent) < self.consensus_min_sensors:
                    continue
                trusted = recent
            c = np.median(np.array([self.L[i] @ self.trackers[i].mu for i in trusted]), axis=0)
            for i in members:
                L = self.L[i]
                p = L @ self.trackers[i].mu
                self.mu_eff[i] = (p - c) / L[0, 0] if L.shape == (1, 1) else np.linalg.solve(L, p - c)

    def update(self, k, eps, S=None):
        H = self.hist
        for i, e in enumerate(eps):
            if e is not None:
                self.trackers[i].update(e)
                if S is not None and S[i] is not None:
                    Si = S[i]
                    self.L[i] = np.sqrt(Si) if Si.shape == (1, 1) else np.linalg.cholesky(Si)
        self._consensus(k)
        for i, e in enumerate(eps):
            tr = self.trackers[i]
            if e is not None and tr.ready:
                raw, d1, d2 = self.raw_trust(i, tr)
                prev = self.T[i]
                eta = self.trust_decrease_rate if raw < prev else self.trust_recovery_rate
                self.T[i] = min(1.0, max(self.trust_floor, prev + eta * (raw - prev)))
                H["d1"][k, i], H["d2"][k, i], H["raw_trust"][k, i] = d1, d2, raw
            H["mean"][k, i] = tr.signed_mean
            mu = self.mu_eff[i]
            H["mean_consensus"][k, i] = float(mu[0]) if tr.m == 1 else float(np.sqrt(mu @ mu / tr.m))
            H["second_moment"][k, i] = tr.m2
            H["variance"][k, i] = tr.variance
            H["rms"][k, i] = np.sqrt(tr.m2)
            H["n_eff"][k, i] = tr.n_eff
        self.hist_T[k] = self.T
        return self.T

    def diagnostics(self):
        return dict(self.hist)


def redundancy_groups(specs: list[SensorSpec], min_size: int = 3) -> list[list[int]]:
    """Groups of sensors with an identical measurement function (same ``H`` or same named ``h``)."""
    groups: dict = {}
    for i, s in enumerate(specs):
        if s.linear and s.H is not None:
            key = ("H", s.H.shape, tuple(np.round(s.H, 12).ravel()))
        else:
            key = ("h", s.measured, s.m)
        groups.setdefault(key, []).append(i)
    return [g for g in groups.values() if len(g) >= min_size]


TRUST_REGISTRY: dict[str, type[TrustEstimator]] = {
    "first_two_moments": FirstTwoMomentsTrustEstimator,
    "nis_gate": NISGateTrust,
    "constant": ConstantTrust,
}

_ALIASES = {
    "first_moment_only": ("first_two_moments", {"use_first_moment": True, "use_second_moment": False}),
    "second_moment_only": ("first_two_moments", {"use_first_moment": False, "use_second_moment": True}),
}


def create_trust(spec: dict | str | None) -> TrustEstimator:
    if spec is None:
        spec = {"method": "first_two_moments"}
    if isinstance(spec, str):
        spec = {"method": spec}
    spec = dict(spec)
    method = str(spec.pop("method", "first_two_moments"))
    if method in _ALIASES:
        base, extra = _ALIASES[method]
        spec = {**spec, **extra}
        est = TRUST_REGISTRY[base](**spec)
        est.key = method
        est.label = {"first_moment_only": "First moment only (ablation)",
                     "second_moment_only": "Second moment only (ablation)"}[method]
        return est
    try:
        cls = TRUST_REGISTRY[method]
    except KeyError:
        raise ConfigurationError(
            f"Unknown trust method {method!r}. Available: {', '.join(list(TRUST_REGISTRY) + list(_ALIASES))}.")
    return cls(**spec)


def list_trust_methods() -> list[dict]:
    out = [{"key": k, "label": c.label} for k, c in TRUST_REGISTRY.items()]
    out += [{"key": "first_moment_only", "label": "First moment only (ablation)"},
            {"key": "second_moment_only", "label": "Second moment only (ablation)"}]
    return out
