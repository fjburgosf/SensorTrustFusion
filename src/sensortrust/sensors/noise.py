r"""Measurement noise models.

Every model returns, for a sensor of dimension ``m`` over ``K`` time steps:

* ``samples``  (K x m): the noise realisation ``v_k``;
* ``variance`` (K x m): the *true* instantaneous variance ``Var[v_k]`` (diagonal
  of ``R_k``), which is stored as ground truth and used by the oracle
  ``R``-aware estimator (case B) and by the noise-variance figures.

and ``nominal_std`` (m,): the standard deviation that a datasheet would
report.  Unless the sensor configuration overrides it with ``assumed_std``,
estimators use ``R_i = diag(nominal_std^2)``.

=================  =========================================================
type               definition
=================  =========================================================
gaussian           v_k ~ N(0, sigma^2)
uniform            v_k ~ U(-a, a),  a = sqrt(3) sigma  (Var = sigma^2)
time_varying       v_k ~ N(0, sigma_k^2), sigma_k = sigma (1 + A sin(2 pi t_k / P))
time_dependent     v_k ~ N(0, sigma_k^2), sigma_k = sigma + s t_k
heteroscedastic    v_k ~ N(0, sigma_k^2), sigma_k = sigma + g |h(x_k)|
correlated         AR(1): v_k = phi v_{k-1} + sqrt(1-phi^2) sigma e_k, phi = exp(-Ts/tau_c)
burst              Gaussian with Markov bursts where sigma is multiplied by F
impulsive          Gaussian + Bernoulli(p) impulses ~ N(0, sigma_imp^2)
=================  =========================================================
"""

from __future__ import annotations

import numpy as np

from ..utils.errors import ConfigurationError


def _pos(spec: dict, name: str, default: float, allow_zero: bool = True) -> float:
    try:
        v = float(spec.get(name, default))
    except (TypeError, ValueError):
        raise ConfigurationError(f"Noise parameter '{name}' must be numeric.") from None
    if not np.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        raise ConfigurationError(f"Noise parameter '{name}' must be {'>=' if allow_zero else '>'} 0, got {v}.")
    return v


def _std_vector(spec: dict, m: int) -> np.ndarray:
    std = np.atleast_1d(np.asarray(spec.get("std", 0.1), dtype=float))
    if std.size == 1:
        std = np.full(m, float(std[0]))
    if std.size != m:
        raise ConfigurationError(f"Noise 'std' must have 1 or {m} entries, got {std.size}.")
    if np.any(~np.isfinite(std)) or np.any(std < 0):
        raise ConfigurationError("Noise 'std' must be finite and non-negative.")
    return std


class NoiseModel:
    type_name = "gaussian"

    def __init__(self, spec: dict, m: int):
        self.spec = dict(spec)
        self.m = m
        self.std = _std_vector(self.spec, m)

    @property
    def nominal_std(self) -> np.ndarray:
        return self.std

    def generate(self, t: np.ndarray, clean: np.ndarray, rng: np.random.Generator):
        K = t.size
        v = rng.standard_normal((K, self.m)) * self.std
        return v, np.broadcast_to(self.std**2, (K, self.m)).copy()

    def describe(self) -> dict:
        return {"type": self.type_name, **{k: v for k, v in self.spec.items() if k != "type"}}


class GaussianNoise(NoiseModel):
    type_name = "gaussian"


class UniformNoise(NoiseModel):
    type_name = "uniform"

    def __init__(self, spec, m):
        spec = dict(spec)
        if "half_width" in spec and "std" not in spec:
            spec["std"] = float(spec["half_width"]) / np.sqrt(3.0)
        super().__init__(spec, m)

    def generate(self, t, clean, rng):
        K = t.size
        a = np.sqrt(3.0) * self.std
        v = rng.uniform(-1.0, 1.0, (K, self.m)) * a
        return v, np.broadcast_to(self.std**2, (K, self.m)).copy()


class TimeVaryingGaussianNoise(NoiseModel):
    type_name = "time_varying"

    def __init__(self, spec, m):
        super().__init__(spec, m)
        self.amp = _pos(self.spec, "amplitude", 0.5)
        if self.amp >= 1.0:
            raise ConfigurationError("time_varying noise: 'amplitude' must be < 1 (relative).")
        self.period = _pos(self.spec, "period", 20.0, allow_zero=False)

    def generate(self, t, clean, rng):
        s = self.std[None, :] * (1.0 + self.amp * np.sin(2 * np.pi * t / self.period))[:, None]
        return rng.standard_normal(s.shape) * s, s**2


class TimeDependentNoise(NoiseModel):
    type_name = "time_dependent"

    def __init__(self, spec, m):
        super().__init__(spec, m)
        self.slope = float(self.spec.get("slope", 0.0))

    def generate(self, t, clean, rng):
        s = np.clip(self.std[None, :] + self.slope * t[:, None], 0.0, None)
        return rng.standard_normal(s.shape) * s, s**2


class HeteroscedasticNoise(NoiseModel):
    type_name = "heteroscedastic"

    def __init__(self, spec, m):
        super().__init__(spec, m)
        self.gain = _pos(self.spec, "gain", 0.05)

    def generate(self, t, clean, rng):
        s = self.std[None, :] + self.gain * np.abs(clean)
        return rng.standard_normal(s.shape) * s, s**2


class CorrelatedNoise(NoiseModel):
    type_name = "correlated"

    def __init__(self, spec, m):
        super().__init__(spec, m)
        if "phi" in self.spec:
            self.phi = float(self.spec["phi"])
            self.tau = None
        else:
            self.tau = _pos(self.spec, "correlation_time", 0.1, allow_zero=False)
            self.phi = None
        if self.phi is not None and not (0.0 <= self.phi < 1.0):
            raise ConfigurationError("correlated noise: 'phi' must be in [0, 1).")

    def generate(self, t, clean, rng):
        K = t.size
        Ts = float(t[1] - t[0]) if K > 1 else 1.0
        phi = self.phi if self.phi is not None else float(np.exp(-Ts / self.tau))
        e = rng.standard_normal((K, self.m))
        v = np.empty((K, self.m))
        v[0] = e[0] * self.std
        c = np.sqrt(1.0 - phi**2) * self.std
        for k in range(1, K):
            v[k] = phi * v[k - 1] + c * e[k]
        return v, np.broadcast_to(self.std**2, (K, self.m)).copy()


class BurstNoise(NoiseModel):
    type_name = "burst"

    def __init__(self, spec, m):
        super().__init__(spec, m)
        self.factor = _pos(self.spec, "factor", 5.0)
        self.burst_rate = _pos(self.spec, "burst_rate", 0.05)          # bursts per second
        self.burst_duration = _pos(self.spec, "burst_duration", 0.5, allow_zero=False)  # s

    def generate(self, t, clean, rng):
        K = t.size
        Ts = float(t[1] - t[0]) if K > 1 else 1.0
        p_on = 1.0 - np.exp(-self.burst_rate * Ts)
        p_off = 1.0 - np.exp(-Ts / self.burst_duration)
        u = rng.random(K)
        state = np.zeros(K, dtype=bool)
        s = False
        for k in range(K):
            s = (u[k] >= p_off) if s else (u[k] < p_on)
            state[k] = s
        mult = np.where(state, self.factor, 1.0)[:, None]
        sd = self.std[None, :] * mult
        return rng.standard_normal(sd.shape) * sd, sd**2


class ImpulsiveNoise(NoiseModel):
    type_name = "impulsive"

    def __init__(self, spec, m):
        super().__init__(spec, m)
        self.p = _pos(self.spec, "probability", 0.01)
        if self.p > 1:
            raise ConfigurationError("impulsive noise: 'probability' must be <= 1.")
        self.imp_std = _pos(self.spec, "impulse_std", 10 * float(self.std.max()))

    def generate(self, t, clean, rng):
        K = t.size
        base = rng.standard_normal((K, self.m)) * self.std
        hit = rng.random((K, self.m)) < self.p
        imp = rng.standard_normal((K, self.m)) * self.imp_std
        var = self.std**2 + self.p * self.imp_std**2
        return base + hit * imp, np.broadcast_to(var, (K, self.m)).copy()


NOISE_REGISTRY: dict[str, type[NoiseModel]] = {
    c.type_name: c
    for c in (GaussianNoise, UniformNoise, TimeVaryingGaussianNoise, TimeDependentNoise,
              HeteroscedasticNoise, CorrelatedNoise, BurstNoise, ImpulsiveNoise)
}


def create_noise(spec: dict | float | None, m: int) -> NoiseModel:
    if spec is None:
        spec = {"type": "gaussian", "std": 0.0}
    elif isinstance(spec, (int, float)):
        spec = {"type": "gaussian", "std": float(spec)}
    typ = str(spec.get("type", "gaussian")).lower()
    try:
        cls = NOISE_REGISTRY[typ]
    except KeyError:
        raise ConfigurationError(
            f"Unknown noise type {typ!r}. Available: {', '.join(NOISE_REGISTRY)}.") from None
    return cls(spec, m)
