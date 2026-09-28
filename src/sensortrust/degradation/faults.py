r"""Degradation (fault) types.

Each degradation acts on one sensor and is modulated by a temporal
:class:`~sensortrust.degradation.profiles.Profile` ``g(t)``.  ``s`` denotes
the ``severity`` parameter, ``tau = t - t_0`` the time since onset and
``Ts`` the base sampling period.  A degradation may contribute to one or more
stages of the measurement pipeline (see :mod:`sensortrust.sensors.sensor`):

====================  ======================  ===============================================
type                  stage                   effect
====================  ======================  ===============================================
bias                  additive offset         b_k = s g_k
progressive_bias      additive offset         b_k = (b_0 + s tau_k) r_k   (s = rate, units/s)
drift                 additive offset         d_k = d_{k-1} + s Ts g_k    (cumulative)
random_walk_drift     additive offset         d_k = d_{k-1} + s sqrt(Ts) e_k g_k
periodic              additive offset         b_k = s sin(2 pi tau_k / P) g_k
intermittent          additive offset         b_k = s c_k g_k, c_k Markov on/off chain
variance_increase     extra noise             R_k = R_0 (1 + (s - 1) g_k)   (s = variance factor)
burst_fault           extra noise             N(0, s^2) during Poisson bursts
outliers              extra noise             +-s (0.5 + U(0,1)) with probability p g_k
sensitivity_loss      multiplicative          alpha_k = 1 - s g_k
saturation            post-processing         clip(z, low, high) while g_k > 0
quantization          post-processing         s round(z / s) while g_k > 0
stuck                 post-processing         z_k = z_{k_f} while g_k >= 1/2
dropout               availability            lost with probability s g_k
packet_loss           availability            Gilbert-Elliott bursts while g_k > 0
====================  ======================  ===============================================

``combined`` expands into several of the above sharing one profile.

Ground truth of each degradation: the boolean ``active`` mask (``g_k > 0.01``,
or the "on" state of intermittent faults) and its metadata (type, severity,
profile, onset, recovery) are stored in the scenario dataset.
"""

from __future__ import annotations

import numpy as np

from ..utils.errors import ConfigurationError
from .profiles import Profile

ACTIVE_THRESHOLD = 0.01


class Degradation:
    type_name = "abstract"
    stage = "offset"   # offset | noise | scale | post | availability
    unit_hint = "measurement units"
    description = ""

    def __init__(self, spec: dict):
        self.spec = dict(spec)
        try:
            self.severity = float(self.spec.get("severity", 1.0))
        except (TypeError, ValueError):
            raise ConfigurationError(f"{self.type_name}: 'severity' must be numeric.") from None
        if not np.isfinite(self.severity):
            raise ConfigurationError(f"{self.type_name}: 'severity' must be finite.")
        self.profile = Profile.from_dict(self.spec.get("profile", {"type": "step",
                                                                    "start": self.spec.get("start", 0.0)}))
        self._check()

    def _check(self) -> None:
        pass

    def _p(self, name, default, nonneg=True):
        try:
            v = float(self.spec.get(name, default))
        except (TypeError, ValueError):
            raise ConfigurationError(f"{self.type_name}: '{name}' must be numeric.") from None
        if not np.isfinite(v) or (nonneg and v < 0):
            raise ConfigurationError(f"{self.type_name}: '{name}' must be finite{' and >= 0' if nonneg else ''}.")
        return v

    # -- default no-op stages ------------------------------------------------
    def offset(self, t, g, rng, m):
        return np.zeros((t.size, m))

    def extra_noise(self, t, g, rng, m, nominal_var):
        z = np.zeros((t.size, m))
        return z, z.copy()

    def scale(self, t, g):
        return np.ones(t.size)

    def post(self, z, t, g, rng):
        return z

    def lost(self, t, g, rng):
        return np.zeros(t.size, dtype=bool)

    def active(self, t, g) -> np.ndarray:
        return g > ACTIVE_THRESHOLD

    def describe(self) -> dict:
        d = {k: v for k, v in self.spec.items() if k != "profile"}
        d["type"] = self.type_name
        d["severity"] = self.severity
        d["profile"] = self.profile.to_dict()
        return d


class Bias(Degradation):
    type_name = "bias"
    description = "Constant (profile-modulated) bias b_k = s g_k."

    def offset(self, t, g, rng, m):
        return np.repeat((self.severity * g)[:, None], m, axis=1)


class ProgressiveBias(Degradation):
    type_name = "progressive_bias"
    unit_hint = "measurement units / s"
    description = "Linearly growing bias b_k = b0 + s (t - t_f) (s = rate)."

    def _check(self):
        self.b0 = self._p("b0", 0.0, nonneg=False)

    def offset(self, t, g, rng, m):
        tau = np.clip(t - self.profile.start, 0.0, None)
        on = (t >= self.profile.start).astype(float) * self.profile.envelope(t)
        return np.repeat(((self.b0 + self.severity * tau) * on)[:, None], m, axis=1)

    def active(self, t, g):
        return (t >= self.profile.start) & (self.profile.envelope(t) > ACTIVE_THRESHOLD)


class Drift(Degradation):
    type_name = "drift"
    unit_hint = "measurement units / s"
    description = "Cumulative drift d_k = d_{k-1} + s Ts g_k (persists after recovery)."

    def offset(self, t, g, rng, m):
        Ts = float(t[1] - t[0]) if t.size > 1 else 1.0
        return np.repeat(np.cumsum(self.severity * Ts * g)[:, None], m, axis=1)

    def active(self, t, g):
        return np.cumsum(g) * self.severity != 0


class RandomWalkDrift(Degradation):
    type_name = "random_walk_drift"
    unit_hint = "measurement units / sqrt(s)"
    description = "Random-walk drift d_k = d_{k-1} + s sqrt(Ts) e_k g_k."

    def offset(self, t, g, rng, m):
        Ts = float(t[1] - t[0]) if t.size > 1 else 1.0
        e = rng.standard_normal((t.size, m))
        return np.cumsum(self.severity * np.sqrt(Ts) * e * g[:, None], axis=0)

    def active(self, t, g):
        return np.cumsum(g) > 0


class Periodic(Degradation):
    type_name = "periodic"
    description = "Periodic bias b_k = s sin(2 pi (t-t_f)/P) g_k."

    def _check(self):
        self.period = self._p("period", 5.0)
        if self.period <= 0:
            raise ConfigurationError("periodic: 'period' must be > 0.")

    def offset(self, t, g, rng, m):
        tau = t - self.profile.start
        return np.repeat((self.severity * np.sin(2 * np.pi * tau / self.period) * g)[:, None], m, axis=1)


class Intermittent(Degradation):
    type_name = "intermittent"
    description = "Intermittent bias switching on/off as a Markov chain."

    def _check(self):
        self.mean_on = self._p("mean_on", 1.0)
        self.mean_off = self._p("mean_off", 2.0)
        if self.mean_on <= 0 or self.mean_off <= 0:
            raise ConfigurationError("intermittent: mean_on and mean_off must be > 0.")
        self._state = None

    def _chain(self, t, rng):
        Ts = float(t[1] - t[0]) if t.size > 1 else 1.0
        p_off = 1.0 - np.exp(-Ts / self.mean_on)
        p_on = 1.0 - np.exp(-Ts / self.mean_off)
        u = rng.random(t.size)
        c = np.zeros(t.size)
        s = True
        for k in range(t.size):
            if t[k] >= self.profile.start:
                c[k] = 1.0 if s else 0.0
                s = (u[k] >= p_off) if s else (u[k] < p_on)
        return c

    def offset(self, t, g, rng, m):
        self._state = self._chain(t, rng)
        return np.repeat((self.severity * self._state * g)[:, None], m, axis=1)

    def active(self, t, g):
        if self._state is None:
            return g > ACTIVE_THRESHOLD
        return (self._state > 0) & (g > ACTIVE_THRESHOLD)


class VarianceIncrease(Degradation):
    type_name = "variance_increase"
    unit_hint = "variance factor (-)"
    description = "Noise variance growth R_k = R_0 (1 + (s - 1) g_k)."

    def _check(self):
        if self.severity < 1.0:
            raise ConfigurationError("variance_increase: 'severity' is a variance factor and must be >= 1.")

    def extra_noise(self, t, g, rng, m, nominal_var):
        var = nominal_var * (self.severity - 1.0) * g[:, None]
        return rng.standard_normal((t.size, m)) * np.sqrt(var), var


class BurstFault(Degradation):
    type_name = "burst_fault"
    description = "Poisson bursts of extra Gaussian noise with std s."

    def _check(self):
        self.burst_rate = self._p("burst_rate", 0.2)
        self.burst_duration = self._p("burst_duration", 0.5)
        if self.burst_duration <= 0:
            raise ConfigurationError("burst_fault: 'burst_duration' must be > 0.")

    def extra_noise(self, t, g, rng, m, nominal_var):
        Ts = float(t[1] - t[0]) if t.size > 1 else 1.0
        p_on = 1.0 - np.exp(-self.burst_rate * Ts)
        p_off = 1.0 - np.exp(-Ts / self.burst_duration)
        u = rng.random(t.size)
        burst = np.zeros(t.size)
        s = False
        for k in range(t.size):
            if g[k] > ACTIVE_THRESHOLD:
                s = (u[k] >= p_off) if s else (u[k] < p_on)
            else:
                s = False
            burst[k] = 1.0 if s else 0.0
        var = (self.severity**2) * (burst * g)[:, None] * np.ones((1, m))
        return rng.standard_normal((t.size, m)) * np.sqrt(var), var


class Outliers(Degradation):
    type_name = "outliers"
    description = "Impulsive outliers of amplitude s(0.5+U(0,1)) with probability p g_k."

    def _check(self):
        self.probability = self._p("probability", 0.05)
        if self.probability > 1:
            raise ConfigurationError("outliers: 'probability' must be <= 1.")

    def extra_noise(self, t, g, rng, m, nominal_var):
        hit = rng.random((t.size, m)) < self.probability * g[:, None]
        amp = self.severity * (0.5 + rng.random((t.size, m)))
        sign = np.where(rng.random((t.size, m)) < 0.5, -1.0, 1.0)
        v = hit * amp * sign
        # E[amp^2] = s^2 E[(0.5+U)^2] = s^2 * 13/12
        var = self.probability * g[:, None] * (self.severity**2) * (13.0 / 12.0) * np.ones((1, m))
        return v, var


class SensitivityLoss(Degradation):
    type_name = "sensitivity_loss"
    unit_hint = "fraction (-)"
    description = "Multiplicative gain loss alpha_k = 1 - s g_k."

    def _check(self):
        if not (0.0 <= self.severity <= 1.0):
            raise ConfigurationError("sensitivity_loss: 'severity' must be in [0, 1].")

    def scale(self, t, g):
        return 1.0 - self.severity * g


class Saturation(Degradation):
    type_name = "saturation"
    description = "Degraded measurement range: clip(z, low, high) while active."

    def _check(self):
        self.low = self._p("low", -self.severity if self.severity > 0 else -1.0, nonneg=False)
        self.high = self._p("high", self.severity if self.severity > 0 else 1.0, nonneg=False)
        if self.low >= self.high:
            raise ConfigurationError("saturation: 'low' must be < 'high'.")

    def post(self, z, t, g, rng):
        on = (g > ACTIVE_THRESHOLD)[:, None]
        return np.where(on, np.clip(z, self.low, self.high), z)


class Quantization(Degradation):
    type_name = "quantization"
    description = "Degraded resolution: s round(z/s) while active."

    def _check(self):
        if self.severity <= 0:
            raise ConfigurationError("quantization: 'severity' (quantum) must be > 0.")

    def post(self, z, t, g, rng):
        on = (g > ACTIVE_THRESHOLD)[:, None]
        return np.where(on, self.severity * np.round(z / self.severity), z)


class Stuck(Degradation):
    type_name = "stuck"
    description = "Stuck sensor: output frozen at the value of the fault instant."

    def post(self, z, t, g, rng):
        z = z.copy()
        held = None
        for k in range(t.size):
            if g[k] >= 0.5:
                if held is None:
                    held = z[k].copy()
                z[k] = held
            else:
                held = None
        return z

    def active(self, t, g):
        return g >= 0.5


class Dropout(Degradation):
    type_name = "dropout"
    unit_hint = "probability (-)"
    description = "Random sample loss with probability s g_k."

    def _check(self):
        if not (0.0 <= self.severity <= 1.0):
            raise ConfigurationError("dropout: 'severity' is a probability in [0, 1].")

    def lost(self, t, g, rng):
        return rng.random(t.size) < self.severity * g


class PacketLoss(Degradation):
    type_name = "packet_loss"
    unit_hint = "probability (-)"
    description = "Gilbert-Elliott bursty packet loss: P(good->bad) = s, mean burst length L samples."

    def _check(self):
        if not (0.0 <= self.severity <= 1.0):
            raise ConfigurationError("packet_loss: 'severity' is a probability in [0, 1].")
        self.burst_length = self._p("burst_length", 10.0)
        if self.burst_length < 1:
            raise ConfigurationError("packet_loss: 'burst_length' must be >= 1 sample.")

    def lost(self, t, g, rng):
        p_gb = self.severity
        p_bg = 1.0 / self.burst_length
        u = rng.random(t.size)
        out = np.zeros(t.size, dtype=bool)
        bad = False
        for k in range(t.size):
            if g[k] > ACTIVE_THRESHOLD:
                bad = (u[k] >= p_bg) if bad else (u[k] < p_gb)
            else:
                bad = False
            out[k] = bad
        return out


DEGRADATION_REGISTRY: dict[str, type[Degradation]] = {
    c.type_name: c
    for c in (Bias, ProgressiveBias, Drift, RandomWalkDrift, Periodic, Intermittent,
              VarianceIncrease, BurstFault, Outliers, SensitivityLoss, Saturation,
              Quantization, Stuck, Dropout, PacketLoss)
}


def expand_degradations(specs: list[dict] | None) -> list[dict]:
    """Expand ``combined`` entries into their components (shared profile)."""
    out: list[dict] = []
    for s in specs or []:
        if not isinstance(s, dict):
            raise ConfigurationError("Each degradation must be a mapping with a 'type'.")
        if str(s.get("type", "")).lower() == "combined":
            comps = s.get("components", [])
            if not comps:
                raise ConfigurationError("'combined' degradation requires a non-empty 'components' list.")
            for c in comps:
                c = dict(c)
                c.setdefault("profile", s.get("profile", {"type": "step", "start": 0.0}))
                c["combined_group"] = s.get("name", "combined")
                out.append(c)
        else:
            out.append(dict(s))
    return out


def create_degradation(spec: dict) -> Degradation:
    typ = str(spec.get("type", "")).lower()
    try:
        cls = DEGRADATION_REGISTRY[typ]
    except KeyError:
        raise ConfigurationError(
            f"Unknown degradation type {typ!r}. Available: {', '.join(DEGRADATION_REGISTRY)}, combined.") from None
    return cls(spec)


def list_degradations() -> list[dict]:
    return [{"type": k, "stage": c.stage if c.stage != "offset" else _stage_of(c), "severity_unit": c.unit_hint,
             "description": c.description} for k, c in DEGRADATION_REGISTRY.items()]


def _stage_of(cls) -> str:
    for name in ("offset", "extra_noise", "scale", "post", "lost"):
        if getattr(cls, name) is not getattr(Degradation, name):
            return {"extra_noise": "noise", "lost": "availability"}.get(name, name)
    return "offset"
