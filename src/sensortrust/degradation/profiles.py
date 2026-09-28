r"""Temporal degradation profiles.

A profile is an *activation function* ``g(t) in [0, 1]`` that modulates the
severity of a degradation.  It is the product of an onset shape ``o`` and a
recovery envelope ``r``:

.. math::

    g(t) = \begin{cases} 0 & t < t_0 \\ o(t - t_0)\, r(t) & t \ge t_0 \end{cases}

Onset shapes (``tau = t - t_0``, ``T_r`` = ``rise_time``):

==============  ==============================================================
``step``        o = 1
``ramp``        o = min(1, tau / T_r)
``exponential`` o = 1 - exp(-tau / T_r)
``sigmoid``     o = (s(tau) - s(0)) / (s(T_r) - s(0)),  s(tau) = 1/(1+exp(-10(tau/T_r - 1/2))),
                clipped to 1 for tau >= T_r
``sinusoidal``  o = (1 - cos(2 pi tau / P)) / 2
``random_walk`` o_k = clip(o_{k-1} + sigma_rw sqrt(Ts) e_k, 0, 1), o at onset = 0
``intermittent`` two-state Markov chain (on/off) with mean durations
                ``mean_on`` / ``mean_off`` [s], starting "on"
``recovery``    alias of ``step`` (or ``ramp`` if ``rise_time`` > 0) that
                requires a recovery time
==============  ==============================================================

Recovery envelope (``t_r`` = ``recovery_time``, ``D_r`` = ``recovery_duration``):

* no recovery: r = 1;
* ``linear``: r = max(0, 1 - (t - t_r)/D_r) for t >= t_r (D_r = 0 -> abrupt end);
* ``exponential``: r = exp(-(t - t_r)/D_r) for t >= t_r.

``duration`` is a shorthand for ``recovery_time = start + duration``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.errors import ConfigurationError

PROFILE_TYPES = ("step", "ramp", "exponential", "sigmoid", "sinusoidal", "random_walk",
                 "intermittent", "recovery")


def _num(d: dict, key: str, default=None, positive=False, nonneg=False):
    v = d.get(key, default)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ConfigurationError(f"Profile parameter '{key}' must be numeric.") from None
    if not np.isfinite(v):
        raise ConfigurationError(f"Profile parameter '{key}' must be finite.")
    if positive and v <= 0:
        raise ConfigurationError(f"Profile parameter '{key}' must be > 0, got {v}.")
    if nonneg and v < 0:
        raise ConfigurationError(f"Profile parameter '{key}' must be >= 0, got {v}.")
    return v


@dataclass
class Profile:
    type: str = "step"
    start: float = 0.0
    rise_time: float = 0.0
    period: float = 10.0
    step_std: float = 0.3
    mean_on: float = 2.0
    mean_off: float = 3.0
    recovery_time: float | None = None
    recovery_duration: float = 0.0
    recovery_shape: str = "linear"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Profile":
        d = dict(d or {})
        typ = str(d.get("type", "step")).lower()
        if typ not in PROFILE_TYPES:
            raise ConfigurationError(f"Unknown profile type {typ!r}. Available: {', '.join(PROFILE_TYPES)}.")
        start = _num(d, "start", 0.0, nonneg=True)
        rise = _num(d, "rise_time", 0.0, nonneg=True)
        if typ in ("ramp", "exponential", "sigmoid") and rise <= 0:
            raise ConfigurationError(f"Profile '{typ}' requires rise_time > 0.")
        rec = _num(d, "recovery_time", None)
        dur = _num(d, "duration", None, nonneg=True)
        if rec is None and dur is not None:
            rec = start + dur
        if rec is not None and rec < start:
            raise ConfigurationError(f"recovery_time ({rec}) must be >= start ({start}).")
        if typ == "recovery" and rec is None:
            raise ConfigurationError("Profile 'recovery' requires recovery_time or duration.")
        shape = str(d.get("recovery_shape", "linear")).lower()
        if shape not in ("linear", "exponential"):
            raise ConfigurationError("recovery_shape must be 'linear' or 'exponential'.")
        rdur = _num(d, "recovery_duration", 0.0, nonneg=True)
        if shape == "exponential" and rec is not None and rdur <= 0:
            raise ConfigurationError("Exponential recovery requires recovery_duration > 0.")
        return cls(
            type=typ, start=start, rise_time=rise,
            period=_num(d, "period", 10.0, positive=True),
            step_std=_num(d, "step_std", 0.3, nonneg=True),
            mean_on=_num(d, "mean_on", 2.0, positive=True),
            mean_off=_num(d, "mean_off", 3.0, positive=True),
            recovery_time=rec, recovery_duration=rdur, recovery_shape=shape,
        )

    def to_dict(self) -> dict:
        d = {"type": self.type, "start": self.start}
        if self.type in ("ramp", "exponential", "sigmoid") or (self.type == "recovery" and self.rise_time):
            d["rise_time"] = self.rise_time
        if self.type == "sinusoidal":
            d["period"] = self.period
        if self.type == "random_walk":
            d["step_std"] = self.step_std
        if self.type == "intermittent":
            d["mean_on"], d["mean_off"] = self.mean_on, self.mean_off
        if self.recovery_time is not None:
            d["recovery_time"] = self.recovery_time
            d["recovery_duration"] = self.recovery_duration
            d["recovery_shape"] = self.recovery_shape
        return d

    # ------------------------------------------------------------------
    def onset(self, t: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        tau = t - self.start
        after = tau >= 0
        o = np.zeros_like(t, dtype=float)
        typ = self.type
        if typ == "recovery":
            typ = "ramp" if self.rise_time > 0 else "step"
        if typ == "step":
            o[after] = 1.0
        elif typ == "ramp":
            o[after] = np.minimum(1.0, tau[after] / self.rise_time)
        elif typ == "exponential":
            o[after] = 1.0 - np.exp(-tau[after] / self.rise_time)
        elif typ == "sigmoid":
            s = lambda x: 1.0 / (1.0 + np.exp(-10.0 * (x / self.rise_time - 0.5)))
            s0, s1 = s(0.0), s(self.rise_time)
            val = (s(tau[after]) - s0) / (s1 - s0)
            o[after] = np.clip(np.where(tau[after] >= self.rise_time, 1.0, val), 0.0, 1.0)
        elif typ == "sinusoidal":
            o[after] = 0.5 * (1.0 - np.cos(2 * np.pi * tau[after] / self.period))
        elif typ == "random_walk":
            rng = rng if rng is not None else np.random.default_rng(0)
            Ts = float(t[1] - t[0]) if t.size > 1 else 1.0
            e = rng.standard_normal(t.size)
            g = 0.0
            for k in np.flatnonzero(after):
                g = min(1.0, max(0.0, g + self.step_std * np.sqrt(Ts) * e[k]))
                o[k] = g
        elif typ == "intermittent":
            rng = rng if rng is not None else np.random.default_rng(0)
            Ts = float(t[1] - t[0]) if t.size > 1 else 1.0
            p_off = 1.0 - np.exp(-Ts / self.mean_on)
            p_on = 1.0 - np.exp(-Ts / self.mean_off)
            u = rng.random(t.size)
            state = True
            for k in np.flatnonzero(after):
                o[k] = 1.0 if state else 0.0
                state = (u[k] >= p_off) if state else (u[k] < p_on)
        return o

    def envelope(self, t: np.ndarray) -> np.ndarray:
        r = np.ones_like(t, dtype=float)
        if self.recovery_time is None:
            return r
        after = t >= self.recovery_time
        dt = t[after] - self.recovery_time
        if self.recovery_shape == "linear":
            if self.recovery_duration <= 0:
                r[after] = 0.0
            else:
                r[after] = np.clip(1.0 - dt / self.recovery_duration, 0.0, 1.0)
        else:
            r[after] = np.exp(-dt / self.recovery_duration)
        return r

    def evaluate(self, t: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Activation ``g(t)`` on the time grid ``t``."""
        t = np.asarray(t, dtype=float)
        return self.onset(t, rng) * self.envelope(t)

    def end_time(self) -> float | None:
        """Time at which the degradation has completely vanished (None = never)."""
        if self.recovery_time is None:
            return None
        if self.recovery_shape == "linear":
            return self.recovery_time + self.recovery_duration
        return None  # exponential recovery never reaches exactly zero
