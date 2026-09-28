r"""Virtual sensors.

Measurement pipeline of sensor ``i`` at base step ``k`` (delay ``d`` steps):

.. math::

    c_k   &= h_i(x_{k-d})                                   \quad\text{(clean measurement)}\\
    y_k   &= \alpha_k \left(c_k + b_i^{nom} + b_k^{deg} + v_k + v_k^{deg}\right)\\
    z_k   &= \mathrm{stuck}\big(\mathrm{post}^{deg}(Q_{\Delta}(\mathrm{sat}_{[l,u]}(y_k)))\big)

followed by the availability mask (sampling rate, nominal dropout,
degradation-induced loss): unavailable samples are ``NaN``.

* ``alpha_k``  product of multiplicative degradations (sensitivity loss);
* ``b^nom``    uncompensated nominal bias (unknown to the estimators);
* ``b^deg``    additive degradations (bias, drift, ...);
* ``v``        nominal noise (see :mod:`sensortrust.sensors.noise`);
* ``v^deg``    degradation noise (variance increase, bursts, outliers);
* ``sat``      nominal range, ``Q_Delta`` nominal resolution.

Ground truth stored per sensor: clean measurement, total systematic error
``E[z|x] - c`` (= ``alpha (c + b) - c``), true noise variance
``alpha^2 (Var v + Var v^deg)``, availability, fault masks and metadata.

:class:`SensorSpec` is the *nominal* description handed to estimators
(measurement function, nominal ``R``, rate); it never contains degradation
information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..degradation.faults import Degradation, create_degradation, expand_degradations
from ..models.base import DynamicModel, MeasurementFunction
from ..utils import rng as rngmod
from ..utils.errors import ConfigurationError, DimensionError
from .noise import NoiseModel, create_noise


@dataclass
class SensorSpec:
    """Nominal sensor description available to estimators (no ground truth)."""

    name: str
    index: int
    m: int
    h: Callable[[np.ndarray], np.ndarray]
    jacobian: Callable[[np.ndarray], np.ndarray]
    linear: bool
    H: np.ndarray | None
    R: np.ndarray
    rate: float
    period_steps: int
    nominal_reliability: float
    unit: str
    measured: str
    target_index: int | None  # state index if the sensor directly measures one state

    def describe(self) -> dict:
        return {
            "name": self.name, "index": self.index, "m": self.m, "linear": self.linear,
            "H": None if self.H is None else self.H.tolist(), "R_nominal": self.R.tolist(),
            "rate_hz": self.rate, "nominal_reliability": self.nominal_reliability,
            "unit": self.unit, "measured": self.measured,
        }


@dataclass
class SensorData:
    """Generated data of one sensor (includes ground truth for evaluation)."""

    z: np.ndarray                  # (K, m) measurements, NaN when unavailable
    clean: np.ndarray              # (K, m) h(x) without any error
    true_bias: np.ndarray          # (K, m) systematic error E[z|x] - clean
    true_var: np.ndarray           # (K, m) true noise variance
    noise: np.ndarray              # (K, m) total random error realisation (before post-processing)
    available: np.ndarray          # (K,) bool
    fault_active: np.ndarray       # (K,) bool, union over degradations
    fault_masks: list[np.ndarray] = field(default_factory=list)   # per degradation
    fault_meta: list[dict] = field(default_factory=list)
    activation: list[np.ndarray] = field(default_factory=list)    # g_k per degradation


class VirtualSensor:
    """Virtual sensor built from a configuration mapping.

    Recognised keys: ``name``, ``measures`` (state index/name or list),
    ``H`` (explicit matrix), ``function`` (named model measurement function),
    ``rate`` [Hz], ``noise`` (mapping or std), ``assumed_std`` (std believed by
    estimators), ``bias`` (nominal uncompensated bias), ``resolution``,
    ``range`` ``[low, high]``, ``delay`` [s], ``dropout`` (nominal probability),
    ``nominal_reliability`` in (0, 1], ``degradations`` (list).
    """

    def __init__(self, cfg: dict, index: int, model: DynamicModel, fs: float):
        if not isinstance(cfg, dict):
            raise ConfigurationError(f"Sensor #{index + 1} configuration must be a mapping.")
        self.cfg = dict(cfg)
        self.index = index
        self.name = str(cfg.get("name", f"S{index + 1}"))
        self.model = model
        self._build_measurement(model)
        self.noise: NoiseModel = create_noise(cfg.get("noise", {"type": "gaussian", "std": 0.1}), self.m)
        # rate
        rate = float(cfg.get("rate", fs))
        if not np.isfinite(rate) or rate <= 0:
            raise ConfigurationError(f"Sensor '{self.name}': rate must be > 0 Hz.")
        if rate > fs + 1e-9:
            raise ConfigurationError(
                f"Sensor '{self.name}': rate {rate} Hz exceeds the simulation sampling frequency {fs} Hz.")
        self.period_steps = max(1, int(round(fs / rate)))
        self.rate = fs / self.period_steps
        self.bias = self._vec(cfg.get("bias", 0.0), "bias")
        res = cfg.get("resolution")
        self.resolution = None if res in (None, 0, 0.0) else float(res)
        if self.resolution is not None and self.resolution <= 0:
            raise ConfigurationError(f"Sensor '{self.name}': resolution must be > 0.")
        rng_ = cfg.get("range")
        if rng_ is not None:
            if len(rng_) != 2 or float(rng_[0]) >= float(rng_[1]):
                raise ConfigurationError(f"Sensor '{self.name}': range must be [low, high] with low < high.")
            self.range = (float(rng_[0]), float(rng_[1]))
        else:
            self.range = None
        delay = float(cfg.get("delay", 0.0))
        if delay < 0:
            raise ConfigurationError(f"Sensor '{self.name}': delay must be >= 0.")
        self.delay_steps = int(round(delay * fs))
        self.dropout = float(cfg.get("dropout", 0.0))
        if not 0.0 <= self.dropout <= 1.0:
            raise ConfigurationError(f"Sensor '{self.name}': dropout probability must be in [0, 1].")
        rel = float(cfg.get("nominal_reliability", 1.0))
        if not 0.0 < rel <= 1.0:
            raise ConfigurationError(f"Sensor '{self.name}': nominal_reliability must be in (0, 1].")
        self.nominal_reliability = rel
        assumed = cfg.get("assumed_std")
        std = self.noise.nominal_std if assumed is None else self._vec(assumed, "assumed_std")
        self.R_nominal = np.diag(np.maximum(std, 1e-12) ** 2)
        self.degradation_specs = expand_degradations(cfg.get("degradations", []))
        self.degradations: list[Degradation] = [create_degradation(d) for d in self.degradation_specs]

    # ------------------------------------------------------------------
    def _vec(self, v, name):
        a = np.atleast_1d(np.asarray(v, dtype=float))
        if a.size == 1:
            a = np.full(self.m, float(a[0]))
        if a.size != self.m:
            raise DimensionError(f"Sensor '{self.name}': '{name}' must have 1 or {self.m} entries.")
        return a

    def _build_measurement(self, model: DynamicModel):
        cfg = self.cfg
        self.target_index = None
        if "function" in cfg and cfg["function"] is not None:
            fname = str(cfg["function"])
            mf: MeasurementFunction | None = model.measurement_functions.get(fname)
            if mf is None:
                raise ConfigurationError(
                    f"Sensor '{self.name}': model '{model.key}' has no measurement function '{fname}'. "
                    f"Available: {', '.join(model.measurement_functions) or 'none'}.")
            self.m = mf.m
            self.h, self.jac, self.linear = mf.h, mf.jacobian, mf.linear
            self.H = mf.jacobian(np.zeros(model.n)) if mf.linear else None
            self.unit = mf.unit
            self.measured = fname
            if mf.linear and self.H is not None and self.H.shape[0] == 1 and np.count_nonzero(self.H) == 1 \
                    and np.isclose(self.H.max(), 1.0):
                self.target_index = int(np.argmax(self.H[0]))
            return
        if "H" in cfg and cfg["H"] is not None:
            H = np.atleast_2d(np.asarray(cfg["H"], dtype=float))
            if H.shape[1] != model.n:
                raise DimensionError(
                    f"Sensor '{self.name}': H has {H.shape[1]} columns but the model has {model.n} states.")
            measured = "H x"
        else:
            meas = cfg.get("measures", 0)
            idx = meas if isinstance(meas, (list, tuple)) else [meas]
            idx = [model.state_index(j) for j in idx]
            H = model.selector(idx)
            measured = ", ".join(model.state_names[j] for j in idx)
        self.H = H
        self.m = H.shape[0]
        self.linear = True
        self.h = lambda x, H=H: H @ x
        self.jac = lambda x, H=H: H
        rows_unit = [model.state_units[int(np.argmax(np.abs(r)))] for r in H]
        self.unit = rows_unit[0] if len(set(rows_unit)) == 1 else "mixed"
        self.measured = measured
        if self.m == 1 and np.count_nonzero(H) == 1 and np.isclose(H.max(), 1.0):
            self.target_index = int(np.argmax(H[0]))

    # ------------------------------------------------------------------
    def spec(self) -> SensorSpec:
        return SensorSpec(
            name=self.name, index=self.index, m=self.m, h=self.h, jacobian=self.jac, linear=self.linear,
            H=None if self.H is None else self.H.copy(), R=self.R_nominal.copy(), rate=self.rate,
            period_steps=self.period_steps, nominal_reliability=self.nominal_reliability, unit=self.unit,
            measured=self.measured, target_index=self.target_index,
        )

    def generate(self, t: np.ndarray, x_true: np.ndarray, seed: int, include_degradations: bool = True) -> SensorData:
        """Generate the measurement sequence of this sensor.

        Random streams are keyed by sensor index and stage, so they are
        independent of each other and of the presence of degradations.
        """
        K = t.size
        m = self.m
        kd = np.clip(np.arange(K) - self.delay_steps, 0, K - 1)
        xs = x_true[kd]
        if self.linear and self.H is not None:
            clean = xs @ self.H.T
        else:
            clean = np.array([self.h(x) for x in xs]).reshape(K, m)
        v, var_nom = self.noise.generate(t, clean, rngmod.stream(seed, "sensor", self.index, "noise"))

        offset = np.repeat(self.bias[None, :], K, axis=0)
        v_deg = np.zeros((K, m))
        var_deg = np.zeros((K, m))
        scale = np.ones(K)
        lost = np.zeros(K, dtype=bool)
        masks, metas, acts = [], [], []
        degs = self.degradations if include_degradations else []
        gs = []
        for j, d in enumerate(degs):
            r = rngmod.stream(seed, "sensor", self.index, "degradation", j)
            g = d.profile.evaluate(t, rngmod.stream(seed, "sensor", self.index, "profile", j))
            gs.append(g)
            offset = offset + d.offset(t, g, r, m)
            vn, vv = d.extra_noise(t, g, r, m, var_nom)
            v_deg += vn
            var_deg += vv
            scale = scale * d.scale(t, g)
            lost |= d.lost(t, g, r)
        for d, g in zip(degs, gs):
            mask = d.active(t, g)
            masks.append(mask)
            acts.append(g)
            idx = np.flatnonzero(mask)
            meta = d.describe()
            meta.update({
                "sensor": self.name,
                "onset_time": float(d.profile.start),
                "first_active_time": float(t[idx[0]]) if idx.size else None,
                "last_active_time": float(t[idx[-1]]) if idx.size else None,
                "recovery_time": d.profile.recovery_time,
                "end_time": d.profile.end_time(),
            })
            metas.append(meta)

        y = scale[:, None] * (clean + offset + v + v_deg)
        true_bias = scale[:, None] * (clean + offset) - clean
        true_var = (scale**2)[:, None] * (var_nom + var_deg)
        z = y
        if self.range is not None:
            z = np.clip(z, self.range[0], self.range[1])
        if self.resolution is not None:
            z = self.resolution * np.round(z / self.resolution)
        stuck = [(d, g) for d, g in zip(degs, gs) if d.type_name == "stuck"]
        for d, g in zip(degs, gs):
            if d.type_name != "stuck":
                z = d.post(z, t, g, None)
        for d, g in stuck:
            z = d.post(z, t, g, None)

        available = (np.arange(K) % self.period_steps) == 0
        if self.dropout > 0:
            available &= rngmod.stream(seed, "sensor", self.index, "dropout").random(K) >= self.dropout
        available &= ~lost
        z = np.where(available[:, None], z, np.nan)
        fault_active = np.zeros(K, dtype=bool)
        for mk in masks:
            fault_active |= mk
        return SensorData(z=z, clean=clean, true_bias=true_bias, true_var=true_var, noise=v + v_deg,
                          available=available, fault_active=fault_active, fault_masks=masks,
                          fault_meta=metas, activation=acts)

    def describe(self) -> dict:
        d = self.spec().describe()
        d.update({
            "noise": self.noise.describe(), "nominal_bias": self.bias.tolist(),
            "resolution": self.resolution, "range": self.range,
            "delay_steps": self.delay_steps, "dropout": self.dropout,
            "degradations": [deg.describe() for deg in self.degradations],
        })
        return d
