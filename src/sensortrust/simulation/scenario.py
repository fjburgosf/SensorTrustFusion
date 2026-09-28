"""Scenario datasets.

A :class:`Scenario` is the complete, frozen outcome of the virtual
experiment *before* any estimation: ground truth, inputs, measurements of all
sensors, noise and fault ground truth.  Fair comparisons are guaranteed by
construction: the experiment manager generates one scenario per seed and
executes every fusion method on exactly that scenario::

    Scenario Seed = 42
    Algorithm A -> scenario_42
    Algorithm B -> scenario_42

Estimators only receive the :class:`MeasurementSet` view (no ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..experiments.config import normalize_config
from ..models import create_model
from ..models.base import DynamicModel
from ..sensors.sensor import SensorData, SensorSpec, VirtualSensor
from ..utils.errors import ConfigurationError
from ..utils.linalg import as_matrix, check_covariance
from ..utils.rng import array_hash
from .ground_truth import GroundTruth, GroundTruthSimulator


@dataclass
class MeasurementSet:
    """Everything an estimator is allowed to see."""

    t: np.ndarray
    Ts: float
    z: list[np.ndarray]            # per sensor (K, m_i), NaN when unavailable
    u: np.ndarray                  # known deterministic inputs (K, nu)
    specs: list[SensorSpec]
    x0: np.ndarray                 # prior mean of the initial state
    P0: np.ndarray                 # prior covariance

    @property
    def K(self) -> int:
        return self.t.size

    @property
    def N(self) -> int:
        return len(self.specs)


@dataclass
class Scenario:
    config: dict
    seed: int
    model: DynamicModel
    truth: GroundTruth
    sensors: list[VirtualSensor]
    data: list[SensorData]
    x0_prior: np.ndarray
    P0_prior: np.ndarray
    include_degradations: bool = True
    _hash: str | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    @property
    def t(self) -> np.ndarray:
        return self.truth.t

    @property
    def K(self) -> int:
        return self.truth.t.size

    @property
    def N(self) -> int:
        return len(self.sensors)

    @property
    def specs(self) -> list[SensorSpec]:
        return [s.spec() for s in self.sensors]

    def measurements(self) -> MeasurementSet:
        return MeasurementSet(
            t=self.t.copy(), Ts=self.model.Ts, z=[d.z.copy() for d in self.data], u=self.truth.u.copy(),
            specs=self.specs, x0=self.x0_prior.copy(), P0=self.P0_prior.copy(),
        )

    def fault_windows(self) -> np.ndarray:
        """(K,) bool: at least one sensor is degraded."""
        out = np.zeros(self.K, dtype=bool)
        for d in self.data:
            out |= d.fault_active
        return out

    def fault_meta(self) -> list[dict]:
        return [m for d in self.data for m in d.fault_meta]

    def fingerprint(self) -> str:
        """SHA-256 of ground truth + all measurements (reproducibility check)."""
        if self._hash is None:
            self._hash = array_hash([self.truth.x] + [d.z for d in self.data])
        return self._hash

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Tabular dataset: time, inputs, ground truth, measurements, fault ground truth."""
        cols: dict[str, np.ndarray] = {"time": self.t}
        for j, nm in enumerate(self.model.state_names):
            cols[f"true_{nm}"] = self.truth.x[:, j]
        for j, nm in enumerate(self.model.input_names):
            cols[f"input_{nm}"] = self.truth.u[:, j]
        for s, d in zip(self.sensors, self.data):
            for r in range(s.m):
                suf = "" if s.m == 1 else f"_{r + 1}"
                cols[f"{s.name}{suf}"] = d.z[:, r]
        for s, d in zip(self.sensors, self.data):
            cols[f"fault_{s.name}"] = d.fault_active.astype(int)
        for s, d in zip(self.sensors, self.data):
            for r in range(s.m):
                suf = "" if s.m == 1 else f"_{r + 1}"
                cols[f"truebias_{s.name}{suf}"] = d.true_bias[:, r]
                cols[f"truevar_{s.name}{suf}"] = d.true_var[:, r]
                cols[f"clean_{s.name}{suf}"] = d.clean[:, r]
        return pd.DataFrame(cols)

    def metadata(self) -> dict:
        return {
            "seed": self.seed,
            "fingerprint": self.fingerprint(),
            "include_degradations": self.include_degradations,
            "K": self.K,
            "Ts": self.model.Ts,
            "model": self.model.describe(),
            "true_initial_state": self.truth.x0.tolist(),
            "prior_x0": self.x0_prior.tolist(),
            "prior_P0": self.P0_prior.tolist(),
            "sensors": [s.describe() for s in self.sensors],
            "faults": self.fault_meta(),
        }


def build_model(cfg: dict) -> DynamicModel:
    fs = float(cfg["simulation"]["fs"])
    return create_model(cfg["model"]["type"], 1.0 / fs, cfg["model"].get("params", {}))


def _prior(cfg: dict, model: DynamicModel):
    mcfg = cfg["model"]
    x0 = np.asarray(mcfg["x0"] if mcfg.get("x0") is not None else model.default_x0, dtype=float)
    if x0.size != model.n:
        raise ConfigurationError(f"model.x0 must have {model.n} entries ({', '.join(model.state_names)}).")
    P0 = check_covariance(as_matrix(mcfg.get("P0", 0.01), model.n, "model.P0"), "model.P0")
    ecfg = cfg.get("estimation", {}) or {}
    xe = np.asarray(ecfg["x0"], dtype=float) if ecfg.get("x0") is not None else x0.copy()
    if xe.size != model.n:
        raise ConfigurationError(f"estimation.x0 must have {model.n} entries.")
    Pe = P0.copy() if ecfg.get("P0") is None else as_matrix(ecfg["P0"], model.n, "estimation.P0")
    Pe = check_covariance(Pe, "estimation.P0", allow_psd=False) if np.any(Pe) else np.eye(model.n) * 1e-6
    return x0, P0, xe, Pe


def generate_scenario(config: dict, seed: int | None = None, include_degradations: bool = True,
                      normalized: bool = False) -> Scenario:
    """Generate the scenario dataset of a configuration.

    ``seed`` overrides ``config.experiment.seed``; ``include_degradations =
    False`` produces the *nominal twin*: identical noise realisations and
    ground truth, without degradations.
    """
    cfg = config if normalized else normalize_config(config)
    seed = int(cfg["experiment"]["seed"] if seed is None else seed)
    model = build_model(cfg)
    x0, P0, xe, Pe = _prior(cfg, model)
    sim = GroundTruthSimulator(model, cfg["simulation"]["duration"], cfg["simulation"]["fs"])
    truth = sim.run(x0, P0, seed, bool(cfg["model"].get("sample_initial_state", True)))
    fs = float(cfg["simulation"]["fs"])
    sensors = [VirtualSensor(s, i, model, fs) for i, s in enumerate(cfg["sensors"])]
    data = [s.generate(truth.t, truth.x, seed, include_degradations) for s in sensors]
    return Scenario(config=cfg, seed=seed, model=model, truth=truth, sensors=sensors, data=data,
                    x0_prior=xe, P0_prior=Pe, include_degradations=include_degradations)
