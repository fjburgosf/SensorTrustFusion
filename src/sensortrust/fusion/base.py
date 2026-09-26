"""Fusion method interface and result container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..models.base import DynamicModel
from ..simulation.scenario import MeasurementSet


@dataclass
class OracleInfo:
    """Ground-truth information handed **only** to methods flagged ``is_oracle``.

    Oracle methods are offline references (ideal knowledge) and are never
    presented as realizable algorithms.
    """

    x_true: np.ndarray
    true_var: list[np.ndarray]
    true_bias: list[np.ndarray]


@dataclass
class MethodResult:
    key: str
    label: str
    category: str
    is_oracle: bool
    description: str
    params: dict
    x: np.ndarray                               # (K, n); NaN for states not estimated
    estimated_states: list[int]
    P: np.ndarray | None = None                 # (K, n, n)
    innovation: list[np.ndarray] | None = None  # per sensor (K, m_i), raw nu at the prior
    innovation_std: list[np.ndarray] | None = None  # per sensor (K, m_i), sqrt(diag S_nominal)
    eps: list[np.ndarray] | None = None         # per sensor (K, m_i), standardised innovation
    nis_sensor: np.ndarray | None = None        # (K, N) eps^T eps (nominal S)
    nis: np.ndarray | None = None               # (K,) stacked NIS with the effective S
    nis_dof: np.ndarray | None = None           # (K,)
    trust: np.ndarray | None = None             # (K, N)
    weights: np.ndarray | None = None           # (K, N)
    alarms: np.ndarray | None = None            # (K, N) bool
    moments: dict[str, np.ndarray] | None = None  # (K, N) each
    detector_stats: dict[str, np.ndarray] | None = None
    r_eff: np.ndarray | None = None             # (K, N) mean diagonal of R_eff
    r_hat: np.ndarray | None = None             # (K, N) online estimate (adaptive)
    runtime_s: float = 0.0
    diverged: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def provides(self) -> list[str]:
        out = ["estimate"]
        for name in ("P", "innovation", "nis", "trust", "weights", "alarms", "moments", "r_eff", "r_hat"):
            if getattr(self, name) is not None:
                out.append(name)
        return out


class FusionMethod(ABC):
    key = "abstract"
    label = "Abstract fusion method"
    category = "baseline"   # baseline | filter | adaptive | trust | estimate_level | oracle
    is_oracle = False
    description = ""

    def __init__(self, label: str | None = None, **params):
        if label:
            self.label = label
        self.params = dict(params)

    @abstractmethod
    def run(self, meas: MeasurementSet, model: DynamicModel, oracle: OracleInfo | None = None) -> MethodResult:
        """Estimate the state from the measurements only (oracle info only if ``is_oracle``)."""

    def describe(self) -> dict:
        return {"key": self.key, "label": self.label, "category": self.category, "oracle": self.is_oracle,
                "params": self.params}


def available_mask(z: np.ndarray) -> np.ndarray:
    return ~np.any(np.isnan(z), axis=1)
