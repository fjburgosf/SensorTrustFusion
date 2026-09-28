r"""Ground-truth generator.

Generates the true trajectory

.. math:: x_{k+1} = f(x_k, u_k) + w_k,\qquad w_k \sim \mathcal N(0, Q),

for ``k = 0 .. K-1`` on the grid ``t_k = k T_s``.  If
``sample_initial_state`` is true, ``x_0 ~ N(x0, P0)``; otherwise ``x_0 = x0``.
The true trajectory is only used by the experimental environment (sensor
generation and evaluation); estimators never receive it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.base import DynamicModel
from ..utils import rng as rngmod
from ..utils.errors import ConfigurationError


def cov_sqrt(C: np.ndarray) -> np.ndarray:
    """Square root ``L`` with ``L L^T = C`` for PSD (possibly singular) ``C``."""
    try:
        return np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(0.5 * (C + C.T))
        return V * np.sqrt(np.clip(w, 0.0, None))


@dataclass
class GroundTruth:
    t: np.ndarray        # (K,)
    x: np.ndarray        # (K, n)
    u: np.ndarray        # (K, nu)
    x0: np.ndarray       # true initial state
    w: np.ndarray        # (K, n) process noise realisation


class GroundTruthSimulator:
    def __init__(self, model: DynamicModel, duration: float, fs: float):
        self.model = model
        self.K = int(round(duration * fs))
        self.t = np.arange(self.K) / fs
        if abs(1.0 / fs - model.Ts) > 1e-12 * max(1.0, model.Ts):
            raise ConfigurationError("Model sampling period differs from 1/fs.")

    def run(self, x0_mean: np.ndarray, P0: np.ndarray, seed: int, sample_initial_state: bool = True) -> GroundTruth:
        model = self.model
        n = model.n
        x0_mean = np.asarray(x0_mean, dtype=float).reshape(n)
        if sample_initial_state:
            x0 = x0_mean + cov_sqrt(P0) @ rngmod.stream(seed, "truth", "x0").standard_normal(n)
        else:
            x0 = x0_mean.copy()
        u = model.inputs(self.t).reshape(self.K, model.nu)
        Lq = cov_sqrt(model.Q)
        w = rngmod.stream(seed, "truth", "process").standard_normal((self.K, n)) @ Lq.T
        x = np.empty((self.K, n))
        x[0] = x0
        if model.is_linear:
            A, B = model.A, model.B  # type: ignore[attr-defined]
            for k in range(self.K - 1):
                x[k + 1] = A @ x[k] + (B @ u[k] if model.nu else 0.0) + w[k]
        else:
            for k in range(self.K - 1):
                x[k + 1] = model.f(x[k], u[k]) + w[k]
        if not np.all(np.isfinite(x)):
            raise ConfigurationError("The ground-truth trajectory diverged (non-finite values). "
                                     "Check the model parameters.")
        return GroundTruth(t=self.t.copy(), x=x, u=u, x0=x0, w=w)
