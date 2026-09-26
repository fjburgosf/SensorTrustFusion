r"""Online estimators of the first two moments of the innovation.

Given the standardised (whitened) innovation of sensor ``i``

.. math:: \varepsilon_{i,k} = L_{i,k}^{-1}\,\nu_{i,k},\qquad S_{i,k} = L_{i,k}L_{i,k}^T,

computed with the *nominal* covariance ``S_{i,k} = H_i P_{k|k-1} H_i^T + R_i``,
the nominal hypothesis ``H0`` (sensor behaves as modelled) implies
``E[eps] = 0`` and ``E[eps eps^T] = I_m``.  The trackers estimate:

* first moment      ``mu_hat(k)  ~ E[eps]``                      (vector, m)
* second moment     ``m2_hat(k)  ~ E[||eps||^2] / m``             (scalar, = 1 under H0)
* variance          ``s2_hat(k)  = m2_hat(k) - ||mu_hat(k)||^2/m`` (central moment, >= 0)
* RMS               ``sqrt(m2_hat(k))``
* effective sample size ``N_eff(k)`` of the estimator.

The second moment, the variance, the RMS and the energy are *different*
quantities: ``m2 = variance + mean^2``; RMS is ``sqrt(m2)``; the (windowed)
energy is ``sum ||nu||^2`` over the window, i.e. ``N * m * m2`` in whitened
units.  They coincide only for zero-mean innovations.

Estimator types:

========== ================================================= ================
type       recursion                                          N_eff
========== ================================================= ================
ewma       mu_k = lam mu_{k-1} + (1-lam) eps_k                 (1+lam)/(1-lam)
window     mu_k = (1/N) sum_{j=k-N+1}^{k} eps_j                n_k = min(k, N)
recursive  mu_k = mu_{k-1} + (eps_k - mu_{k-1}) / k            n_k = k
========== ================================================= ================

(the same recursion is applied to ``||eps||^2/m`` for ``m2``).  The EWMA is
initialised at the nominal values (``mu = 0``, ``m2 = 1``); window and
recursive estimators start empty.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..utils.errors import ConfigurationError

MOMENT_ESTIMATORS = ("ewma", "window", "recursive")


class MomentTracker:
    """Moment tracker of one sensor."""

    def __init__(self, m: int, estimator: str = "ewma", lam: float = 0.98, window: int = 100):
        estimator = str(estimator).lower()
        if estimator not in MOMENT_ESTIMATORS:
            raise ConfigurationError(f"Unknown moment estimator {estimator!r}; use one of {MOMENT_ESTIMATORS}.")
        if estimator == "ewma" and not (0.0 < lam < 1.0):
            raise ConfigurationError("EWMA factor 'lam' must be in (0, 1).")
        if estimator == "window" and int(window) < 2:
            raise ConfigurationError("Moment window must contain at least 2 samples.")
        self.m = m
        self.kind = estimator
        self.lam = float(lam)
        self.N = int(window)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        if self.kind == "ewma":
            self.mu = np.zeros(self.m)
            self.m2 = 1.0
        else:
            self.mu = np.zeros(self.m)
            self.m2 = 0.0
        if self.kind == "window":
            self._buf_e: deque = deque(maxlen=self.N)
            self._buf_q: deque = deque(maxlen=self.N)
            self._sum_e = np.zeros(self.m)
            self._sum_q = 0.0

    @property
    def n_eff(self) -> float:
        if self.kind == "ewma":
            return (1.0 + self.lam) / (1.0 - self.lam)
        return float(max(self.n, 1))

    @property
    def ready(self) -> bool:
        return self.kind == "ewma" or self.n >= 2

    def update(self, eps: np.ndarray) -> None:
        q = float(eps @ eps) / self.m
        self.n += 1
        if self.kind == "ewma":
            a = 1.0 - self.lam
            self.mu = self.mu + a * (eps - self.mu)
            self.m2 = self.m2 + a * (q - self.m2)
        elif self.kind == "recursive":
            self.mu = self.mu + (eps - self.mu) / self.n
            self.m2 = self.m2 + (q - self.m2) / self.n
        else:
            if len(self._buf_e) == self.N:
                self._sum_e -= self._buf_e[0]
                self._sum_q -= self._buf_q[0]
            self._buf_e.append(eps.copy())
            self._buf_q.append(q)
            self._sum_e += eps
            self._sum_q += q
            L = len(self._buf_e)
            self.mu = self._sum_e / L
            self.m2 = self._sum_q / L

    @property
    def mean_norm(self) -> float:
        """``||mu|| / sqrt(m)`` (equals ``|mu|`` for scalar sensors)."""
        return float(np.sqrt(self.mu @ self.mu / self.m))

    @property
    def variance(self) -> float:
        return max(0.0, self.m2 - float(self.mu @ self.mu) / self.m)

    @property
    def signed_mean(self) -> float:
        """Signed mean for scalar sensors, ``||mu||/sqrt(m)`` otherwise."""
        return float(self.mu[0]) if self.m == 1 else self.mean_norm
