"""Dynamic model interface.

A dynamic model describes the true evolution of the system

    x_{k+1} = f(x_k, u_k) + w_k,      w_k ~ N(0, Q)

and, for linear models, ``f(x, u) = A x + B u``.  Measurement functions
``h_i(x)`` belong to the sensors; a model may additionally publish *named*
nonlinear measurement functions (e.g. the horizontal position of a pendulum
bob) that sensors can refer to by name.

Models are used in two different roles that must not be confused:

* by the **ground-truth simulator** (the "world"), and
* by the **estimators** (the "belief" of the researcher).

Both use the same class, but the estimator only receives the nominal model
parameters written in the configuration; it never receives the true state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..utils.errors import ConfigurationError, DimensionError
from ..utils.linalg import as_matrix, check_covariance


@dataclass
class MeasurementFunction:
    """Named (possibly nonlinear) measurement function ``z = h(x)``."""

    name: str
    m: int
    h: Callable[[np.ndarray], np.ndarray]
    jacobian: Callable[[np.ndarray], np.ndarray]
    unit: str = ""
    description: str = ""
    linear: bool = False


# ----------------------------------------------------------------------------
# Deterministic input signals u(t)
# ----------------------------------------------------------------------------

def input_signal(spec: dict | None, t: np.ndarray) -> np.ndarray:
    """Evaluate a scalar deterministic input profile on the time grid ``t``.

    ``spec = {"type": "sine"|"square"|"constant"|"none"|"steps",
    "amplitude": A, "frequency": f [Hz], "offset": c, "phase": phi [rad],
    "values": [...], "times": [...]}``
    """
    t = np.asarray(t, dtype=float)
    if spec is None:
        return np.zeros_like(t)
    typ = str(spec.get("type", "sine")).lower()
    A = float(spec.get("amplitude", 1.0))
    f = float(spec.get("frequency", 0.1))
    c = float(spec.get("offset", 0.0))
    ph = float(spec.get("phase", 0.0))
    if typ == "none":
        return np.zeros_like(t)
    if typ == "constant":
        return np.full_like(t, c if "offset" in spec else A)
    if typ == "sine":
        return c + A * np.sin(2 * np.pi * f * t + ph)
    if typ == "cosine":
        return c + A * np.cos(2 * np.pi * f * t + ph)
    if typ == "square":
        duty = float(spec.get("duty", 0.5))
        frac = np.mod(f * t + ph / (2 * np.pi), 1.0)
        return c + A * (frac < duty)
    if typ == "steps":
        times = np.asarray(spec.get("times", [0.0]), dtype=float)
        values = np.asarray(spec.get("values", [A]), dtype=float)
        if times.size != values.size:
            raise ConfigurationError("input 'steps': 'times' and 'values' must have equal length.")
        idx = np.searchsorted(times, t, side="right") - 1
        out = np.where(idx >= 0, values[np.clip(idx, 0, None)], c)
        return out.astype(float)
    raise ConfigurationError(f"Unknown input signal type {typ!r}.")


# ----------------------------------------------------------------------------
# Base classes
# ----------------------------------------------------------------------------

class DynamicModel(ABC):
    """Abstract discrete-time dynamic model with sampling period ``Ts``."""

    #: registry key
    key: str = "abstract"
    #: human readable name
    label: str = "Abstract model"
    description: str = ""

    def __init__(self, Ts: float, params: dict | None = None):
        if not np.isfinite(Ts) or Ts <= 0:
            raise ConfigurationError(f"Sampling period Ts must be positive, got {Ts}.")
        self.Ts = float(Ts)
        self.params: dict = dict(params or {})
        self.state_names: list[str] = []
        self.state_units: list[str] = []
        self.input_names: list[str] = []
        self.default_x0: np.ndarray = np.zeros(0)
        self.Q: np.ndarray = np.zeros((0, 0))
        self.measurement_functions: dict[str, MeasurementFunction] = {}

    # -- dimensions ---------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.state_names)

    @property
    def nu(self) -> int:
        return len(self.input_names)

    # -- dynamics -----------------------------------------------------------
    @property
    @abstractmethod
    def is_linear(self) -> bool: ...

    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Deterministic one-step transition ``x_{k+1} = f(x_k, u_k)``."""

    def F(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Jacobian ``df/dx`` (central finite differences by default)."""
        x = np.asarray(x, dtype=float)
        n = x.size
        J = np.empty((n, n))
        for j in range(n):
            h = 1e-6 * max(1.0, abs(x[j]))
            e = np.zeros(n)
            e[j] = h
            J[:, j] = (self.f(x + e, u) - self.f(x - e, u)) / (2 * h)
        return J

    def inputs(self, t: np.ndarray) -> np.ndarray:
        """Deterministic input sequence (K x nu)."""
        return np.zeros((np.asarray(t).size, self.nu))

    # -- helpers ------------------------------------------------------------
    def selector(self, indices: list[int]) -> np.ndarray:
        """Linear measurement matrix selecting state components."""
        H = np.zeros((len(indices), self.n))
        for r, j in enumerate(indices):
            if not (0 <= int(j) < self.n):
                raise DimensionError(
                    f"Sensor measures state index {j}, but model '{self.key}' has {self.n} states "
                    f"({', '.join(self.state_names)})."
                )
            H[r, int(j)] = 1.0
        return H

    def state_index(self, name_or_index) -> int:
        if isinstance(name_or_index, (int, np.integer)):
            i = int(name_or_index)
        elif isinstance(name_or_index, str) and name_or_index in self.state_names:
            i = self.state_names.index(name_or_index)
        else:
            try:
                i = int(name_or_index)
            except (TypeError, ValueError):
                raise ConfigurationError(
                    f"Unknown state {name_or_index!r}; available: {self.state_names}.") from None
        if not 0 <= i < self.n:
            raise DimensionError(f"State index {i} out of range for model '{self.key}'.")
        return i

    def validate(self) -> None:
        if len(self.state_units) != self.n:
            raise DimensionError("state_units must match the number of states.")
        if self.Q.shape != (self.n, self.n):
            raise DimensionError(f"Process noise Q must be {self.n}x{self.n}, got {self.Q.shape}.")
        self.Q = check_covariance(self.Q, "process noise covariance Q")
        if self.default_x0.size != self.n:
            raise DimensionError("default initial state has wrong dimension.")

    def describe(self) -> dict:
        d = {
            "key": self.key,
            "label": self.label,
            "linear": self.is_linear,
            "Ts": self.Ts,
            "states": list(self.state_names),
            "units": list(self.state_units),
            "inputs": list(self.input_names),
            "params": self.params,
            "Q": self.Q.tolist(),
        }
        if self.is_linear:
            d["A"] = self.A.tolist()  # type: ignore[attr-defined]
            d["B"] = self.B.tolist()  # type: ignore[attr-defined]
        return d


class LinearModel(DynamicModel):
    """Linear time-invariant model ``x_{k+1} = A x_k + B u_k + w_k``."""

    def __init__(self, Ts: float, params: dict | None = None):
        super().__init__(Ts, params)
        self.A = np.zeros((0, 0))
        self.B = np.zeros((0, 0))

    @property
    def is_linear(self) -> bool:
        return True

    def f(self, x, u):
        if self.nu:
            return self.A @ x + self.B @ u
        return self.A @ x

    def F(self, x, u):
        return self.A

    def validate(self) -> None:
        super().validate()
        if self.A.shape != (self.n, self.n):
            raise DimensionError(f"A must be {self.n}x{self.n}, got {self.A.shape}.")
        if self.B.shape != (self.n, self.nu):
            raise DimensionError(f"B must be {self.n}x{self.nu}, got {self.B.shape}.")


# ----------------------------------------------------------------------------
# Discretisation helpers
# ----------------------------------------------------------------------------

def discretize(Ac: np.ndarray, Bc: np.ndarray, Qc: np.ndarray, Ts: float):
    """Exact zero-order-hold discretisation with Van Loan's method.

    Returns ``(A, B, Q)`` with ``A = e^{Ac Ts}``, ``B = int_0^Ts e^{Ac s} ds Bc``
    and ``Q = int_0^Ts e^{Ac s} Qc e^{Ac^T s} ds``.
    """
    from scipy.linalg import expm

    n = Ac.shape[0]
    nu = Bc.shape[1]
    M = np.zeros((n + nu, n + nu))
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    Md = expm(M * Ts)
    A = Md[:n, :n]
    B = Md[:n, n:]
    V = np.zeros((2 * n, 2 * n))
    V[:n, :n] = -Ac
    V[:n, n:] = Qc
    V[n:, n:] = Ac.T
    Vd = expm(V * Ts)
    Q = Vd[n:, n:].T @ Vd[:n, n:]
    return A, B, 0.5 * (Q + Q.T)


def white_noise_Q(order: int, q: float, Ts: float) -> np.ndarray:
    """Discrete process noise of a continuous white-noise kinematic model.

    ``order = 1``: random walk (Q = q Ts);
    ``order = 2``: white-noise acceleration (position/velocity);
    ``order = 3``: white-noise jerk (position/velocity/acceleration).
    ``q`` is the power spectral density of the driving noise.
    """
    T = Ts
    if order == 1:
        return np.array([[q * T]])
    if order == 2:
        return q * np.array([[T**3 / 3, T**2 / 2], [T**2 / 2, T]])
    if order == 3:
        return q * np.array([
            [T**5 / 20, T**4 / 8, T**3 / 6],
            [T**4 / 8, T**3 / 3, T**2 / 2],
            [T**3 / 6, T**2 / 2, T],
        ])
    raise ValueError("order must be 1, 2 or 3")


def param(params: dict, name: str, default: float, positive: bool = False,
          nonnegative: bool = False) -> float:
    try:
        v = float(params.get(name, default))
    except (TypeError, ValueError):
        raise ConfigurationError(f"Model parameter '{name}' must be numeric.") from None
    if not np.isfinite(v):
        raise ConfigurationError(f"Model parameter '{name}' must be finite.")
    if positive and v <= 0:
        raise ConfigurationError(f"Model parameter '{name}' must be > 0, got {v}.")
    if nonnegative and v < 0:
        raise ConfigurationError(f"Model parameter '{name}' must be >= 0, got {v}.")
    return v


def matrix_param(params: dict, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    if name not in params:
        raise ConfigurationError(f"Model parameter '{name}' is required.")
    M = np.atleast_2d(np.asarray(params[name], dtype=float))
    if shape is not None and M.shape != shape:
        raise DimensionError(f"Model parameter '{name}' must have shape {shape}, got {M.shape}.")
    return M


__all__ = [
    "DynamicModel",
    "LinearModel",
    "MeasurementFunction",
    "input_signal",
    "discretize",
    "white_noise_Q",
    "param",
    "matrix_param",
    "as_matrix",
]
