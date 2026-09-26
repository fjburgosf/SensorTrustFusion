"""Built-in benchmark dynamic models.

=====================  ===================================================
key                    description
=====================  ===================================================
scalar_signal          Model 1 - variable scalar signal (driven random walk)
constant_velocity      Model 2 - position / velocity, white-noise acceleration
constant_acceleration  Model 3 - position / velocity / acceleration
mass_spring_damper     Model 4 - forced damped oscillator (ZOH, Van Loan)
thermal                Model 5 - two-node lumped thermal system
pendulum               Model 6 - nonlinear damped pendulum (RK4)
custom_linear          user-defined linear model from matrices
=====================  ===================================================
"""

from __future__ import annotations

import numpy as np

from ..utils.errors import ConfigurationError
from ..utils.linalg import as_matrix
from .base import (
    DynamicModel,
    LinearModel,
    MeasurementFunction,
    discretize,
    input_signal,
    matrix_param,
    param,
    white_noise_Q,
)


class ScalarSignalModel(LinearModel):
    r"""Model 1 - variable scalar signal.

    ``x_{k+1} = x_k + Ts * u_k + w_k`` with the deterministic rate input
    ``u(t) = 2 pi f A cos(2 pi f t)`` so that, without process noise, the
    signal is ``x(t) = x_0 + A sin(2 pi f t)``.  ``Q = q Ts``.
    """

    key = "scalar_signal"
    label = "Scalar variable signal"
    description = "Scalar signal x(t) = x0 + A sin(2*pi*f*t) perturbed by a random walk."

    def __init__(self, Ts, params=None):
        super().__init__(Ts, params)
        self.amplitude = param(self.params, "amplitude", 1.0, nonnegative=True)
        self.frequency = param(self.params, "frequency", 0.1, nonnegative=True)
        q = param(self.params, "q", 1e-3, nonnegative=True)
        unit = str(self.params.get("unit", "a.u."))
        self.state_names = ["signal"]
        self.state_units = [unit]
        self.input_names = ["rate"]
        self.A = np.eye(1)
        self.B = np.array([[Ts]])
        self.Q = white_noise_Q(1, q, Ts)
        self.default_x0 = np.zeros(1)
        self.validate()

    def inputs(self, t):
        w = 2 * np.pi * self.frequency
        return (self.amplitude * w * np.cos(w * np.asarray(t)))[:, None]


class ConstantVelocityModel(LinearModel):
    """Model 2 - constant velocity: ``x = [p, v]``, white-noise acceleration of PSD ``q``."""

    key = "constant_velocity"
    label = "Constant velocity"
    description = "Kinematic model x=[p, v]; acceleration is continuous white noise of PSD q."

    def __init__(self, Ts, params=None):
        super().__init__(Ts, params)
        q = param(self.params, "q", 0.05, nonnegative=True)
        self.state_names = ["position", "velocity"]
        self.state_units = ["m", "m/s"]
        self.input_names = []
        self.A = np.array([[1.0, Ts], [0.0, 1.0]])
        self.B = np.zeros((2, 0))
        self.Q = white_noise_Q(2, q, Ts)
        self.default_x0 = np.array([0.0, 1.0])
        self.validate()


class ConstantAccelerationModel(LinearModel):
    """Model 3 - constant acceleration: ``x = [p, v, a]``, white-noise jerk of PSD ``q``."""

    key = "constant_acceleration"
    label = "Constant acceleration"
    description = "Kinematic model x=[p, v, a]; jerk is continuous white noise of PSD q."

    def __init__(self, Ts, params=None):
        super().__init__(Ts, params)
        q = param(self.params, "q", 0.01, nonnegative=True)
        self.state_names = ["position", "velocity", "acceleration"]
        self.state_units = ["m", "m/s", "m/s^2"]
        self.input_names = []
        self.A = np.array([[1.0, Ts, 0.5 * Ts**2], [0.0, 1.0, Ts], [0.0, 0.0, 1.0]])
        self.B = np.zeros((3, 0))
        self.Q = white_noise_Q(3, q, Ts)
        self.default_x0 = np.array([0.0, 1.0, 0.0])
        self.validate()


class MassSpringDamperModel(LinearModel):
    r"""Model 4 - forced mass-spring-damper.

    ``m p'' + c p' + k p = F(t) + w(t)``, state ``x = [p, v]``; exact ZOH
    discretisation (Van Loan) of the continuous model.  ``q`` is the PSD of
    the random force.
    """

    key = "mass_spring_damper"
    label = "Mass-spring-damper oscillator"
    description = "Forced damped oscillator m p'' + c p' + k p = F(t) + w(t)."

    def __init__(self, Ts, params=None):
        super().__init__(Ts, params)
        m = param(self.params, "mass", 1.0, positive=True)
        k = param(self.params, "stiffness", 4.0, nonnegative=True)
        c = param(self.params, "damping", 0.4, nonnegative=True)
        q = param(self.params, "q", 0.01, nonnegative=True)
        self.force = dict(self.params.get("force", {"type": "sine", "amplitude": 1.0, "frequency": 0.2}))
        Ac = np.array([[0.0, 1.0], [-k / m, -c / m]])
        Bc = np.array([[0.0], [1.0 / m]])
        Qc = np.array([[0.0, 0.0], [0.0, q / m**2]])
        self.A, self.B, self.Q = discretize(Ac, Bc, Qc, Ts)
        self.state_names = ["position", "velocity"]
        self.state_units = ["m", "m/s"]
        self.input_names = ["force"]
        self.default_x0 = np.array([0.5, 0.0])
        self.validate()

    def inputs(self, t):
        return input_signal(self.force, t)[:, None]


class ThermalModel(LinearModel):
    r"""Model 5 - simplified two-node thermal system.

    Heater node ``T_h`` (capacity ``C_h``) connected through ``R_ho`` to an
    object node ``T_o`` (capacity ``C_o``) that exchanges heat with ambient
    through ``R_oa``:

        C_h T_h' = P(t) - (T_h - T_o)/R_ho
        C_o T_o' = (T_h - T_o)/R_ho - (T_o - T_amb)/R_oa

    Inputs ``u = [P, T_amb]``; exact ZOH discretisation.
    """

    key = "thermal"
    label = "Simplified thermal system"
    description = "Two-node lumped thermal model (heater + object) with square-wave heating power."

    def __init__(self, Ts, params=None):
        super().__init__(Ts, params)
        Ch = param(self.params, "C_heater", 5.0, positive=True)
        Co = param(self.params, "C_object", 20.0, positive=True)
        Rho = param(self.params, "R_heater_object", 0.5, positive=True)
        Roa = param(self.params, "R_object_ambient", 1.0, positive=True)
        self.T_amb = param(self.params, "T_ambient", 25.0)
        q = param(self.params, "q", 0.01, nonnegative=True)
        self.power = dict(self.params.get("power", {"type": "square", "amplitude": 20.0,
                                                     "frequency": 1 / 60.0, "duty": 0.5}))
        Ac = np.array([[-1 / (Ch * Rho), 1 / (Ch * Rho)],
                       [1 / (Co * Rho), -1 / (Co * Rho) - 1 / (Co * Roa)]])
        Bc = np.array([[1 / Ch, 0.0], [0.0, 1 / (Co * Roa)]])
        Qc = np.diag([q, q])
        self.A, self.B, self.Q = discretize(Ac, Bc, Qc, Ts)
        self.state_names = ["heater_temperature", "object_temperature"]
        self.state_units = ["degC", "degC"]
        self.input_names = ["heating_power", "ambient_temperature"]
        self.default_x0 = np.array([self.T_amb, self.T_amb])
        self.validate()

    def inputs(self, t):
        t = np.asarray(t)
        return np.column_stack([input_signal(self.power, t), np.full(t.size, self.T_amb)])


class PendulumModel(DynamicModel):
    r"""Model 6 - nonlinear damped pendulum.

    ``theta'' = -(g/L) sin(theta) - b theta' + tau(t) + w(t)``, state
    ``x = [theta, omega]``; the discrete transition is one classical RK4 step
    of length ``Ts``.  The Jacobian of the RK4 map is obtained by central
    finite differences.  Named measurement functions:

    * ``angle``                 ``h(x) = theta``               (linear)
    * ``angular_rate``          ``h(x) = omega``               (linear)
    * ``horizontal_position``   ``h(x) = L sin(theta)``        (nonlinear)
    * ``vertical_position``     ``h(x) = -L cos(theta)``       (nonlinear)
    """

    key = "pendulum"
    label = "Nonlinear pendulum"
    description = "Damped pendulum with nonlinear dynamics and nonlinear position sensors."

    def __init__(self, Ts, params=None):
        super().__init__(Ts, params)
        self.g = param(self.params, "gravity", 9.81, positive=True)
        self.L = param(self.params, "length", 1.0, positive=True)
        self.b = param(self.params, "damping", 0.1, nonnegative=True)
        q = param(self.params, "q", 0.01, nonnegative=True)
        self.torque = dict(self.params.get("torque", {"type": "sine", "amplitude": 0.5, "frequency": 0.3}))
        self.state_names = ["angle", "angular_rate"]
        self.state_units = ["rad", "rad/s"]
        self.input_names = ["torque"]
        self.Q = white_noise_Q(2, q, Ts)
        self.default_x0 = np.array([0.8, 0.0])
        L = self.L
        self.measurement_functions = {
            "angle": MeasurementFunction(
                "angle", 1, lambda x: np.array([x[0]]), lambda x: np.array([[1.0, 0.0]]),
                "rad", "pendulum angle", linear=True),
            "angular_rate": MeasurementFunction(
                "angular_rate", 1, lambda x: np.array([x[1]]), lambda x: np.array([[0.0, 1.0]]),
                "rad/s", "angular rate", linear=True),
            "horizontal_position": MeasurementFunction(
                "horizontal_position", 1, lambda x: np.array([L * np.sin(x[0])]),
                lambda x: np.array([[L * np.cos(x[0]), 0.0]]), "m", "horizontal bob position L sin(theta)"),
            "vertical_position": MeasurementFunction(
                "vertical_position", 1, lambda x: np.array([-L * np.cos(x[0])]),
                lambda x: np.array([[L * np.sin(x[0]), 0.0]]), "m", "vertical bob position -L cos(theta)"),
        }
        self.validate()

    @property
    def is_linear(self) -> bool:
        return False

    def _deriv(self, x, u):
        return np.array([x[1], -(self.g / self.L) * np.sin(x[0]) - self.b * x[1] + u[0]])

    def f(self, x, u):
        h = self.Ts
        k1 = self._deriv(x, u)
        k2 = self._deriv(x + 0.5 * h * k1, u)
        k3 = self._deriv(x + 0.5 * h * k2, u)
        k4 = self._deriv(x + h * k3, u)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def inputs(self, t):
        return input_signal(self.torque, t)[:, None]


class CustomLinearModel(LinearModel):
    """User-defined linear model.

    Parameters (``params``):

    * ``A``, ``B`` (optional), ``Q``: discrete matrices, **or**
    * ``Ac``, ``Bc`` (optional), ``Qc`` with ``continuous: true``: continuous
      matrices discretised exactly (ZOH / Van Loan);
    * ``state_names``, ``state_units`` (optional);
    * ``inputs``: list of input signal specifications (one per input);
    * ``x0``: default initial state.
    """

    key = "custom_linear"
    label = "Custom linear model"
    description = "User-defined linear time-invariant model."

    def __init__(self, Ts, params=None):
        super().__init__(Ts, params)
        p = self.params
        continuous = bool(p.get("continuous", False))
        if continuous:
            Ac = matrix_param(p, "Ac")
            n = Ac.shape[0]
            Bc = np.atleast_2d(np.asarray(p.get("Bc", np.zeros((n, 0))), dtype=float)).reshape(n, -1)
            Qc = as_matrix(p.get("Qc", 0.0), n, "Qc")
            self.A, self.B, self.Q = discretize(Ac, Bc, Qc, Ts)
        else:
            self.A = matrix_param(p, "A")
            n = self.A.shape[0]
            self.B = np.atleast_2d(np.asarray(p.get("B", np.zeros((n, 0))), dtype=float)).reshape(n, -1)
            self.Q = as_matrix(p.get("Q", 0.0), n, "Q")
        n = self.A.shape[0]
        self.state_names = list(p.get("state_names", [f"x{i + 1}" for i in range(n)]))
        self.state_units = list(p.get("state_units", ["a.u."] * n))
        nu = self.B.shape[1]
        self.input_specs = list(p.get("inputs", [{"type": "none"}] * nu))
        if len(self.input_specs) != nu:
            raise ConfigurationError(f"custom_linear: {nu} inputs in B but {len(self.input_specs)} input specs.")
        self.input_names = [s.get("name", f"u{i + 1}") for i, s in enumerate(self.input_specs)]
        self.default_x0 = np.asarray(p.get("x0", np.zeros(n)), dtype=float)
        self.validate()

    def inputs(self, t):
        t = np.asarray(t)
        if not self.nu:
            return np.zeros((t.size, 0))
        return np.column_stack([input_signal(s, t) for s in self.input_specs])


MODEL_REGISTRY: dict[str, type[DynamicModel]] = {
    cls.key: cls
    for cls in (
        ScalarSignalModel,
        ConstantVelocityModel,
        ConstantAccelerationModel,
        MassSpringDamperModel,
        ThermalModel,
        PendulumModel,
        CustomLinearModel,
    )
}


def register_model(cls: type[DynamicModel]) -> type[DynamicModel]:
    """Class decorator to register a user model under ``cls.key``."""
    if not issubclass(cls, DynamicModel):
        raise TypeError("Only DynamicModel subclasses can be registered.")
    MODEL_REGISTRY[cls.key] = cls
    return cls


def create_model(key: str, Ts: float, params: dict | None = None) -> DynamicModel:
    try:
        cls = MODEL_REGISTRY[key]
    except KeyError:
        raise ConfigurationError(
            f"Unknown model {key!r}. Available: {', '.join(sorted(MODEL_REGISTRY))}.") from None
    return cls(Ts, params or {})


def list_models() -> list[dict]:
    return [{"key": k, "label": c.label, "description": c.description} for k, c in MODEL_REGISTRY.items()]
