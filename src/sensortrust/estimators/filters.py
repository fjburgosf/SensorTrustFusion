r"""Gaussian recursive estimators: KF, EKF and UKF.

All filters share the interface used by the fusion layer:

* :meth:`predict` - time update ``x_{k|k-1}``, ``P_{k|k-1}``;
* :meth:`innovation` - innovation of sensor ``i`` at the prior,
  ``nu_i = z_i - hat z_i``, and its covariance ``S_i = H_i P H_i^T + R_i``
  computed with a caller-supplied ``R_i`` (the fusion layer passes the
  *nominal* ``R_i`` for trust estimation);
* :meth:`update` - joint (centralized) measurement update with any subset of
  sensors and caller-supplied *effective* covariances ``R_i^eff``.

Kalman filter (linear model, linear sensors)::

    x_{k|k-1} = A x_{k-1|k-1} + B u_{k-1}
    P_{k|k-1} = A P_{k-1|k-1} A^T + Q
    nu_k      = z_k - H x_{k|k-1}
    S_k       = H P_{k|k-1} H^T + R
    K_k       = P_{k|k-1} H^T S_k^{-1}          (solved, never inverted)
    x_{k|k}   = x_{k|k-1} + K_k nu_k
    P_{k|k}   = (I - K H) P_{k|k-1} (I - K H)^T + K R K^T   (Joseph form)

The EKF replaces ``A`` and ``H`` by the Jacobians of ``f`` and ``h`` at the
current estimate and uses ``f``/``h`` for the mean propagation.  The UKF uses
the scaled unscented transform (Julier & Uhlmann; Wan & van der Merwe) with
parameters ``alpha``, ``beta``, ``kappa``.
"""

from __future__ import annotations

import numpy as np

from ..models.base import DynamicModel
from ..sensors.sensor import SensorSpec
from ..utils.errors import MethodNotApplicableError
from ..utils.linalg import symmetrize


class GaussianFilter:
    key = "abstract"
    label = "Gaussian filter"

    def __init__(self, model: DynamicModel, specs: list[SensorSpec], x0: np.ndarray, P0: np.ndarray):
        self.model = model
        self.specs = specs
        self.n = model.n
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.I = np.eye(self.n)

    # ------------------------------------------------------------------
    def predict(self, u: np.ndarray) -> None:
        raise NotImplementedError

    def predicted_measurement(self, i: int):
        """Return ``(z_hat_i, H_i, Pzz_i)`` at the current prior (Pzz excludes R)."""
        raise NotImplementedError

    def innovation(self, i: int, z: np.ndarray, R: np.ndarray):
        zh, H, Pzz = self.predicted_measurement(i)
        nu = z - zh
        S = Pzz + R
        return nu, S, H, Pzz

    def update(self, items: list[tuple[int, np.ndarray, np.ndarray]]) -> tuple[float, int]:
        """Joint update with ``items = [(i, z_i, R_i_eff), ...]``.

        Returns ``(NIS, dof)`` of the stacked innovation with the effective
        covariances (``NIS = nu^T S^{-1} nu``).
        """
        raise NotImplementedError

    @staticmethod
    def _nis(nu: np.ndarray, S: np.ndarray) -> float:
        try:
            return float(nu @ np.linalg.solve(S, nu))
        except np.linalg.LinAlgError:
            return float("nan")


class KalmanFilter(GaussianFilter):
    key = "kf"
    label = "Kalman filter"

    def __init__(self, model, specs, x0, P0):
        if not model.is_linear:
            raise MethodNotApplicableError(
                f"The linear Kalman filter requires a linear model; '{model.key}' is nonlinear (use EKF/UKF).")
        for s in specs:
            if not s.linear:
                raise MethodNotApplicableError(
                    f"The linear Kalman filter requires linear sensors; '{s.name}' is nonlinear (use EKF/UKF).")
        super().__init__(model, specs, x0, P0)
        self.A = model.A  # type: ignore[attr-defined]
        self.B = model.B  # type: ignore[attr-defined]
        self.Q = model.Q
        self.nu_in = model.nu

    def predict(self, u):
        self.x = self.A @ self.x + (self.B @ u if self.nu_in else 0.0)
        self.P = symmetrize(self.A @ self.P @ self.A.T + self.Q)

    def predicted_measurement(self, i):
        H = self.specs[i].H
        return H @ self.x, H, H @ self.P @ H.T

    def update(self, items):
        if not items:
            return float("nan"), 0
        Hs, nus, Rs = [], [], []
        for i, z, R in items:
            H = self.specs[i].H
            Hs.append(H)
            nus.append(z - H @ self.x)
            Rs.append(R)
        return self._linear_update(np.vstack(Hs), np.concatenate(nus), _blockdiag(Rs))

    def _linear_update(self, H, nu, R):
        P = self.P
        PHt = P @ H.T
        S = symmetrize(H @ PHt + R)
        try:
            K = np.linalg.solve(S, PHt.T).T
        except np.linalg.LinAlgError:
            K = PHt @ np.linalg.pinv(S)
        self.x = self.x + K @ nu
        IKH = self.I - K @ H
        self.P = symmetrize(IKH @ P @ IKH.T + K @ R @ K.T)
        return self._nis(nu, S), nu.size


class ExtendedKalmanFilter(KalmanFilter):
    key = "ekf"
    label = "Extended Kalman filter"

    def __init__(self, model, specs, x0, P0):
        GaussianFilter.__init__(self, model, specs, x0, P0)
        self.Q = model.Q
        self.nu_in = model.nu

    def predict(self, u):
        F = self.model.F(self.x, u)
        self.x = self.model.f(self.x, u)
        self.P = symmetrize(F @ self.P @ F.T + self.Q)

    def predicted_measurement(self, i):
        s = self.specs[i]
        H = s.jacobian(self.x)
        return s.h(self.x), H, H @ self.P @ H.T

    def update(self, items):
        if not items:
            return float("nan"), 0
        Hs, nus, Rs = [], [], []
        for i, z, R in items:
            s = self.specs[i]
            Hs.append(s.jacobian(self.x))
            nus.append(z - s.h(self.x))
            Rs.append(R)
        return self._linear_update(np.vstack(Hs), np.concatenate(nus), _blockdiag(Rs))


class UnscentedKalmanFilter(GaussianFilter):
    key = "ukf"
    label = "Unscented Kalman filter"

    def __init__(self, model, specs, x0, P0, alpha: float = 0.5, beta: float = 2.0, kappa: float = 0.0):
        super().__init__(model, specs, x0, P0)
        n = self.n
        self.alpha, self.beta, self.kappa = float(alpha), float(beta), float(kappa)
        lam = alpha**2 * (n + kappa) - n
        self.c = n + lam
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * self.c))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / self.c
        self.Wc[0] = lam / self.c + (1 - alpha**2 + beta)
        self._sigma = None

    def _sigma_points(self, x, P):
        try:
            L = np.linalg.cholesky(self.c * symmetrize(P))
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(symmetrize(P))
            L = V * np.sqrt(np.clip(w, 1e-15, None) * self.c)
        X = np.empty((2 * self.n + 1, self.n))
        X[0] = x
        X[1:self.n + 1] = x + L.T
        X[self.n + 1:] = x - L.T
        return X

    def predict(self, u):
        X = self._sigma_points(self.x, self.P)
        Y = np.array([self.model.f(xi, u) for xi in X])
        x = self.Wm @ Y
        D = Y - x
        self.x = x
        self.P = symmetrize((self.Wc[:, None] * D).T @ D + self.model.Q)
        self._sigma = None

    def _points(self):
        if self._sigma is None:
            self._sigma = self._sigma_points(self.x, self.P)
        return self._sigma

    def _zsig(self, i):
        s = self.specs[i]
        return np.array([s.h(xi) for xi in self._points()]).reshape(-1, s.m)

    def predicted_measurement(self, i):
        Z = self._zsig(i)
        zh = self.Wm @ Z
        D = Z - zh
        Pzz = (self.Wc[:, None] * D).T @ D
        H = self.specs[i].jacobian(self.x)  # reported only (diagnostics / weights)
        return zh, H, Pzz

    def update(self, items):
        if not items:
            return float("nan"), 0
        X = self._points()
        Zs, zs, Rs = [], [], []
        for i, z, R in items:
            Zs.append(self._zsig(i))
            zs.append(z)
            Rs.append(R)
        Z = np.hstack(Zs)
        zh = self.Wm @ Z
        DZ = Z - zh
        DX = X - self.x
        R = _blockdiag(Rs)
        S = symmetrize((self.Wc[:, None] * DZ).T @ DZ + R)
        Pxz = (self.Wc[:, None] * DX).T @ DZ
        try:
            K = np.linalg.solve(S, Pxz.T).T
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(S)
        nu = np.concatenate(zs) - zh
        self.x = self.x + K @ nu
        self.P = symmetrize(self.P - K @ S @ K.T)
        self._sigma = None
        return self._nis(nu, S), nu.size


def _blockdiag(blocks):
    if len(blocks) == 1:
        return blocks[0]
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n))
    r = 0
    for b in blocks:
        m = b.shape[0]
        out[r:r + m, r:r + m] = b
        r += m
    return out


FILTERS = {"kf": KalmanFilter, "ekf": ExtendedKalmanFilter, "ukf": UnscentedKalmanFilter}


def create_filter(kind: str, model, specs, x0, P0, **kw) -> GaussianFilter:
    try:
        cls = FILTERS[kind]
    except KeyError:
        raise MethodNotApplicableError(f"Unknown filter {kind!r}; available: {', '.join(FILTERS)}.") from None
    return cls(model, specs, x0, P0, **kw)
