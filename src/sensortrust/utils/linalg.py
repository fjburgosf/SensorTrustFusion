"""Numerically careful linear-algebra helpers."""

from __future__ import annotations

import numpy as np

from .errors import CovarianceError, DimensionError


def as_matrix(value, n: int | None = None, name: str = "matrix") -> np.ndarray:
    """Convert a scalar / vector / nested list to a 2-D float matrix.

    * scalar ``s`` with ``n`` given  -> ``s * I_n``
    * 1-D vector of length ``n``     -> ``diag(v)``
    * 2-D array                      -> itself (shape checked if ``n`` given)
    """
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        if n is None:
            return a.reshape(1, 1)
        return float(a) * np.eye(n)
    if a.ndim == 1:
        if n is not None and a.size != n:
            raise DimensionError(f"{name}: expected {n} diagonal entries, got {a.size}.")
        return np.diag(a)
    if a.ndim == 2:
        if n is not None and a.shape != (n, n):
            raise DimensionError(f"{name}: expected shape ({n}, {n}), got {a.shape}.")
        return a.copy()
    raise DimensionError(f"{name}: cannot interpret array with {a.ndim} dimensions.")


def check_covariance(C: np.ndarray, name: str = "covariance", allow_psd: bool = True,
                     tol: float = 1e-10) -> np.ndarray:
    """Validate symmetry and positive (semi)definiteness; returns symmetrised copy."""
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise CovarianceError(f"{name} must be square, got shape {C.shape}.")
    if not np.all(np.isfinite(C)):
        raise CovarianceError(f"{name} contains non-finite values.")
    scale = max(1.0, float(np.max(np.abs(C))))
    if np.max(np.abs(C - C.T)) > 1e-8 * scale:
        raise CovarianceError(f"{name} is not symmetric.")
    C = 0.5 * (C + C.T)
    eig = np.linalg.eigvalsh(C)
    if allow_psd:
        if eig.min() < -tol * scale:
            raise CovarianceError(f"{name} is not positive semidefinite (min eigenvalue {eig.min():.3g}).")
    elif eig.min() <= 0:
        raise CovarianceError(f"{name} is not positive definite (min eigenvalue {eig.min():.3g}).")
    return C


def symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)


def safe_inv_spd(S: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric positive definite matrix through Cholesky."""
    try:
        L = np.linalg.cholesky(S)
        Linv = np.linalg.inv(L)
        return Linv.T @ Linv
    except np.linalg.LinAlgError:
        return np.linalg.pinv(S)


def whiten(nu: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Return ``L^{-1} nu`` with ``S = L L^T`` (standardised innovation)."""
    if S.shape == (1, 1):
        return nu / np.sqrt(S[0, 0])
    try:
        L = np.linalg.cholesky(S)
        return np.linalg.solve(L, nu)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(S)
        w = np.clip(w, 1e-15, None)
        return (V.T @ nu) / np.sqrt(w)


def clip_spd(M: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Project ``M`` so that ``M - floor`` is positive semidefinite (eigen-clipping)."""
    D = symmetrize(M - floor)
    w, V = np.linalg.eigh(D)
    w = np.clip(w, 0.0, None)
    return floor + (V * w) @ V.T
