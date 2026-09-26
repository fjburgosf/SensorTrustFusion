"""Design of experiments.

Samplers produce points in the unit hypercube ``[0, 1)^d`` which are mapped
to each parameter through the inverse CDF of its distribution:

==============  ================================================================
sampler         method
==============  ================================================================
random          independent uniform draws (plain Monte Carlo)
lhs             Latin Hypercube Sampling (scipy.stats.qmc.LatinHypercube)
sobol           scrambled Sobol sequence (scipy.stats.qmc.Sobol)
grid            full factorial grid over explicit ``values`` (or ``levels``)
==============  ================================================================

Parameter distributions (``distribution`` key): ``uniform(low, high)``,
``loguniform(low, high)``, ``normal(mean, std)``, ``truncnormal(mean, std,
low, high)``, ``integer(low, high)`` (inclusive) and ``choice(values)``.
The sampling itself is seeded (``master_seed``) and every generated design is
stored with the study.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats
from scipy.stats import qmc

from ..utils import rng as rngmod
from ..utils.errors import ConfigurationError

SAMPLERS = ("random", "lhs", "sobol", "grid")
DISTRIBUTIONS = ("uniform", "loguniform", "normal", "truncnormal", "integer", "choice")


def _check_param(p: dict) -> None:
    if "path" not in p:
        raise ConfigurationError("Each varied parameter requires a 'path'.")
    d = p.get("distribution", "uniform")
    if d not in DISTRIBUTIONS:
        raise ConfigurationError(f"Unknown distribution {d!r}; use one of {DISTRIBUTIONS}.")
    if d in ("uniform", "loguniform", "integer"):
        if "low" not in p or "high" not in p or float(p["low"]) > float(p["high"]):
            raise ConfigurationError(f"Parameter {p['path']}: '{d}' requires low <= high.")
        if d == "loguniform" and float(p["low"]) <= 0:
            raise ConfigurationError(f"Parameter {p['path']}: loguniform requires low > 0.")
    if d in ("normal", "truncnormal") and float(p.get("std", 0)) <= 0:
        raise ConfigurationError(f"Parameter {p['path']}: '{d}' requires std > 0.")
    if d == "choice" and not p.get("values"):
        raise ConfigurationError(f"Parameter {p['path']}: 'choice' requires a non-empty 'values' list.")


def map_unit(u: float, p: dict):
    d = p.get("distribution", "uniform")
    u = min(max(float(u), 1e-12), 1 - 1e-12)
    if d == "uniform":
        return float(p["low"]) + u * (float(p["high"]) - float(p["low"]))
    if d == "loguniform":
        lo, hi = np.log(float(p["low"])), np.log(float(p["high"]))
        return float(np.exp(lo + u * (hi - lo)))
    if d == "normal":
        return float(stats.norm.ppf(u, float(p["mean"]), float(p["std"])))
    if d == "truncnormal":
        m, s = float(p["mean"]), float(p["std"])
        a, b = (float(p.get("low", -np.inf)) - m) / s, (float(p.get("high", np.inf)) - m) / s
        return float(stats.truncnorm.ppf(u, a, b, loc=m, scale=s))
    if d == "integer":
        lo, hi = int(p["low"]), int(p["high"])
        return int(min(hi, lo + np.floor(u * (hi - lo + 1))))
    vals = list(p["values"])
    return vals[min(len(vals) - 1, int(np.floor(u * len(vals))))]


def unit_samples(n: int, d: int, sampler: str = "random", seed: int = 0) -> np.ndarray:
    if sampler == "random":
        return rngmod.stream(seed, "design", "random").random((n, d))
    if sampler == "lhs":
        return qmc.LatinHypercube(d=d, seed=rngmod.stream(seed, "design", "lhs")).random(n)
    if sampler == "sobol":
        eng = qmc.Sobol(d=d, scramble=True, seed=rngmod.stream(seed, "design", "sobol"))
        m = int(np.ceil(np.log2(max(n, 2))))
        return eng.random_base2(m)[:n]
    raise ConfigurationError(f"Unknown sampler {sampler!r}; use one of {SAMPLERS}.")


def sample_design(parameters: list[dict], n: int, sampler: str = "random", seed: int = 0) -> list[dict]:
    """Return ``n`` dictionaries ``{path: value}``."""
    if not parameters:
        return [{} for _ in range(n)]
    if sampler == "grid":
        return grid_design(parameters)
    for p in parameters:
        _check_param(p)
    U = unit_samples(n, len(parameters), sampler, seed)
    return [{p["path"]: map_unit(U[i, j], p) for j, p in enumerate(parameters)} for i in range(U.shape[0])]


def grid_values(p: dict) -> list:
    if "values" in p:
        return list(p["values"])
    lv = int(p.get("levels", 5))
    if lv < 2:
        raise ConfigurationError("Grid 'levels' must be >= 2.")
    if p.get("distribution") == "loguniform" or p.get("scale") == "log":
        return list(np.geomspace(float(p["low"]), float(p["high"]), lv))
    return list(np.linspace(float(p["low"]), float(p["high"]), lv))


def grid_design(parameters: list[dict]) -> list[dict]:
    axes = [grid_values(p) for p in parameters]
    return [{p["path"]: v for p, v in zip(parameters, combo)} for combo in itertools.product(*axes)]
