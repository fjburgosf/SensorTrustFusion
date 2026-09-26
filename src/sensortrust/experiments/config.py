"""Experiment configuration schema.

An experiment is fully described by a plain mapping (YAML / JSON file)::

    experiment:  {name, description, seed, tags}
    model:       {type, params, x0, P0, sample_initial_state}
    simulation:  {duration [s], fs [Hz]}
    sensors:     [ {name, measures | H | function, rate, noise, bias, ...,
                    degradations: [ {type, severity, profile: {...}} ]} ]
    estimation:  {x0, P0}              # prior given to the estimators
    methods:     [ {name, ...method options} ]
    metrics:     {target_state, warmup, rmse_max, trust_threshold, ...}
    output:      {figures: [png, svg], figure_dpi}

:func:`normalize_config` fills defaults and validates types, ranges and
cross-references so that errors are reported before running anything.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..utils.errors import ConfigurationError
from ..utils.rng import validate_seed

DEFAULT_METHODS = [
    {"name": "simple_average"},
    {"name": "inverse_variance"},
    {"name": "kf"},
    {"name": "adaptive_kf"},
    {"name": "sensortrust_kf"},
]

DEFAULT_METRICS = {
    "target_state": 0,
    "warmup": 1.0,               # s excluded at the beginning of every metric
    "rmse_max": None,            # specification for success / robustness
    "divergence_threshold": None,  # |error| above this -> diverged (default: 1e3 * max sensor std)
    "trust_threshold": 0.5,      # response time: T_i below this value
    "recovery_fraction": 0.9,    # recovery: T_i back above this fraction of its nominal mean
    "weight_fraction": 0.5,      # weight reduction time: w_i below this fraction of nominal mean
}

DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "Unnamed experiment", "description": "", "seed": 42, "tags": []},
    "model": {"type": "constant_velocity", "params": {}, "x0": None, "P0": 0.01, "sample_initial_state": True},
    "simulation": {"duration": 60.0, "fs": 100.0},
    "sensors": [],
    "estimation": {"x0": None, "P0": None},
    "methods": None,
    "metrics": DEFAULT_METRICS,
    "output": {"figures": ["png"], "figure_dpi": 150, "save_dataset": True},
}


def _merge(defaults: dict, user: dict) -> dict:
    out = copy.deepcopy(defaults)
    for k, v in (user or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def normalize_config(cfg: dict) -> dict:
    """Return a validated deep copy of ``cfg`` with all defaults filled in."""
    if not isinstance(cfg, dict):
        raise ConfigurationError("The experiment configuration must be a mapping.")
    unknown = set(cfg) - set(DEFAULTS) - {"montecarlo", "analysis", "dataset"}
    if unknown:
        raise ConfigurationError(f"Unknown configuration section(s): {', '.join(sorted(unknown))}.")
    c = _merge(DEFAULTS, _to_builtin(cfg))
    c["experiment"]["seed"] = validate_seed(c["experiment"]["seed"])
    sim = c["simulation"]
    try:
        sim["duration"] = float(sim["duration"])
        sim["fs"] = float(sim["fs"])
    except (TypeError, ValueError):
        raise ConfigurationError("simulation.duration and simulation.fs must be numeric.") from None
    if not (np.isfinite(sim["duration"]) and sim["duration"] > 0):
        raise ConfigurationError("simulation.duration must be > 0 s.")
    if not (np.isfinite(sim["fs"]) and sim["fs"] > 0):
        raise ConfigurationError("simulation.fs must be > 0 Hz.")
    K = int(round(sim["duration"] * sim["fs"]))
    if K < 2:
        raise ConfigurationError("The horizon must contain at least 2 samples (duration * fs >= 2).")
    if K > 5_000_000:
        raise ConfigurationError("Horizon too long (more than 5e6 samples).")
    if not isinstance(c["sensors"], list) or len(c["sensors"]) == 0:
        raise ConfigurationError("At least one sensor must be configured.")
    names = []
    for i, s in enumerate(c["sensors"]):
        if not isinstance(s, dict):
            raise ConfigurationError(f"Sensor #{i + 1} must be a mapping.")
        s.setdefault("name", f"S{i + 1}")
        s.setdefault("degradations", [])
        names.append(s["name"])
        for d in s["degradations"]:
            prof = d.get("profile", {}) if isinstance(d, dict) else {}
            for key in ("start", "recovery_time"):
                v = prof.get(key)
                if v is not None and float(v) > sim["duration"]:
                    raise ConfigurationError(
                        f"Sensor '{s['name']}': degradation {key}={v} s is outside the horizon "
                        f"[0, {sim['duration']}] s.")
    if len(set(names)) != len(names):
        raise ConfigurationError(f"Sensor names must be unique, got {names}.")
    if c["methods"] is None:
        c["methods"] = copy.deepcopy(DEFAULT_METHODS)
    if not isinstance(c["methods"], list) or not c["methods"]:
        raise ConfigurationError("'methods' must be a non-empty list.")
    for i, m in enumerate(c["methods"]):
        if isinstance(m, str):
            c["methods"][i] = {"name": m}
        elif not isinstance(m, dict) or "name" not in m:
            raise ConfigurationError(f"Method #{i + 1} must be a name or a mapping with 'name'.")
    labels = [m.get("label", m["name"]) for m in c["methods"]]
    if len(set(labels)) != len(labels):
        raise ConfigurationError("Method labels must be unique (use 'label' to distinguish variants).")
    met = c["metrics"]
    w = float(met.get("warmup", 0.0))
    if w < 0 or w >= sim["duration"]:
        raise ConfigurationError("metrics.warmup must be in [0, duration).")
    return c


def config_hash(cfg: dict) -> str:
    """SHA-256 of the canonical JSON form of a (normalized) configuration."""
    payload = json.dumps(_to_builtin(cfg), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def syntax_error_position(exc) -> tuple:
    """(line, column), 1-based, of a YAML or JSON syntax error ('?' if unknown)."""
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    if mark is not None:
        return mark.line + 1, mark.column + 1
    if isinstance(exc, json.JSONDecodeError):
        return exc.lineno, exc.colno
    return "?", "?"


def load_config(path: str | Path) -> dict:
    """Load a YAML or JSON configuration file (not normalized)."""
    p = Path(path)
    if not p.exists():
        raise ConfigurationError(f"Configuration file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        if p.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        line, col = syntax_error_position(exc)
        raise ConfigurationError(f"Cannot parse configuration file {p.name}: syntax error at line {line}, "
                                 f"column {col}.") from None
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration file {p.name} does not contain a mapping.")
    return data


def save_config(cfg: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _to_builtin(cfg)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


to_builtin = _to_builtin
