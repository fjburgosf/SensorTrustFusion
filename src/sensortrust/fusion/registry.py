"""Registry of fusion methods and named presets.

A method is requested in a configuration by name, optionally with
overrides::

    methods:
      - name: kf
      - name: sensortrust_kf
        label: "SensorTrust (lam=0.95)"
        trust: {lam: 0.95}

Presets (``METHOD_PRESETS``) map a name to a method class and default
options.  User options are merged over the preset (dictionaries such as
``trust`` are merged key by key).
"""

from __future__ import annotations

import copy

from ..utils.errors import ConfigurationError
from .base import FusionMethod
from .centralized import CentralizedFilterFusion
from .estimate_level import EstimateLevelFusion
from .static import BestSensorOracle, FixedWeightedAverage, InverseVarianceWeighting, SimpleAverage

CLASSES: dict[str, type[FusionMethod]] = {
    "simple_average": SimpleAverage,
    "fixed_weights": FixedWeightedAverage,
    "inverse_variance": InverseVarianceWeighting,
    "best_sensor_oracle": BestSensorOracle,
    "centralized": CentralizedFilterFusion,
    "estimate_level": EstimateLevelFusion,
}

_ST_TRUST = {"method": "first_two_moments"}
_ST_DET = {"method": "moment_test"}

METHOD_PRESETS: dict[str, dict] = {
    "simple_average": {"class": "simple_average"},
    "fixed_weights": {"class": "fixed_weights"},
    "inverse_variance": {"class": "inverse_variance"},
    "best_sensor_oracle": {"class": "best_sensor_oracle"},
    "kf": {"class": "centralized", "filter": "kf", "r_strategy": "nominal", "label": "Kalman filter (nominal R)",
           "description": "Centralized Kalman filter with fixed nominal R (case A)."},
    "kf_true_r": {"class": "centralized", "filter": "kf", "r_strategy": "oracle",
                  "label": "KF with true R (oracle)",
                  "description": "ORACLE: centralized KF using the true time-varying noise variance (case B)."},
    "adaptive_kf": {"class": "centralized", "filter": "kf", "r_strategy": "adaptive",
                    "label": "Adaptive KF (Sage-Husa R)",
                    "description": "Centralized KF with online Sage-Husa covariance matching of each R_i (case C)."},
    "sensortrust_kf": {"class": "centralized", "filter": "kf", "r_strategy": "trust", "trust": _ST_TRUST,
                       "detector": _ST_DET, "label": "SensorTrust KF (first two moments)",
                       "description": "Centralized KF with R_i scaled by the relative First-Two-Moments trust (case D)."},
    "sensortrust_kf_mean_only": {"class": "centralized", "filter": "kf", "r_strategy": "trust",
                                 "trust": {"method": "first_moment_only"}, "detector": _ST_DET,
                                 "label": "SensorTrust KF (first moment only)",
                                 "description": "Ablation: trust from the first moment of the innovation only."},
    "sensortrust_kf_m2_only": {"class": "centralized", "filter": "kf", "r_strategy": "trust",
                               "trust": {"method": "second_moment_only"}, "detector": _ST_DET,
                               "label": "SensorTrust KF (second moment only)",
                               "description": "Ablation: trust from the second (central) moment only."},
    "kf_nis_gate": {"class": "centralized", "filter": "kf", "r_strategy": "trust",
                    "trust": {"method": "nis_gate"}, "label": "KF with NIS gate",
                    "description": "Centralized KF rejecting samples outside the chi-square validation gate."},
    "single_sensor_kf": {"class": "centralized", "filter": "kf", "r_strategy": "nominal",
                         "label": "KF single sensor",
                         "description": "Kalman filter using only the sensors listed in 'use_sensors' "
                                        "(individual-sensor reference)."},
    "ekf": {"class": "centralized", "filter": "ekf", "r_strategy": "nominal", "label": "EKF (nominal R)"},
    "adaptive_ekf": {"class": "centralized", "filter": "ekf", "r_strategy": "adaptive",
                     "label": "Adaptive EKF (Sage-Husa R)"},
    "sensortrust_ekf": {"class": "centralized", "filter": "ekf", "r_strategy": "trust", "trust": _ST_TRUST,
                        "detector": _ST_DET, "label": "SensorTrust EKF"},
    "ukf": {"class": "centralized", "filter": "ukf", "r_strategy": "nominal", "label": "UKF (nominal R)"},
    "sensortrust_ukf": {"class": "centralized", "filter": "ukf", "r_strategy": "trust", "trust": _ST_TRUST,
                        "detector": _ST_DET, "label": "SensorTrust UKF"},
    "ci_fusion": {"class": "estimate_level", "filter": "kf", "weighting": "equal",
                  "label": "Estimate-level CI (equal weights)",
                  "description": "Local KF per sensor fused by Covariance Intersection with equal weights."},
    "sensortrust_ci": {"class": "estimate_level", "filter": "kf", "weighting": "trust", "trust": _ST_TRUST,
                       "detector": _ST_DET, "label": "SensorTrust estimate-level CI",
                       "description": "Local KF per sensor fused by Covariance Intersection with trust weights."},
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def create_method(spec: dict | str) -> FusionMethod:
    if isinstance(spec, str):
        spec = {"name": spec}
    spec = dict(spec)
    name = spec.pop("name", None)
    if name is None:
        raise ConfigurationError("Method specification requires 'name'.")
    if name in METHOD_PRESETS:
        opts = _merge(METHOD_PRESETS[name], spec)
    elif name in CLASSES:
        opts = {"class": name, **spec}
    else:
        raise ConfigurationError(
            f"Unknown fusion method {name!r}. Available: {', '.join(sorted(set(METHOD_PRESETS) | set(CLASSES)))}.")
    cls = CLASSES[opts.pop("class")]
    if cls in (CentralizedFilterFusion, EstimateLevelFusion):
        opts.setdefault("key", name)
    else:
        opts.pop("description", None)
    try:
        m = cls(**opts)
    except TypeError as exc:
        raise ConfigurationError(f"Invalid options for method {name!r}: {exc}") from None
    m.key = name
    return m


def list_methods() -> list[dict]:
    out = []
    for name in METHOD_PRESETS:
        m = create_method(name)
        out.append({"name": name, "label": m.label, "category": m.category, "oracle": m.is_oracle,
                    "description": m.description})
    return out
