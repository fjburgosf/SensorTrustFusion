"""Experiment layer: configuration, manager, benchmarks, Monte Carlo, sweeps,
robustness envelopes, sensitivity analysis and design of experiments.

Submodules are imported lazily to keep ``import sensortrust`` light and to
avoid circular imports.
"""

import importlib

_LAZY = {
    "normalize_config": "config", "load_config": "config", "save_config": "config", "config_hash": "config",
    "ExperimentManager": "manager", "ExperimentResult": "manager", "run_experiment": "manager",
    "BENCHMARKS": "benchmarks", "get_benchmark": "benchmarks", "list_benchmarks": "benchmarks",
    "MonteCarloEngine": "montecarlo", "run_montecarlo": "montecarlo",
    "run_sweep": "sweep", "robustness_envelope": "robustness", "robustness_map": "robustness",
    "run_robustness": "robustness", "examples": "benchmarks", "evaluate_point": "runner",
    "run_sensitivity": "sensitivity",
}


def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(name)
