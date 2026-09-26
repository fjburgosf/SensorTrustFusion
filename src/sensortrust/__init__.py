"""SensorTrust Fusion.

A Scientific Platform for Adaptive Multisensor Fusion Under Sensor
Degradation and Uncertainty.

The package is organised in independent layers::

    models        -> dynamic models (linear / nonlinear)
    simulation    -> ground truth generator and scenario datasets
    sensors       -> virtual sensors and noise models
    degradation   -> degradation types and temporal profiles
    estimators    -> KF / EKF / UKF
    trust         -> innovation moments, trust estimators, degradation detectors
    fusion        -> fusion methods (baselines, centralized, estimate-level)
    metrics       -> estimation, consistency, detection, trust and weight metrics
    statistics    -> descriptive statistics and paired hypothesis tests
    experiments   -> experiment manager, benchmarks, Monte Carlo, sweeps,
                     robustness envelope, sensitivity analysis
    visualization -> scientific figures
    io            -> dataset import/export, configuration files
    gui           -> PySide6 graphical interface (no scientific logic)

The GUI is optional: every capability is available from Python and from the
command line (``python -m sensortrust --help``).
"""

from .version import __version__, SOFTWARE_NAME, FULL_NAME

__all__ = ["__version__", "SOFTWARE_NAME", "FULL_NAME"]
