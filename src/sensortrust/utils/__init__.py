"""Utility helpers (errors, random streams, logging, linear algebra, config paths)."""

from .errors import (
    SensorTrustError,
    ConfigurationError,
    DimensionError,
    CovarianceError,
    DatasetError,
)

__all__ = [
    "SensorTrustError",
    "ConfigurationError",
    "DimensionError",
    "CovarianceError",
    "DatasetError",
]
