"""Fusion methods: baselines, centralized filters (cases A-D) and estimate-level fusion."""

from .base import FusionMethod, MethodResult, OracleInfo
from .centralized import R_STRATEGIES, CentralizedFilterFusion
from .estimate_level import EstimateLevelFusion
from .registry import CLASSES, METHOD_PRESETS, create_method, list_methods
from .static import BestSensorOracle, FixedWeightedAverage, InverseVarianceWeighting, SimpleAverage

__all__ = [
    "FusionMethod", "MethodResult", "OracleInfo", "R_STRATEGIES", "CentralizedFilterFusion",
    "EstimateLevelFusion", "CLASSES", "METHOD_PRESETS", "create_method", "list_methods",
    "BestSensorOracle", "FixedWeightedAverage", "InverseVarianceWeighting", "SimpleAverage",
]
