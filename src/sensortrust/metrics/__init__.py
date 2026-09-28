"""Scientific metrics: estimation error, consistency (NIS/NEES), detection, trust, weights, robustness."""

from .consistency import consistency_summary, nees_series
from .estimation import error_stats, estimation_metrics, phase_masks
from .evaluate import SUMMARY_COLUMNS, evaluate, flatten, target_index
from .sensor_metrics import detection_metrics, trust_metrics, weight_metrics

__all__ = [
    "consistency_summary", "nees_series", "error_stats", "estimation_metrics", "phase_masks",
    "SUMMARY_COLUMNS", "evaluate", "flatten", "target_index", "detection_metrics", "trust_metrics",
    "weight_metrics",
]
