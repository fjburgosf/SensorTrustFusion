"""Degradation engine: degradation types and temporal profiles."""

from .faults import (
    ACTIVE_THRESHOLD,
    DEGRADATION_REGISTRY,
    Degradation,
    create_degradation,
    expand_degradations,
    list_degradations,
)
from .profiles import PROFILE_TYPES, Profile

__all__ = [
    "ACTIVE_THRESHOLD",
    "DEGRADATION_REGISTRY",
    "Degradation",
    "create_degradation",
    "expand_degradations",
    "list_degradations",
    "PROFILE_TYPES",
    "Profile",
]
