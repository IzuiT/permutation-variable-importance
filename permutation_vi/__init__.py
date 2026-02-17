"""
One Permutation Variable Importance

A Python implementation of the paper "One Permutation Is All You Need:
Fast, Reliable Variable Importance and Model Stress-Testing"
(arXiv:2512.13892v2)

This package provides deterministic, efficient variable importance methods:
- DirectVariableImportance: Single-permutation VI (10-100x faster than traditional methods)
- SystemicVariableImportance: Systemic VI with correlation propagation for stress-testing
"""

from .direct_importance import DirectVariableImportance
from .systemic_importance import SystemicVariableImportance
from .utils import (
    circular_displacement,
    estimate_correlation_threshold,
    verify_permutation_optimality,
    compare_with_sklearn
)

__version__ = "1.0.0"
__author__ = "Implementation of arXiv:2512.13892v2"

__all__ = [
    "DirectVariableImportance",
    "SystemicVariableImportance",
    "circular_displacement",
    "estimate_correlation_threshold",
    "verify_permutation_optimality",
    "compare_with_sklearn"
]
