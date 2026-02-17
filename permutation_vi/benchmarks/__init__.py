"""
Benchmark suite for validating permutation importance methods.

This module implements the comprehensive test suite from the paper
"One Permutation Is All You Need" (arXiv:2512.13892v2).
"""

from .synthetic_suite import (
    generate_linear_data,
    generate_friedman_data,
    create_scenario_grid,
    run_single_scenario,
    aggregate_results
)

__all__ = [
    'generate_linear_data',
    'generate_friedman_data',
    'create_scenario_grid',
    'run_single_scenario',
    'aggregate_results'
]
