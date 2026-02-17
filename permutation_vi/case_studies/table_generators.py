"""
Table formatting utilities for case study results.

This module provides functions to format experimental results into
publication-ready tables matching the paper's format.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

try:
    from .metrics import format_mean_std
except ImportError:
    from metrics import format_mean_std


def format_master_performance_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    task_type: str = 'regression'
) -> pd.DataFrame:
    """
    Format master model performance results (Tables 5, 8).

    Parameters
    ----------
    results : dict
        Nested dictionary: {model_name: {metric_name: (mean, std)}}
        For regression: metrics are 'r2', 'mae', 'nonzero'
        For classification: metrics are 'pr_auc', 'roc_auc', 'nonzero'
    task_type : str, default='regression'
        Either 'regression' or 'classification'

    Returns
    -------
    df : pd.DataFrame
        Formatted table with model names as columns

    Examples
    --------
    >>> results = {
    ...     'TRUST': {'r2': (0.48, 0.08), 'mae': (0.048, 0.002), 'nonzero': (10.4, 0.1)},
    ...     'OLS': {'r2': (0.49, 0.10), 'mae': (0.048, 0.002), 'nonzero': (12.0, 0.0)}
    ... }
    >>> table = format_master_performance_table(results, 'regression')

    Notes
    -----
    Paper format (Table 5, 8):
    - Rows: Metrics
    - Columns: Model names
    - Values: "mean ± std" with appropriate precision
    """
    if task_type == 'regression':
        metric_names = ['R²', 'MAE', 'Nonzero coef.']
        metric_keys = ['r2', 'mae', 'nonzero']
        precisions = [2, 3, 1]
    else:  # classification
        metric_names = ['PR-AUC', 'ROC-AUC', 'Nonzero coef.']
        metric_keys = ['pr_auc', 'roc_auc', 'nonzero']
        precisions = [2, 2, 1]

    # Build table data
    table_data = {}
    for model_name, model_results in results.items():
        table_data[model_name] = []
        for metric_key, precision in zip(metric_keys, precisions):
            mean, std = model_results[metric_key]
            if metric_key == 'nonzero':
                # Nonzero coefficients: integer formatting
                table_data[model_name].append(format_mean_std(mean, std, precision=precision))
            else:
                # Performance metrics: decimal formatting
                table_data[model_name].append(format_mean_std(mean, std, precision=precision))

    df = pd.DataFrame(table_data, index=metric_names)
    return df


def format_vi_comparison_table(
    vi_results: Dict[str, Dict[str, float]],
    master_name: str
) -> pd.DataFrame:
    """
    Format Direct VI comparison results (Tables 6, 7, 9, 10, 15-18).

    Parameters
    ----------
    vi_results : dict
        Nested dictionary: {method_name: {metric_name: value}}
        Methods: 'Direct-Opt', 'Direct-Approx', 'Breiman (B=1)', 'Breiman (B=10)'
        Metrics: 'ground_truth_cor', 'max_score_diff', 'mean_score_diff', 'time_ms'
    master_name : str
        Name of master model (e.g., 'TRUST', 'OLS')

    Returns
    -------
    df : pd.DataFrame
        Formatted table with method names as columns

    Examples
    --------
    >>> vi_results = {
    ...     'Direct-Opt': {'ground_truth_cor': 0.944, 'max_score_diff': 0.473,
    ...                    'mean_score_diff': 0.079, 'time_ms': 1.63},
    ...     'Direct-Approx': {'ground_truth_cor': 0.944, 'max_score_diff': 0.473,
    ...                       'mean_score_diff': 0.079, 'time_ms': 1.45}
    ... }
    >>> table = format_vi_comparison_table(vi_results, 'TRUST')

    Notes
    -----
    Paper format (Tables 6, 7, 9, 10):
    - Rows: Metrics
    - Columns: VI method names
    - Values: formatted with appropriate precision
    """
    metric_display_names = [
        'Ground-truth cor',
        'Max score diff',
        'Mean score diff',
        'Time (ms)'
    ]
    metric_keys = [
        'ground_truth_cor',
        'max_score_diff',
        'mean_score_diff',
        'time_ms'
    ]
    precisions = [3, 3, 3, 2]

    # Build table data
    table_data = {}
    for method_name, method_results in vi_results.items():
        table_data[method_name] = []
        for metric_key, precision in zip(metric_keys, precisions):
            value = method_results[metric_key]
            if metric_key == 'time_ms':
                # Time: show with 2 decimals
                table_data[method_name].append(f"{value:.{precision}f}")
            else:
                # Other metrics: 3 decimals
                table_data[method_name].append(f"{value:.{precision}f}")

    df = pd.DataFrame(table_data, index=metric_display_names)
    return df


def format_stability_table(
    top5_sets: Dict[str, List[Tuple[int, ...]]],
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Format stability test results (Tables 19, 20).

    Parameters
    ----------
    top5_sets : dict
        Dictionary: {method_name: list_of_top5_tuples}
        Each top5_tuple contains indices of top 5 features
        Example: {'Direct-Opt': [(0,1,2,3,4), (0,1,2,3,4), ...],
                  'Breiman (B=1)': [(0,1,3,5,6), (0,2,3,4,5), ...]}
    feature_names : list
        List of feature names for display

    Returns
    -------
    df : pd.DataFrame
        Table showing unique top-5 sets and their frequencies

    Notes
    -----
    Paper format (Tables 19, 20):
    - Rows: Unique top-5 feature sets
    - Columns: Method names
    - Values: Frequency count (out of 10 runs)
    - DVI methods: Always 10 for one set (deterministic)
    - Breiman methods: Distributed across multiple sets
    """
    # Collect all unique top-5 sets
    all_sets = set()
    for method_sets in top5_sets.values():
        for feature_set in method_sets:
            all_sets.add(tuple(sorted(feature_set)))

    all_sets = sorted(all_sets)

    # Count frequencies
    table_data = {}
    for method_name, method_sets in top5_sets.items():
        counts = {}
        for feature_set in method_sets:
            key = tuple(sorted(feature_set))
            counts[key] = counts.get(key, 0) + 1

        table_data[method_name] = [counts.get(fs, 0) for fs in all_sets]

    # Create readable row labels (feature indices)
    row_labels = [str(list(fs)) for fs in all_sets]

    df = pd.DataFrame(table_data, index=row_labels)
    df.index.name = 'Top 5 indices'

    return df


def format_svi_decomposition(
    feature_name: str,
    systemic: float,
    direct: float,
    indirect: float
) -> str:
    """
    Format SVI decomposition for pretty printing.

    Parameters
    ----------
    feature_name : str
        Name of feature (e.g., 'black', 'Sex-Marital_status')
    systemic : float
        Systemic importance score
    direct : float
        Direct importance score
    indirect : float
        Indirect importance score

    Returns
    -------
    formatted : str
        Multi-line formatted string for display

    Examples
    --------
    >>> output = format_svi_decomposition('black', 0.0044, 0.0009, 0.0035)
    >>> print(output)
    Feature: black
      Systemic:  0.44%
      Direct:    0.09%
      Indirect:  0.35%
      Multiplier: 4.89x (systemic/direct)

    Notes
    -----
    Used for displaying SVI fairness testing results.
    Paper reports these values for protected attributes.
    """
    multiplier = systemic / direct if direct > 1e-6 else float('inf')

    lines = [
        f"Feature: {feature_name}",
        f"  Systemic:  {systemic*100:.2f}%",
        f"  Direct:    {direct*100:.2f}%",
        f"  Indirect:  {indirect*100:.2f}%",
    ]

    if multiplier != float('inf'):
        lines.append(f"  Multiplier: {multiplier:.2f}x (systemic/direct)")
    else:
        lines.append(f"  Multiplier: ∞ (direct ≈ 0)")

    return "\n".join(lines)


def save_table_csv(df: pd.DataFrame, filepath: str):
    """
    Save DataFrame to CSV with proper formatting.

    Parameters
    ----------
    df : pd.DataFrame
        Table to save
    filepath : str
        Output file path

    Notes
    -----
    Saves with index for proper table structure.
    """
    df.to_csv(filepath, index=True)
    print(f"Saved table to: {filepath}")


def print_table(df: pd.DataFrame, title: str = None):
    """
    Print DataFrame with nice formatting.

    Parameters
    ----------
    df : pd.DataFrame
        Table to print
    title : str, optional
        Title to display above table
    """
    if title:
        print(f"\n{title}")
        print("=" * len(title))

    print(df.to_string())
    print()


if __name__ == "__main__":
    # Test table generators
    print("Testing table generators...")
    print("=" * 60)

    # Test master performance table (regression)
    print("\nTest 1: Master Performance Table (Regression)")
    results_reg = {
        'TRUST': {
            'r2': (0.48, 0.08),
            'mae': (0.048, 0.002),
            'nonzero': (10.4, 0.1)
        },
        'OLS': {
            'r2': (0.49, 0.10),
            'mae': (0.048, 0.002),
            'nonzero': (12.0, 0.0)
        },
        'RF': {
            'r2': (0.46, 0.08),
            'mae': (0.048, 0.002),
            'nonzero': (125802.4, 99.9)
        }
    }
    table1 = format_master_performance_table(results_reg, 'regression')
    print_table(table1, "Table 5: Master Model Performance (Boston HMDA)")

    # Test VI comparison table
    print("\nTest 2: VI Comparison Table")
    vi_results = {
        'Direct-Opt': {
            'ground_truth_cor': 0.944,
            'max_score_diff': 0.473,
            'mean_score_diff': 0.079,
            'time_ms': 1.63
        },
        'Direct-Approx': {
            'ground_truth_cor': 0.944,
            'max_score_diff': 0.473,
            'mean_score_diff': 0.079,
            'time_ms': 1.45
        },
        'Breiman (B=1)': {
            'ground_truth_cor': 0.943,
            'max_score_diff': 0.472,
            'mean_score_diff': 0.079,
            'time_ms': 5.06
        },
        'Breiman (B=10)': {
            'ground_truth_cor': 0.944,
            'max_score_diff': 0.471,
            'mean_score_diff': 0.079,
            'time_ms': 33.84
        }
    }
    table2 = format_vi_comparison_table(vi_results, 'TRUST')
    print_table(table2, "Table 6: Direct VI Comparison (TRUST master)")

    # Test SVI decomposition
    print("\nTest 3: SVI Decomposition")
    svi_text = format_svi_decomposition('black', 0.0044, 0.0009, 0.0035)
    print(svi_text)

    print("\n" + "=" * 60)
    print("All table generator tests passed!")
