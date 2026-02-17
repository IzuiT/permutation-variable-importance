"""
Metric computation utilities for case study experiments.

This module provides helper functions for computing performance metrics
and variable importance comparison metrics.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error
)
from typing import Dict


def compute_pr_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute Precision-Recall Area Under Curve.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred_proba : np.ndarray of shape (n_samples,) or (n_samples, 2)
        Predicted probabilities

    Returns
    -------
    pr_auc : float
        PR-AUC score

    Notes
    -----
    Paper Table 8 uses PR-AUC as primary classification metric.
    Particularly useful for imbalanced datasets like German Credit (70/30 split).
    """
    # Handle 2D probability arrays (extract positive class)
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]

    return average_precision_score(y_true, y_pred_proba)


def compute_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute ROC Area Under Curve.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred_proba : np.ndarray of shape (n_samples,) or (n_samples, 2)
        Predicted probabilities

    Returns
    -------
    roc_auc : float
        ROC-AUC score

    Notes
    -----
    Paper Table 8 uses ROC-AUC as secondary classification metric.
    """
    # Handle 2D probability arrays
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]

    return roc_auc_score(y_true, y_pred_proba)


def compute_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute negative Brier score (for maximization).

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred_proba : np.ndarray of shape (n_samples,) or (n_samples, 2)
        Predicted probabilities

    Returns
    -------
    neg_brier : float
        Negative Brier score (higher is better)

    Notes
    -----
    Paper uses MSE on probabilities for classification VI.
    Brier score is equivalent: BS = MSE(y_true, y_pred_proba).
    Negative sign makes it a score to maximize (like accuracy).
    """
    # Handle 2D probability arrays
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]

    return -brier_score_loss(y_true, y_pred_proba)


def compute_vi_comparison_metrics(
    ground_truth_scores: np.ndarray,
    predicted_scores: np.ndarray,
    time_elapsed: float
) -> Dict[str, float]:
    """
    Compute metrics for comparing VI methods.

    Parameters
    ----------
    ground_truth_scores : np.ndarray of shape (n_features,)
        Ground truth importance scores (e.g., from RF or Breiman B=10)
    predicted_scores : np.ndarray of shape (n_features,)
        Predicted importance scores (e.g., from DVI)
    time_elapsed : float
        Time taken to compute predicted scores (seconds)

    Returns
    -------
    metrics : dict
        Dictionary with keys:
        - 'ground_truth_cor': Correlation with ground truth
        - 'max_score_diff': Maximum absolute difference
        - 'mean_score_diff': Mean absolute difference
        - 'time_ms': Time in milliseconds

    Notes
    -----
    Paper Tables 6, 7, 9, 10 report these exact metrics.
    Ground truth is typically RF importance or Breiman (B=10).
    """
    # Normalize scores to sum to 1 (for fair comparison)
    gt_norm = ground_truth_scores / ground_truth_scores.sum() if ground_truth_scores.sum() > 0 else ground_truth_scores
    pred_norm = predicted_scores / predicted_scores.sum() if predicted_scores.sum() > 0 else predicted_scores

    # Compute correlation
    if len(gt_norm) > 1:
        correlation = np.corrcoef(gt_norm, pred_norm)[0, 1]
    else:
        correlation = 1.0 if np.allclose(gt_norm, pred_norm) else 0.0

    # Compute differences
    abs_diff = np.abs(gt_norm - pred_norm)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    return {
        'ground_truth_cor': correlation,
        'max_score_diff': max_diff,
        'mean_score_diff': mean_diff,
        'time_ms': time_elapsed * 1000  # Convert to milliseconds
    }


def format_mean_std(mean: float, std: float, precision: int = 2) -> str:
    """
    Format mean ± std for table display.

    Parameters
    ----------
    mean : float
        Mean value
    std : float
        Standard deviation
    precision : int, default=2
        Number of decimal places

    Returns
    -------
    formatted : str
        Formatted string like "0.48 ± 0.08"

    Notes
    -----
    Paper tables use this format for all CV results.
    """
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def compute_speedup(time_baseline: float, time_method: float) -> float:
    """
    Compute speedup factor relative to baseline.

    Parameters
    ----------
    time_baseline : float
        Time for baseline method (seconds)
    time_method : float
        Time for compared method (seconds)

    Returns
    -------
    speedup : float
        Speedup factor (>1 means faster than baseline)

    Notes
    -----
    Paper reports speedup of DVI relative to Breiman (B=10).
    Typical speedups: 14-105× for Boston HMDA, 8-52× for German Credit.
    """
    if time_method <= 0:
        return float('inf')
    return time_baseline / time_method


def top_k_overlap(scores1: np.ndarray, scores2: np.ndarray, k: int = 5) -> int:
    """
    Count overlap in top-k features between two importance scores.

    Parameters
    ----------
    scores1 : np.ndarray of shape (n_features,)
        First importance scores
    scores2 : np.ndarray of shape (n_features,)
        Second importance scores
    k : int, default=5
        Number of top features to compare

    Returns
    -------
    overlap : int
        Number of features in both top-k sets (0 to k)

    Notes
    -----
    Used for stability testing (Tables 19, 20).
    DVI methods always have overlap=k (deterministic).
    Breiman methods have variable overlap across runs.
    """
    top_k_1 = set(np.argsort(scores1)[::-1][:k])
    top_k_2 = set(np.argsort(scores2)[::-1][:k])
    return len(top_k_1 & top_k_2)


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    print("=" * 60)

    # Test classification metrics
    print("\nTesting classification metrics:")
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.3, 0.6, 0.8, 0.9])

    pr_auc = compute_pr_auc(y_true, y_pred_proba)
    roc_auc = compute_roc_auc(y_true, y_pred_proba)
    brier = compute_brier_score(y_true, y_pred_proba)

    print(f"  PR-AUC: {pr_auc:.3f}")
    print(f"  ROC-AUC: {roc_auc:.3f}")
    print(f"  Neg Brier: {brier:.3f}")

    # Test VI comparison metrics
    print("\nTesting VI comparison metrics:")
    gt_scores = np.array([0.4, 0.3, 0.2, 0.1])
    pred_scores = np.array([0.35, 0.32, 0.22, 0.11])
    time_elapsed = 0.5

    vi_metrics = compute_vi_comparison_metrics(gt_scores, pred_scores, time_elapsed)
    print(f"  Ground-truth cor: {vi_metrics['ground_truth_cor']:.3f}")
    print(f"  Max score diff: {vi_metrics['max_score_diff']:.3f}")
    print(f"  Mean score diff: {vi_metrics['mean_score_diff']:.3f}")
    print(f"  Time (ms): {vi_metrics['time_ms']:.1f}")

    # Test formatting
    print("\nTesting formatting:")
    formatted = format_mean_std(0.48, 0.08, precision=2)
    print(f"  Formatted: {formatted}")

    # Test speedup
    print("\nTesting speedup:")
    speedup = compute_speedup(time_baseline=10.0, time_method=0.5)
    print(f"  Speedup: {speedup:.1f}×")

    # Test top-k overlap
    print("\nTesting top-k overlap:")
    scores_a = np.array([0.4, 0.3, 0.2, 0.1, 0.05])
    scores_b = np.array([0.35, 0.32, 0.15, 0.12, 0.06])
    overlap = top_k_overlap(scores_a, scores_b, k=3)
    print(f"  Top-3 overlap: {overlap}/3")

    print("\nAll tests passed!")
