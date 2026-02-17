"""
Utility functions for variable importance computation.

This module provides helper functions for:
- Circular displacement computation
- Correlation threshold estimation
- Permutation optimality verification
- Comparison with scikit-learn baselines
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Callable, Tuple, Optional
import time


def circular_displacement(a: int, b: int, n: int) -> int:
    """
    Compute circular displacement between two positions on a cycle.

    The circular displacement is the minimum distance when wrapping around
    a circular array of size n. This is used to verify permutation optimality.

    Parameters
    ----------
    a : int
        First position (0 to n-1)
    b : int
        Second position (0 to n-1)
    n : int
        Size of the circular array

    Returns
    -------
    int
        Circular displacement: min(|a-b|, n-|a-b|)

    Examples
    --------
    >>> circular_displacement(0, 5, 10)
    5
    >>> circular_displacement(1, 9, 10)
    2

    References
    ----------
    Paper equation (3), Proposition 1
    """
    diff = abs(a - b)
    return min(diff, n - diff)


def verify_permutation_optimality(permutation: np.ndarray, verbose: bool = False) -> Tuple[bool, int]:
    """
    Verify that a permutation achieves the max-min optimality criterion.

    According to Proposition 1, the optimal permutation π⌊n/2⌋ achieves
    m(π) = ⌊n/2⌋, which is the maximum possible value.

    Parameters
    ----------
    permutation : np.ndarray
        Permutation array where permutation[i] is the new position of element i
    verbose : bool, default=False
        If True, print diagnostic information

    Returns
    -------
    is_optimal : bool
        True if permutation achieves maximum min displacement
    min_displacement : int
        The minimum circular displacement achieved

    Examples
    --------
    >>> perm = np.roll(np.arange(10), 5)  # Shift by 5
    >>> is_optimal, min_disp = verify_permutation_optimality(perm)
    >>> is_optimal
    True
    >>> min_disp
    5

    References
    ----------
    Paper Proposition 1
    """
    n = len(permutation)
    expected_optimal = n // 2

    # Compute minimum circular displacement
    min_disp = n  # Start with maximum possible
    for i in range(n):
        disp = circular_displacement(i, permutation[i], n)
        min_disp = min(min_disp, disp)

    is_optimal = (min_disp == expected_optimal)

    if verbose:
        print(f"Array size: n = {n}")
        print(f"Expected optimal min displacement: ⌊n/2⌋ = {expected_optimal}")
        print(f"Achieved min displacement: {min_disp}")
        print(f"Optimal: {is_optimal}")

    return is_optimal, min_disp


def estimate_correlation_threshold(
    X: np.ndarray,
    alpha: float = 0.01,
    method: str = 'spearman',
    random_state: int = 42
) -> float:
    """
    Estimate correlation threshold for systemic importance via permutation testing.

    This implements the FWER (Family-Wise Error Rate) control described in
    Section 5.1 of the paper. We permute columns independently to create a
    null distribution, then take the (1-α) quantile.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    alpha : float, default=0.01
        Significance level for FWER control
    method : {'spearman', 'pearson'}, default='spearman'
        Correlation method to use
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    threshold : float
        Correlation threshold τ such that P(max|ρ_ij| > τ | H0) ≤ α

    Notes
    -----
    The threshold provides conservative FWER control across all pairwise
    correlations. Correlations below this threshold are considered spurious
    sampling artifacts.

    References
    ----------
    Paper Section 5.1, equations (19)-(21)
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape

    # Create null representation by independently permuting each column
    X_permuted = np.zeros_like(X)
    for j in range(n_features):
        perm_idx = rng.permutation(n_samples)
        X_permuted[:, j] = X[perm_idx, j]

    # Compute all pairwise correlations under null
    correlations = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if method == 'spearman':
                corr, _ = spearmanr(X_permuted[:, i], X_permuted[:, j])
            elif method == 'pearson':
                corr, _ = pearsonr(X_permuted[:, i], X_permuted[:, j])
            else:
                raise ValueError(f"Unknown correlation method: {method}")

            # Handle NaN (can occur with constant columns)
            if not np.isnan(corr):
                correlations.append(abs(corr))

    # Take (1-alpha) quantile
    threshold = np.quantile(correlations, 1 - alpha)

    return threshold


def compare_with_sklearn(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    scoring: Optional[str] = None,
    random_state: int = 42
) -> dict:
    """
    Benchmark DirectVariableImportance against sklearn's permutation_importance.

    This function compares the proposed method with the traditional Breiman-style
    baseline implemented in scikit-learn.

    Parameters
    ----------
    model : estimator
        Trained model with predict() method
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Target values
    n_repeats : int, default=10
        Number of repetitions for sklearn baseline
    scoring : str or None, default=None
        Scoring function (e.g., 'r2', 'accuracy')
    random_state : int, default=42
        Random seed for sklearn

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'sklearn_scores': Normalized importance from sklearn
        - 'sklearn_time': Time taken by sklearn (seconds)
        - 'our_scores_optimal': Scores from optimal permutation
        - 'our_time_optimal': Time for optimal permutation
        - 'our_scores_approx': Scores from approximate permutation
        - 'our_time_approx': Time for approximate permutation
        - 'correlation_optimal': Correlation with sklearn (optimal)
        - 'correlation_approx': Correlation with sklearn (approx)
        - 'speedup_optimal': Speedup factor (optimal)
        - 'speedup_approx': Speedup factor (approx)

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    >>> model = RandomForestRegressor(random_state=42).fit(X, y)
    >>> results = compare_with_sklearn(model, X, y)
    >>> print(f"Speedup: {results['speedup_optimal']:.1f}x")
    """
    from sklearn.inspection import permutation_importance
    from .direct_importance import DirectVariableImportance

    # Sklearn baseline
    start = time.time()
    perm_imp = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state
    )
    sklearn_time = time.time() - start

    sklearn_scores = perm_imp.importances_mean
    # Normalize to sum to 1 (handling negative values)
    sklearn_scores = np.maximum(sklearn_scores, 0)
    if sklearn_scores.sum() > 0:
        sklearn_scores = sklearn_scores / sklearn_scores.sum()
    else:
        sklearn_scores = np.ones(len(sklearn_scores)) / len(sklearn_scores)

    # Our method - optimal
    dvi_optimal = DirectVariableImportance(
        permutation_type='optimal',
        scoring_metric='mse'  # Match sklearn's default
    )
    start = time.time()
    our_scores_optimal = dvi_optimal.fit(model, X, y)
    our_time_optimal = time.time() - start

    # Our method - approximate
    dvi_approx = DirectVariableImportance(
        permutation_type='approximate',
        scoring_metric='mse'
    )
    start = time.time()
    our_scores_approx = dvi_approx.fit(model, X, y)
    our_time_approx = time.time() - start

    # Compute correlations
    corr_optimal = np.corrcoef(sklearn_scores, our_scores_optimal)[0, 1]
    corr_approx = np.corrcoef(sklearn_scores, our_scores_approx)[0, 1]

    return {
        'sklearn_scores': sklearn_scores,
        'sklearn_time': sklearn_time,
        'our_scores_optimal': our_scores_optimal,
        'our_time_optimal': our_time_optimal,
        'our_scores_approx': our_scores_approx,
        'our_time_approx': our_time_approx,
        'correlation_optimal': corr_optimal,
        'correlation_approx': corr_approx,
        'speedup_optimal': sklearn_time / our_time_optimal if our_time_optimal > 0 else float('inf'),
        'speedup_approx': sklearn_time / our_time_approx if our_time_approx > 0 else float('inf')
    }


def friedman_function(X: np.ndarray) -> np.ndarray:
    """
    Compute the Friedman benchmark function.

    This is a standard nonlinear benchmark function used in machine learning
    for testing regression methods. Defined in Friedman (1991) and used
    extensively in the paper's experiments.

    The function is: y = 10*sin(π*x1*x2) + 20*(x3 - 0.5)^2 + 10*x4 + 5*x5

    Only the first 5 features are used; remaining features are noise.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix with features in [0, 1]
        Must have at least 5 columns

    Returns
    -------
    y : np.ndarray of shape (n_samples,)
        Target values

    Notes
    -----
    Features should be uniform [0, 1] for standard behavior.
    Features beyond the first 5 are ignored (noise features).

    References
    ----------
    Paper Section 3.2.1, page 9
    Friedman, J. H. (1991). "Multivariate adaptive regression splines."
    The Annals of Statistics, 19(1), 1-67.

    Examples
    --------
    >>> rng = np.random.RandomState(42)
    >>> X = rng.uniform(0, 1, size=(100, 10))
    >>> y = friedman_function(X)
    >>> y.shape
    (100,)
    """
    if X.shape[1] < 5:
        raise ValueError("X must have at least 5 features for Friedman function")

    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    y = (
        10 * np.sin(np.pi * x1 * x2) +
        20 * (x3 - 0.5) ** 2 +
        10 * x4 +
        5 * x5
    )

    return y


def compute_friedman_ground_truth(
    n_samples: int = 100000,
    p: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute ground-truth importance for Friedman function via variance decomposition.

    This computes the contribution of each feature to the total variance of the
    Friedman function output using empirical variance analysis (Sobol indices).

    Parameters
    ----------
    n_samples : int, default=100000
        Number of Monte Carlo samples for variance estimation
    p : int, default=5
        Total number of features (first 5 are informative)
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    importance : np.ndarray of shape (p,)
        Normalized importance scores summing to 1

    Notes
    -----
    For each feature j:
    1. Sample X uniformly from [0,1]^p
    2. Compute y = friedman(X)
    3. For feature j: fix all other features, vary only x_j
    4. Measure variance contribution: Var_j = Var(E[y | x_j])
    5. Normalize: importance_j = Var_j / sum(Var_i)

    This is equivalent to first-order Sobol indices (main effects), which
    measure the fraction of output variance attributable to each input.

    References
    ----------
    - Sobol, I. M. (2001). "Global sensitivity indices for nonlinear
      mathematical models and their Monte Carlo estimates"
    - Paper Section 3.2.1 (Friedman function benchmarks)

    Examples
    --------
    >>> gt = compute_friedman_ground_truth(n_samples=100000, p=10)
    >>> gt.sum()
    1.0
    >>> gt[5:]  # Noise features should have near-zero importance
    array([0., 0., 0., 0., 0.])
    """
    rng = np.random.RandomState(random_state)

    # Generate large sample for variance estimation
    X_base = rng.uniform(0, 1, size=(n_samples, p))
    y_base = friedman_function(X_base)

    # Compute variance contribution for each feature
    variances = np.zeros(p)

    for j in range(p):
        # For feature j, compute E[y | x_j] by averaging over other features
        # We discretize x_j into bins and compute conditional expectation
        n_bins = 50
        x_j_values = X_base[:, j]

        # Sort by feature j values
        sort_idx = np.argsort(x_j_values)
        y_sorted = y_base[sort_idx]

        # Compute conditional expectations in bins
        bin_size = n_samples // n_bins
        conditional_means = []

        for b in range(n_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_bins - 1 else n_samples
            conditional_means.append(np.mean(y_sorted[start:end]))

        # Variance of conditional expectation = main effect variance
        variances[j] = np.var(conditional_means)

    # Normalize to sum to 1
    if variances.sum() > 0:
        importance = variances / variances.sum()
    else:
        # Fallback: uniform importance (shouldn't happen for Friedman)
        importance = np.ones(p) / p

    return importance


def compute_rf_ground_truth(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 100,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute ground truth importance using Random Forest (averaged over runs).

    For case studies, RF importance is used as "ground truth" benchmark
    for comparing VI methods. Averaging over multiple runs reduces variance.

    Parameters
    ----------
    model : estimator
        Trained Random Forest model
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Target values
    n_repeats : int, default=100
        Number of runs to average over
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    importance : np.ndarray of shape (n_features,)
        Averaged importance scores (normalized to sum to 1)

    Notes
    -----
    Paper case studies (Sections 5.2, 5.3) use RF importance as benchmark
    for ground-truth correlation metric in Tables 6-10.

    This provides a stable reference by averaging:
    1. Fit RF with different seeds
    2. Compute permutation importance
    3. Average across runs
    4. Normalize to sum to 1

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    >>> rf = RandomForestRegressor(random_state=42).fit(X, y)
    >>> gt_scores = compute_rf_ground_truth(rf, X, y, n_repeats=10)
    >>> gt_scores.sum()
    1.0
    """
    from sklearn.inspection import permutation_importance

    scores_list = []

    for seed in range(n_repeats):
        # Compute permutation importance with this seed
        perm_imp = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=random_state + seed
        )

        # Extract and normalize scores
        scores = perm_imp.importances_mean
        scores = np.maximum(scores, 0)  # No negative scores

        if scores.sum() > 0:
            scores = scores / scores.sum()
        else:
            scores = np.ones(len(scores)) / len(scores)

        scores_list.append(scores)

    # Average across all runs
    mean_scores = np.mean(scores_list, axis=0)

    # Normalize final result
    if mean_scores.sum() > 0:
        mean_scores = mean_scores / mean_scores.sum()

    return mean_scores
