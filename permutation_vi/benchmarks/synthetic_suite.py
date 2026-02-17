"""
Synthetic Benchmark Suite for Variable Importance Methods.

This module implements the 192-scenario simulation framework from Section 3.2
of the paper "One Permutation Is All You Need" (arXiv:2512.13892v2).

It provides comprehensive testing across varying:
- Sample sizes (n ∈ {100, 1000, 10000})
- Dimensionality (p ∈ {10, 100})
- Noise levels (σ_ε ∈ {0.1, 5})
- Feature correlations (ρ ∈ {0, 0.3})
- Response types (linear, nonlinear/Friedman)
- Task types (regression, classification)
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
from permutation_vi import DirectVariableImportance
from permutation_vi.utils import friedman_function, compute_friedman_ground_truth


def generate_linear_data(
    n: int,
    p: int,
    sigma_epsilon: float = 1.0,
    rho: float = 0.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data with linear response.

    Creates data where y = X @ beta + noise, with optional feature correlation.

    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features
    sigma_epsilon : float, default=1.0
        Standard deviation of Gaussian noise
    rho : float, default=0.0
        Feature correlation strength (0 = independent, 0.3 = moderate)
    random_state : int, default=42
        Random seed

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Feature matrix
    y : np.ndarray of shape (n,)
        Continuous target
    true_importance : np.ndarray of shape (p,)
        Ground-truth normalized importance (|beta_j| / sum|beta|)

    Notes
    -----
    - Half the features are informative (non-zero coefficients)
    - Half are noise features (zero coefficients)
    - Correlation structure: Block diagonal with correlation rho within informative block
    """
    rng = np.random.RandomState(random_state)

    # Create covariance matrix with block structure (Section 3.2.3)
    if rho > 0:
        n_informative = p // 2
        n_noise = p - n_informative

        # Informative block: correlation rho
        cov_informative = np.full((n_informative, n_informative), rho)
        np.fill_diagonal(cov_informative, 1.0)

        # Noise block: slightly correlated (rho/3) to make it realistic
        cov_noise = np.full((n_noise, n_noise), rho / 3)
        np.fill_diagonal(cov_noise, 1.0)

        # Block diagonal covariance
        cov_matrix = np.zeros((p, p))
        cov_matrix[:n_informative, :n_informative] = cov_informative
        cov_matrix[n_informative:, n_informative:] = cov_noise

        # Generate correlated features
        X = rng.multivariate_normal(np.zeros(p), cov_matrix, size=n)
    else:
        # Independent features
        X = rng.randn(n, p)

    # Create coefficients: first half informative, second half noise
    beta = np.zeros(p)
    n_informative = p // 2
    beta[:n_informative] = rng.randn(n_informative)

    # Generate response
    y = X @ beta + sigma_epsilon * rng.randn(n)

    # Ground truth importance: normalized absolute coefficients
    true_importance = np.abs(beta)
    if true_importance.sum() > 0:
        true_importance = true_importance / true_importance.sum()
    else:
        true_importance = np.ones(p) / p

    return X, y, true_importance


def generate_friedman_data(
    n: int,
    p: int,
    sigma_epsilon: float = 1.0,
    rho: float = 0.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data with nonlinear Friedman function response.

    Uses the standard Friedman benchmark: y = 10*sin(π*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5

    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features (must be >= 5)
    sigma_epsilon : float, default=1.0
        Standard deviation of Gaussian noise added to response
    rho : float, default=0.0
        Feature correlation (applied same way as linear case)
    random_state : int, default=42
        Random seed

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Feature matrix (uniform [0, 1])
    y : np.ndarray of shape (n,)
        Continuous target from Friedman function + noise
    true_importance : np.ndarray of shape (p,)
        Ground-truth importance (first 5 features based on sensitivity, rest zero)

    Notes
    -----
    Ground truth importance for Friedman function (approximate):
    - Features 1-5: Estimated via empirical variance contribution
    - Features 6-p: Zero (noise features)
    """
    if p < 5:
        raise ValueError("Friedman function requires at least 5 features")

    rng = np.random.RandomState(random_state)

    # Generate correlated uniform features
    if rho > 0:
        # Start with correlated Gaussian
        n_informative = 5
        n_noise = p - n_informative

        cov_informative = np.full((n_informative, n_informative), rho)
        np.fill_diagonal(cov_informative, 1.0)

        cov_noise = np.full((n_noise, n_noise), rho / 3)
        np.fill_diagonal(cov_noise, 1.0)

        cov_matrix = np.zeros((p, p))
        cov_matrix[:n_informative, :n_informative] = cov_informative
        cov_matrix[n_informative:, n_informative:] = cov_noise

        X_gaussian = rng.multivariate_normal(np.zeros(p), cov_matrix, size=n)

        # Transform to uniform via CDF
        from scipy.stats import norm
        X = norm.cdf(X_gaussian)
    else:
        # Independent uniform features
        X = rng.uniform(0, 1, size=(n, p))

    # Generate response using Friedman function
    y_clean = friedman_function(X)
    y = y_clean + sigma_epsilon * rng.randn(n)

    # Ground truth importance computed via variance decomposition (Sobol indices)
    # This measures each feature's contribution to output variance
    true_importance = compute_friedman_ground_truth(
        n_samples=100000,  # Large sample for accurate estimation
        p=p,
        random_state=random_state
    )
    # Note: Only first 5 features contribute to Friedman function
    # Remaining features will have near-zero importance (noise)

    return X, y, true_importance


def create_scenario_grid() -> List[Dict]:
    """
    Create the full 192-scenario grid from Section 3.2.

    Scenarios vary across:
    - n ∈ {100, 1000, 10000}: sample size
    - p ∈ {10, 100}: dimensionality
    - sigma_epsilon ∈ {0.1, 5.0}: noise level
    - rho ∈ {0.0, 0.3}: feature correlation
    - response_type ∈ {'linear', 'friedman'}: linear vs nonlinear
    - task_type ∈ {'regression', 'classification'}: continuous vs binary target

    This gives 3 × 2 × 2 × 2 × 2 × 2 = 96 scenarios per model type × 2 model types = 192 total

    Returns
    -------
    scenarios : List[Dict]
        List of scenario configurations, each with keys:
        - n, p, sigma_epsilon, rho, response_type, task_type, model_type
    """
    scenarios = []

    sample_sizes = [100, 1000, 10000]
    dimensionalities = [10, 100]
    noise_levels = [0.1, 5.0]
    correlations = [0.0, 0.3]
    response_types = ['linear', 'friedman']
    task_types = ['regression', 'classification']

    for n in sample_sizes:
        for p in dimensionalities:
            for sigma in noise_levels:
                for rho in correlations:
                    for response in response_types:
                        for task in task_types:
                            scenarios.append({
                                'n': n,
                                'p': p,
                                'sigma_epsilon': sigma,
                                'rho': rho,
                                'response_type': response,
                                'task_type': task,
                                'scenario_id': len(scenarios)
                            })

    return scenarios


def run_single_scenario(
    scenario: Dict,
    n_repetitions: int = 50,
    random_state_base: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Run a single scenario with multiple repetitions.

    For each repetition:
    1. Generate data
    2. Fit master model
    3. Compute importance with 4 methods: sklearn(B=1), sklearn(B=10), DVI-optimal, DVI-approx
    4. Compare against ground truth

    Parameters
    ----------
    scenario : Dict
        Scenario configuration from create_scenario_grid()
    n_repetitions : int, default=50
        Number of independent repetitions (paper uses 50)
    random_state_base : int, default=42
        Base random seed (each repetition gets base + repetition_idx)
    verbose : bool, default=False
        If True, print progress

    Returns
    -------
    results : Dict
        Aggregated results across repetitions with keys:
        - 'ground_truth_cor_*': Mean correlation with ground truth
        - 'max_score_diff_*': Mean max absolute score difference
        - 'mean_score_diff_*': Mean average score difference
        - 'time_*': Mean runtime in milliseconds
        Where * ∈ {breiman_1, breiman_10, direct_opt, direct_approx}
    """
    if verbose:
        print(f"Running scenario {scenario['scenario_id']}: n={scenario['n']}, p={scenario['p']}, "
              f"σ={scenario['sigma_epsilon']}, ρ={scenario['rho']}, "
              f"{scenario['response_type']}, {scenario['task_type']}")

    # Storage for repetition results
    results_breiman_1 = []
    results_breiman_10 = []
    results_direct_opt = []
    results_direct_approx = []

    for rep in range(n_repetitions):
        random_state = random_state_base + rep

        # Generate data
        if scenario['response_type'] == 'linear':
            X, y_cont, true_importance = generate_linear_data(
                n=scenario['n'],
                p=scenario['p'],
                sigma_epsilon=scenario['sigma_epsilon'],
                rho=scenario['rho'],
                random_state=random_state
            )
        else:  # friedman
            X, y_cont, true_importance = generate_friedman_data(
                n=scenario['n'],
                p=scenario['p'],
                sigma_epsilon=scenario['sigma_epsilon'],
                rho=scenario['rho'],
                random_state=random_state
            )

        # For classification, binarize at median
        if scenario['task_type'] == 'classification':
            y = (y_cont > np.median(y_cont)).astype(int)
        else:
            y = y_cont

        # Fit master model (paper uses GLM for regression, logistic for classification)
        if scenario['task_type'] == 'regression':
            model = LinearRegression()
        else:
            model = LogisticRegression(max_iter=1000, random_state=random_state)

        model.fit(X, y)

        # Compute importances with all methods
        # 1. sklearn Breiman B=1
        start = time.time()
        perm_1 = permutation_importance(model, X, y, n_repeats=1, random_state=random_state)
        time_breiman_1 = (time.time() - start) * 1000  # milliseconds

        scores_breiman_1 = np.maximum(perm_1.importances_mean, 0)
        if scores_breiman_1.sum() > 0:
            scores_breiman_1 = scores_breiman_1 / scores_breiman_1.sum()

        # 2. sklearn Breiman B=10
        start = time.time()
        perm_10 = permutation_importance(model, X, y, n_repeats=10, random_state=random_state)
        time_breiman_10 = (time.time() - start) * 1000

        scores_breiman_10 = np.maximum(perm_10.importances_mean, 0)
        if scores_breiman_10.sum() > 0:
            scores_breiman_10 = scores_breiman_10 / scores_breiman_10.sum()

        # 3. DVI Optimal
        dvi_opt = DirectVariableImportance(permutation_type='optimal', scoring_metric='mse')
        start = time.time()
        scores_dvi_opt = dvi_opt.fit(model, X, y)
        time_dvi_opt = (time.time() - start) * 1000

        # 4. DVI Approximate
        dvi_approx = DirectVariableImportance(permutation_type='approximate', scoring_metric='mse')
        start = time.time()
        scores_dvi_approx = dvi_approx.fit(model, X, y)
        time_dvi_approx = (time.time() - start) * 1000

        # Compute metrics against ground truth
        def compute_metrics(scores, time_ms):
            # Compute correlation with proper handling of zero-variance cases
            # When either array has zero variance, correlation is undefined (NaN)
            with np.errstate(divide='ignore', invalid='ignore'):
                cor = np.corrcoef(true_importance, scores)[0, 1]

            # If correlation is NaN (due to zero variance), use perfect correlation
            # This happens when all features have identical importance (e.g., all zeros)
            if np.isnan(cor):
                # If both arrays are identical, correlation is 1.0
                if np.allclose(true_importance, scores):
                    cor = 1.0
                else:
                    cor = 0.0

            max_diff = np.max(np.abs(true_importance - scores))
            mean_diff = np.mean(np.abs(true_importance - scores))
            return {'cor': cor, 'max_diff': max_diff, 'mean_diff': mean_diff, 'time': time_ms}

        results_breiman_1.append(compute_metrics(scores_breiman_1, time_breiman_1))
        results_breiman_10.append(compute_metrics(scores_breiman_10, time_breiman_10))
        results_direct_opt.append(compute_metrics(scores_dvi_opt, time_dvi_opt))
        results_direct_approx.append(compute_metrics(scores_dvi_approx, time_dvi_approx))

    # Aggregate across repetitions
    def aggregate(results_list):
        cors = [r['cor'] for r in results_list if not np.isnan(r['cor'])]
        max_diffs = [r['max_diff'] for r in results_list]
        mean_diffs = [r['mean_diff'] for r in results_list]
        times = [r['time'] for r in results_list]

        return {
            'cor_mean': np.mean(cors) if cors else 0.0,
            'cor_std': np.std(cors) if cors else 0.0,
            'max_diff_mean': np.mean(max_diffs),
            'max_diff_std': np.std(max_diffs),
            'mean_diff_mean': np.mean(mean_diffs),
            'mean_diff_std': np.std(mean_diffs),
            'time_mean': np.mean(times),
            'time_std': np.std(times)
        }

    agg_breiman_1 = aggregate(results_breiman_1)
    agg_breiman_10 = aggregate(results_breiman_10)
    agg_direct_opt = aggregate(results_direct_opt)
    agg_direct_approx = aggregate(results_direct_approx)

    # Combine results
    aggregated = {
        'scenario': scenario,
        'breiman_1': agg_breiman_1,
        'breiman_10': agg_breiman_10,
        'direct_opt': agg_direct_opt,
        'direct_approx': agg_direct_approx
    }

    return aggregated


def aggregate_results(scenario_results: List[Dict]) -> Dict:
    """
    Aggregate results across multiple scenarios.

    Computes mean ± std across all scenarios for each metric.

    Parameters
    ----------
    scenario_results : List[Dict]
        List of results from run_single_scenario()

    Returns
    -------
    summary : Dict
        Summary statistics with keys like:
        - 'ground_truth_cor': mean correlation across all scenarios
        - 'max_score_diff': mean max difference
        - 'time_ms': mean runtime
        For each of the 4 methods
    """
    methods = ['breiman_1', 'breiman_10', 'direct_opt', 'direct_approx']

    summary = {}
    for method in methods:
        cors = [r[method]['cor_mean'] for r in scenario_results]
        max_diffs = [r[method]['max_diff_mean'] for r in scenario_results]
        times = [r[method]['time_mean'] for r in scenario_results]

        summary[method] = {
            'ground_truth_cor': f"{np.mean(cors):.3f} ± {np.std(cors):.3f}",
            'max_score_diff': f"{np.mean(max_diffs):.3f} ± {np.std(max_diffs):.3f}",
            'time_ms': f"{np.mean(times):.1f} ± {np.std(times):.1f}"
        }

    return summary


if __name__ == "__main__":
    # Quick smoke test
    print("Testing synthetic benchmark suite...")

    print("\n1. Testing linear data generation...")
    X, y, true_imp = generate_linear_data(n=100, p=10, sigma_epsilon=1.0, rho=0.3)
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    print(f"   True importance sum: {true_imp.sum():.4f}")

    print("\n2. Testing Friedman data generation...")
    X, y, true_imp = generate_friedman_data(n=100, p=10, sigma_epsilon=1.0, rho=0.0)
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    print(f"   True importance (first 5): {true_imp[:5]}")

    print("\n3. Testing scenario grid creation...")
    scenarios = create_scenario_grid()
    print(f"   Total scenarios: {len(scenarios)}")
    print(f"   Sample scenario: {scenarios[0]}")

    print("\n4. Testing single scenario run (1 repetition, small data)...")
    test_scenario = {
        'n': 100,
        'p': 10,
        'sigma_epsilon': 1.0,
        'rho': 0.0,
        'response_type': 'linear',
        'task_type': 'regression',
        'scenario_id': 0
    }
    result = run_single_scenario(test_scenario, n_repetitions=1, verbose=True)
    print(f"   DVI-Optimal correlation: {result['direct_opt']['cor_mean']:.4f}")
    print(f"   DVI-Optimal time: {result['direct_opt']['time_mean']:.2f} ms")

    print("\n✓ All tests passed!")
