"""
Example: Systemic Variable Importance (SVI)

This example demonstrates how SVI can detect indirect feature reliance
through correlation networks, useful for fairness auditing and stress-testing.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
from permutation_vi import SystemicVariableImportance, DirectVariableImportance


def create_correlated_features(n_samples=1000, random_state=42):
    """
    Create a dataset with intentional correlations.

    We'll create a protected attribute (feature 0) that's correlated
    with several other features (proxies), simulating a fairness scenario.
    """
    rng = np.random.RandomState(random_state)

    # Protected attribute (e.g., sensitive demographic)
    protected = rng.binomial(1, 0.5, n_samples)

    # Create proxy features correlated with protected attribute
    # Proxy 1: Strongly correlated
    proxy1 = protected + 0.3 * rng.randn(n_samples)

    # Proxy 2: Moderately correlated
    proxy2 = 0.6 * protected + 0.4 * rng.randn(n_samples)

    # Proxy 3: Weakly correlated
    proxy3 = 0.3 * protected + 0.7 * rng.randn(n_samples)

    # Independent features (noise)
    noise1 = rng.randn(n_samples)
    noise2 = rng.randn(n_samples)
    noise3 = rng.randn(n_samples)

    # Create target that depends on proxies but NOT directly on protected
    # This simulates a model that uses proxies instead of protected attribute
    y = (2 * proxy1 + 1.5 * proxy2 + noise1 +
         0.5 * noise2 + 0.8 * noise3)

    # Convert to binary classification
    y_binary = (y > np.median(y)).astype(int)

    # Combine features
    X = np.column_stack([
        protected,  # Feature 0: Protected attribute
        proxy1,     # Feature 1: Strong proxy
        proxy2,     # Feature 2: Moderate proxy
        proxy3,     # Feature 3: Weak proxy
        noise1,     # Feature 4: Noise
        noise2,     # Feature 5: Noise (slight signal)
        noise3      # Feature 6: Noise (slight signal)
    ])

    feature_names = [
        'Protected', 'Proxy1', 'Proxy2', 'Proxy3',
        'Noise1', 'Noise2', 'Noise3'
    ]

    return X, y_binary, feature_names


def example_fairness_audit():
    """
    Demonstrate SVI for fairness auditing.

    We'll train a model WITHOUT the protected attribute and show
    that SVI can detect indirect reliance through proxies.
    """
    print("=" * 60)
    print("Example 1: Fairness Auditing with SVI")
    print("=" * 60)

    # Create correlated dataset
    X, y, feature_names = create_correlated_features(n_samples=1000)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Scenario 1: Model using ALL features (including protected)
    print("\nScenario 1: Model with protected attribute")
    print("-" * 60)

    model_with_protected = LogisticRegression(random_state=42, max_iter=1000)
    model_with_protected.fit(X_train, y_train)
    print(f"Test Accuracy: {model_with_protected.score(X_test, y_test):.4f}")

    # Compute systemic importance
    svi = SystemicVariableImportance(
        permutation_type='optimal',
        scoring_metric='mae',
        correlation_method='spearman',
        alpha=0.01
    )
    systemic, direct, indirect = svi.fit(
        model_with_protected, X_test,
        feature_names=feature_names,
        return_decomposition=True
    )

    print("\nImportance Decomposition:")
    print(f"{'Feature':<12} {'Systemic':<12} {'Direct':<12} {'Indirect':<12}")
    print("-" * 48)
    for i, name in enumerate(feature_names):
        print(f"{name:<12} {systemic[i]:>10.4f}  {direct[i]:>10.4f}  {indirect[i]:>10.4f}")

    print(f"\nProtected attribute importance:")
    print(f"  Direct:   {direct[0]:.4f}")
    print(f"  Indirect: {indirect[0]:.4f}")
    print(f"  Systemic: {systemic[0]:.4f}")
    print(f"  Multiplier: {systemic[0] / direct[0] if direct[0] > 0 else 'N/A':.2f}x")

    # Scenario 2: Model WITHOUT protected attribute (fairness attempt)
    print("\n\nScenario 2: Model without protected attribute")
    print("-" * 60)

    # Remove protected attribute (column 0)
    X_train_fair = np.delete(X_train, 0, axis=1)
    X_test_fair = np.delete(X_test, 0, axis=1)
    feature_names_fair = feature_names[1:]

    model_fair = LogisticRegression(random_state=42, max_iter=1000)
    model_fair.fit(X_train_fair, y_train)
    print(f"Test Accuracy: {model_fair.score(X_test_fair, y_test):.4f}")

    # Compute DVI (traditional)
    print("\nTraditional Direct Variable Importance:")
    dvi = DirectVariableImportance(permutation_type='optimal')
    direct_only = dvi.fit(model_fair, X_test_fair, feature_names=feature_names_fair)

    for i, name in enumerate(feature_names_fair):
        print(f"  {name:<12}: {direct_only[i]:.4f}")

    # Now audit for indirect protected attribute reliance
    # We need to reconstruct X_test with protected attribute for correlation analysis
    print("\n\nFairness Audit: Checking indirect reliance on protected attribute")
    print("-" * 60)

    # For SVI analysis, we need correlations with protected attribute
    # We'll create a wrapper that adds back the protected column for correlation
    # but model still doesn't use it directly

    # Compute correlation threshold
    from permutation_vi.utils import estimate_correlation_threshold
    threshold = estimate_correlation_threshold(X_test, alpha=0.01)
    print(f"\nCorrelation threshold (α=0.01): {threshold:.4f}")

    # Find proxies of protected attribute
    from scipy.stats import spearmanr
    print("\nFeatures correlated with protected attribute:")
    for i, name in enumerate(feature_names[1:], 1):
        corr, _ = spearmanr(X_test[:, 0], X_test[:, i])
        significant = "***" if abs(corr) > threshold else ""
        print(f"  {name:<12}: ρ = {corr:>6.3f} {significant}")

    print("\n*** indicates statistically significant correlation")
    print("These features could serve as proxies for the protected attribute!")


def example_stress_testing():
    """
    Demonstrate SVI for model stress-testing.

    Show how correlated feature shocks propagate through the system.
    """
    print("\n\n")
    print("=" * 60)
    print("Example 2: Model Stress-Testing with SVI")
    print("=" * 60)

    # Create data with feature blocks
    rng = np.random.RandomState(42)
    n_samples = 800

    # Block 1: Highly correlated features (e.g., economic indicators)
    z1 = rng.randn(n_samples)
    econ1 = z1 + 0.2 * rng.randn(n_samples)
    econ2 = z1 + 0.2 * rng.randn(n_samples)
    econ3 = z1 + 0.2 * rng.randn(n_samples)

    # Block 2: Moderately correlated features
    z2 = rng.randn(n_samples)
    demo1 = z2 + 0.5 * rng.randn(n_samples)
    demo2 = z2 + 0.5 * rng.randn(n_samples)

    # Independent features
    indep1 = rng.randn(n_samples)
    indep2 = rng.randn(n_samples)

    X = np.column_stack([econ1, econ2, econ3, demo1, demo2, indep1, indep2])
    y = 2 * econ1 + 1.5 * econ2 + demo1 + 0.5 * indep1 + rng.randn(n_samples)

    feature_names = [
        'Econ1', 'Econ2', 'Econ3',
        'Demo1', 'Demo2',
        'Indep1', 'Indep2'
    ]

    # Train model
    model = LinearRegression()
    model.fit(X, y)
    print(f"\nModel R²: {model.score(X, y):.4f}")

    # Compare DVI vs SVI
    print("\n" + "=" * 60)
    print("Direct vs Systemic Importance")
    print("=" * 60)

    svi = SystemicVariableImportance(
        permutation_type='optimal',
        scoring_metric='mae',
        alpha=0.01
    )
    systemic, direct, indirect = svi.fit(
        model, X,
        feature_names=feature_names,
        return_decomposition=True
    )

    print(f"\n{'Feature':<10} {'Direct':<12} {'Systemic':<12} {'Indirect':<12} {'Amplif.':<10}")
    print("-" * 54)
    for i, name in enumerate(feature_names):
        amplification = systemic[i] / direct[i] if direct[i] > 0.001 else 1.0
        print(f"{name:<10} {direct[i]:>10.4f}  {systemic[i]:>10.4f}  "
              f"{indirect[i]:>10.4f}  {amplification:>8.2f}x")

    print("\nInterpretation:")
    print("- Amplification > 1: Feature importance amplified by correlations")
    print("- Amplification < 1: Feature importance dampened by correlations")
    print("- High indirect scores: Feature has many significant proxies")

    # Show correlation network
    print("\n" + "=" * 60)
    print("Correlation Network")
    print("=" * 60)
    print(f"Threshold: {svi.correlation_threshold_:.4f}")

    network = svi.get_correlation_network()
    print("\nSignificant correlations (above threshold):")
    for i, name_i in enumerate(feature_names):
        for j, name_j in enumerate(feature_names):
            if network[i, j] == 1:
                corr_val = svi.correlation_matrix_[i, j]
                print(f"  {name_i:<10} ↔ {name_j:<10}: ρ = {corr_val:>6.3f}")


if __name__ == "__main__":
    # Run examples
    example_fairness_audit()
    example_stress_testing()

    print("\n\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
