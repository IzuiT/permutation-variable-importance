"""
Example: Direct Variable Importance (DVI)

This example demonstrates the basic usage of DirectVariableImportance
on synthetic regression and classification tasks.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
from permutation_vi import DirectVariableImportance


def example_regression():
    """Demonstrate DVI on a regression task."""
    print("=" * 60)
    print("Example 1: Regression Task")
    print("=" * 60)

    # Generate synthetic data
    # 5 informative features, 5 noise features
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    print("\nTraining Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Train R²: {model.score(X_train, y_train):.4f}")
    print(f"Test R²:  {model.score(X_test, y_test):.4f}")

    # Compute DVI with optimal permutation
    print("\nComputing Direct Variable Importance (optimal)...")
    dvi_optimal = DirectVariableImportance(
        permutation_type='optimal',
        scoring_metric='mae'
    )
    scores_optimal = dvi_optimal.fit(model, X_test)

    # Compute DVI with approximate permutation
    print("Computing Direct Variable Importance (approximate)...")
    dvi_approx = DirectVariableImportance(
        permutation_type='approximate',
        scoring_metric='mae'
    )
    scores_approx = dvi_approx.fit(model, X_test)

    # Display results
    print("\nFeature Importances:")
    print(f"{'Feature':<10} {'Optimal':<12} {'Approximate':<12}")
    print("-" * 34)
    for i in range(10):
        print(f"Feature {i:<2} {scores_optimal[i]:>10.4f}  {scores_approx[i]:>10.4f}")

    # Top features
    print("\nTop 5 Features (Optimal):")
    top_features = dvi_optimal.get_top_features(n=5)
    for idx, score in top_features:
        print(f"  Feature {idx}: {score:.4f}")

    # Verify scores sum to 1
    print(f"\nSum of scores (optimal): {scores_optimal.sum():.6f}")
    print(f"Sum of scores (approx):  {scores_approx.sum():.6f}")

    # Correlation between methods
    corr = np.corrcoef(scores_optimal, scores_approx)[0, 1]
    print(f"\nCorrelation between optimal and approximate: {corr:.4f}")


def example_classification():
    """Demonstrate DVI on a classification task."""
    print("\n\n")
    print("=" * 60)
    print("Example 2: Classification Task")
    print("=" * 60)

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=8,
        n_redundant=3,
        n_classes=2,
        random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test Accuracy:  {model.score(X_test, y_test):.4f}")

    # Compute DVI
    print("\nComputing Direct Variable Importance...")
    dvi = DirectVariableImportance(
        permutation_type='optimal',
        scoring_metric='mae'  # MAE on probability differences
    )
    scores = dvi.fit(model, X_test)

    # Display top features
    print("\nTop 10 Features:")
    top_features = dvi.get_top_features(n=10)
    for idx, score in top_features:
        print(f"  Feature {idx:2d}: {score:.4f}")

    # Verify determinism
    print("\nVerifying determinism (running fit() again)...")
    scores_again = dvi.fit(model, X_test)
    max_diff = np.max(np.abs(scores - scores_again))
    print(f"Maximum difference: {max_diff:.10f}")
    if max_diff == 0:
        print("✓ Results are perfectly deterministic!")


def example_different_metrics():
    """Compare different scoring metrics."""
    print("\n\n")
    print("=" * 60)
    print("Example 3: Comparing Scoring Metrics")
    print("=" * 60)

    # Generate data
    X, y = make_regression(
        n_samples=500,
        n_features=8,
        n_informative=4,
        random_state=42
    )

    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Try different metrics
    metrics = ['mae', 'mse', 'rmse']
    results = {}

    for metric in metrics:
        dvi = DirectVariableImportance(
            permutation_type='optimal',
            scoring_metric=metric
        )
        results[metric] = dvi.fit(model, X)

    # Display results
    print("\nFeature Importances by Metric:")
    print(f"{'Feature':<10} {'MAE':<12} {'MSE':<12} {'RMSE':<12}")
    print("-" * 46)
    for i in range(8):
        print(f"Feature {i:<2} {results['mae'][i]:>10.4f}  "
              f"{results['mse'][i]:>10.4f}  {results['rmse'][i]:>10.4f}")

    # Correlations between metrics
    print("\nCorrelations between metrics:")
    print(f"MAE vs MSE:  {np.corrcoef(results['mae'], results['mse'])[0,1]:.4f}")
    print(f"MAE vs RMSE: {np.corrcoef(results['mae'], results['rmse'])[0,1]:.4f}")
    print(f"MSE vs RMSE: {np.corrcoef(results['mse'], results['rmse'])[0,1]:.4f}")


if __name__ == "__main__":
    # Run all examples
    example_regression()
    example_classification()
    example_different_metrics()

    print("\n\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
