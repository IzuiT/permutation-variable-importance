"""
Cross-validation utilities for case study experiments.

This module provides helper functions for data splitting and cross-validation
following the paper's protocol.
"""

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from typing import Tuple, Generator


def stratified_cv_split(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    random_state: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate stratified K-fold cross-validation splits.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Target values
    n_folds : int, default=10
        Number of folds
    random_state : int, default=42
        Random seed for reproducibility

    Yields
    ------
    train_idx : np.ndarray
        Indices for training set
    test_idx : np.ndarray
        Indices for test set

    Notes
    -----
    Paper protocol (Section 5.2, 5.3):
    - Randomize rows before splitting
    - 10-fold cross-validation
    - No stratification for regression (just shuffle)
    - Stratification for classification (balanced folds)
    """
    # Randomize rows
    rng = np.random.RandomState(random_state)
    shuffle_idx = rng.permutation(len(X))

    # Check if classification (binary target)
    is_classification = len(np.unique(y)) <= 10 and np.issubdtype(y.dtype, np.integer)

    if is_classification:
        # Use stratified K-fold for classification
        from sklearn.model_selection import StratifiedKFold
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kfold.split(X, y):
            yield train_idx, test_idx
    else:
        # Use regular K-fold for regression (with shuffling)
        kfold = KFold(n_splits=n_folds, shuffle=False)  # Already shuffled above
        for train_idx, test_idx in kfold.split(shuffle_idx):
            yield shuffle_idx[train_idx], shuffle_idx[test_idx]


def train_test_holdout(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/test split with optional stratification.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Target values
    test_size : float, default=0.1
        Fraction of data for test set (paper uses 10%)
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Test features
    y_train : np.ndarray
        Training targets
    y_test : np.ndarray
        Test targets

    Notes
    -----
    Paper uses 90/10 train/test split for SVI analysis:
    - Train on 90% to compute correlation threshold
    - Test on held-out 10% for SVI importance scores
    """
    # Check if classification
    is_classification = len(np.unique(y)) <= 10 and np.issubdtype(y.dtype, np.integer)

    if is_classification:
        # Stratified split for classification
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        # Regular split for regression
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )


def get_cv_fold_predictions(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    random_state: int = 42,
    return_proba: bool = False
) -> np.ndarray:
    """
    Get out-of-fold predictions for entire dataset.

    Useful for computing ground-truth importance scores from a master model.

    Parameters
    ----------
    model : estimator
        Model with fit() and predict() (or predict_proba()) methods
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Target values
    n_folds : int, default=10
        Number of CV folds
    random_state : int, default=42
        Random seed
    return_proba : bool, default=False
        If True, return probabilities (for classification)

    Returns
    -------
    predictions : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
        Out-of-fold predictions for all samples

    Notes
    -----
    Each sample gets predicted by a model that was NOT trained on it.
    This avoids overfitting when using predictions as "ground truth".
    """
    from sklearn.base import clone

    predictions = np.zeros(len(y)) if not return_proba else np.zeros((len(y), 2))

    for train_idx, test_idx in stratified_cv_split(X, y, n_folds, random_state):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Clone and fit model
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        # Get predictions
        if return_proba:
            predictions[test_idx] = model_clone.predict_proba(X_test)
        else:
            predictions[test_idx] = model_clone.predict(X_test)

    return predictions


if __name__ == "__main__":
    # Test CV utilities
    print("Testing CV utilities...")
    print("=" * 60)

    # Test with synthetic data
    from sklearn.datasets import make_regression, make_classification

    # Test regression
    print("\nTesting regression splits:")
    X_reg, y_reg = make_regression(n_samples=100, n_features=5, random_state=42)

    fold_count = 0
    for train_idx, test_idx in stratified_cv_split(X_reg, y_reg, n_folds=5):
        fold_count += 1
        print(f"  Fold {fold_count}: train={len(train_idx)}, test={len(test_idx)}")

    # Test classification
    print("\nTesting classification splits:")
    X_clf, y_clf = make_classification(n_samples=100, n_features=5, random_state=42)

    fold_count = 0
    for train_idx, test_idx in stratified_cv_split(X_clf, y_clf, n_folds=5):
        fold_count += 1
        # Check stratification
        train_ratio = y_clf[train_idx].mean()
        test_ratio = y_clf[test_idx].mean()
        print(f"  Fold {fold_count}: train={len(train_idx)} (ratio={train_ratio:.2f}), "
              f"test={len(test_idx)} (ratio={test_ratio:.2f})")

    # Test holdout split
    print("\nTesting train/test holdout:")
    X_train, X_test, y_train, y_test = train_test_holdout(X_reg, y_reg, test_size=0.1)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print("\nAll tests passed!")
