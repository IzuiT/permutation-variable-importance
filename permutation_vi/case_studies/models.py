"""
Model factory for case study experiments.

This module provides functions to create and train models used in the paper's
case studies, including:
- TRUST (sparse linear regression)
- OLS (ordinary least squares)
- Random Forest
- Logistic Regression (regularized and unregularized)
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    average_precision_score, roc_auc_score
)
from typing import Dict, Tuple, Any, Optional


class RelaxedLasso:
    """
    Relaxed Lasso approximation for TRUST model.

    TRUST is a sparse linear regression model from the trust-free library.
    This class provides a sklearn-compatible approximation using:
    1. LassoCV to select features
    2. OLS refitting on selected features only

    This is equivalent to TRUST with depth=0 (linear model, no tree).

    References
    ----------
    Paper Section 5.2
    TRUST: https://github.com/adc-trust-ai/trust-free
    """

    def __init__(self, cv=5, random_state=42):
        """
        Initialize RelaxedLasso.

        Parameters
        ----------
        cv : int, default=5
            Number of cross-validation folds for LassoCV
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.cv = cv
        self.random_state = random_state
        self.lasso_ = None
        self.ols_ = None
        self.selected_features_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit the Relaxed Lasso model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        self : RelaxedLasso
            Fitted estimator
        """
        # Step 1: Fit Lasso to select features
        self.lasso_ = LassoCV(cv=self.cv, random_state=self.random_state, n_jobs=-1)
        self.lasso_.fit(X, y)

        # Step 2: Identify non-zero coefficients
        self.selected_features_ = np.where(self.lasso_.coef_ != 0)[0]

        # Step 3: Refit OLS on selected features
        self.coef_ = np.zeros(X.shape[1])
        if len(self.selected_features_) > 0:
            self.ols_ = LinearRegression()
            self.ols_.fit(X[:, self.selected_features_], y)
            self.coef_[self.selected_features_] = self.ols_.coef_
            self.intercept_ = self.ols_.intercept_
        else:
            # No features selected - use mean prediction
            self.intercept_ = np.mean(y)

        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if len(self.selected_features_) > 0:
            return self.ols_.predict(X[:, self.selected_features_])
        else:
            return np.full(X.shape[0], self.intercept_)


class TrustWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for trust-free TRUST so clone() and
    permutation_importance work. Delegates fit/predict to inner TRUST and
    inherits get_params/set_params from BaseEstimator.
    """

    def __init__(self, max_depth=0, **kwargs):
        super().__init__()
        from trust import TRUST
        self.max_depth = max_depth
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._trust_kwargs = dict(max_depth=max_depth, **kwargs)
        self._trust = TRUST(**self._trust_kwargs)

    def set_params(self, **params):
        super().set_params(**params)
        from trust import TRUST
        self._trust_kwargs = self.get_params(deep=False)
        self._trust = TRUST(**self._trust_kwargs)
        return self

    def fit(self, X, y):
        self._trust.fit(X, y)
        return self

    def predict(self, X):
        return self._trust.predict(X)

    def __getattr__(self, name):
        # Expose coef_ / feature_importances_ from inner TRUST for CV nonzero count
        if name in ('coef_', 'feature_importances_'):
            return getattr(self._trust, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def get_trust_model():
    """
    Get TRUST model or fallback to RelaxedLasso.

    Returns
    -------
    model : estimator
        TRUST model if available (wrapped for sklearn clone), else RelaxedLasso
    """
    try:
        from trust import TRUST
        TRUST  # ensure import works
        return TrustWrapper(max_depth=0)
    except ImportError:
        return RelaxedLasso()


def get_boston_models() -> Dict[str, Any]:
    """
    Get models for Boston HMDA regression experiments.

    Returns
    -------
    models : dict
        Dictionary with keys 'TRUST', 'OLS', 'RF'

    Notes
    -----
    Paper Section 5.2:
    - TRUST: Sparse linear regression (relaxed Lasso approximation)
    - OLS: Ordinary least squares baseline
    - RF: Random Forest (100 trees) as black-box benchmark
    """
    return {
        'TRUST': get_trust_model(),
        'OLS': LinearRegression(),
        'RF': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }


def get_german_models() -> Dict[str, Any]:
    """
    Get models for German Credit classification experiments.

    Returns
    -------
    models : dict
        Dictionary with keys 'l1_logistic', 'logistic', 'RF'

    Notes
    -----
    Paper Section 5.3:
    - l1_logistic: Strongly regularized (most coefficients zeroed)
    - logistic: Unregularized (all features used)
    - RF: Random Forest (100 trees) as benchmark
    """
    return {
        'l1_logistic': LogisticRegression(
            penalty='l1',
            solver='saga',  # saga supports l1 penalty
            C=0.01,  # Strong penalty (small C = high regularization)
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'logistic': LogisticRegression(
            penalty=None,  # No regularization
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'RF': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }


def cross_validate_regression(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    random_state: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Cross-validate regression model following paper protocol.

    Parameters
    ----------
    model : estimator
        Regression model with fit() and predict() methods
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Target values
    n_folds : int, default=10
        Number of CV folds
    random_state : int, default=42
        Random seed for row randomization

    Returns
    -------
    results : dict
        Dictionary with keys: 'r2', 'mae', 'nonzero'
        Each value is tuple (mean, std)

    Notes
    -----
    Paper protocol (Section 5.2):
    - Randomize rows before folding
    - 10-fold cross-validation
    - Report mean ± std
    """
    # Randomize rows (paper protocol)
    rng = np.random.RandomState(random_state)
    shuffle_idx = rng.permutation(len(X))
    X_shuffled = X[shuffle_idx]
    y_shuffled = y[shuffle_idx]

    # K-fold split (no shuffle needed, already randomized)
    kfold = KFold(n_splits=n_folds, shuffle=False)

    scores = {'r2': [], 'mae': [], 'nonzero': []}

    for train_idx, test_idx in kfold.split(X_shuffled):
        X_train, X_test = X_shuffled[train_idx], X_shuffled[test_idx]
        y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]

        # Clone model to avoid fitting same instance
        from sklearn.base import clone
        model_clone = clone(model)

        # Fit and predict
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)

        # Compute metrics
        scores['r2'].append(r2_score(y_test, y_pred))
        scores['mae'].append(mean_absolute_error(y_test, y_pred))

        # Count nonzero coefficients
        if hasattr(model_clone, 'coef_'):
            scores['nonzero'].append(np.count_nonzero(model_clone.coef_))
        elif hasattr(model_clone, 'feature_importances_'):
            # For tree-based models, count features with non-zero importance
            scores['nonzero'].append(np.count_nonzero(model_clone.feature_importances_ > 0))
        else:
            scores['nonzero'].append(X.shape[1])  # All features used

    # Return mean ± std
    return {
        'r2': (np.mean(scores['r2']), np.std(scores['r2'])),
        'mae': (np.mean(scores['mae']), np.std(scores['mae'])),
        'nonzero': (np.mean(scores['nonzero']), np.std(scores['nonzero']))
    }


def cross_validate_classification(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    random_state: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Cross-validate classification model following paper protocol.

    Parameters
    ----------
    model : estimator
        Classification model with fit() and predict_proba() methods
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Binary target (0 or 1)
    n_folds : int, default=10
        Number of CV folds
    random_state : int, default=42
        Random seed for row randomization

    Returns
    -------
    results : dict
        Dictionary with keys: 'pr_auc', 'roc_auc', 'nonzero'
        Each value is tuple (mean, std)

    Notes
    -----
    Paper protocol (Section 5.3):
    - Same CV procedure as regression
    - Metrics: PR-AUC, ROC-AUC, nonzero coefficients
    """
    # Randomize rows
    rng = np.random.RandomState(random_state)
    shuffle_idx = rng.permutation(len(X))
    X_shuffled = X[shuffle_idx]
    y_shuffled = y[shuffle_idx]

    # K-fold split
    kfold = KFold(n_splits=n_folds, shuffle=False)

    scores = {'pr_auc': [], 'roc_auc': [], 'nonzero': []}

    for train_idx, test_idx in kfold.split(X_shuffled):
        X_train, X_test = X_shuffled[train_idx], X_shuffled[test_idx]
        y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]

        # Clone model
        from sklearn.base import clone
        model_clone = clone(model)

        # Fit and predict probabilities
        model_clone.fit(X_train, y_train)
        y_pred_proba = model_clone.predict_proba(X_test)[:, 1]

        # Compute metrics
        scores['pr_auc'].append(average_precision_score(y_test, y_pred_proba))
        scores['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))

        # Count nonzero coefficients
        if hasattr(model_clone, 'coef_'):
            scores['nonzero'].append(np.count_nonzero(model_clone.coef_))
        elif hasattr(model_clone, 'feature_importances_'):
            scores['nonzero'].append(np.count_nonzero(model_clone.feature_importances_ > 0))
        else:
            scores['nonzero'].append(X.shape[1])

    # Return mean ± std
    return {
        'pr_auc': (np.mean(scores['pr_auc']), np.std(scores['pr_auc'])),
        'roc_auc': (np.mean(scores['roc_auc']), np.std(scores['roc_auc'])),
        'nonzero': (np.mean(scores['nonzero']), np.std(scores['nonzero']))
    }


if __name__ == "__main__":
    # Test model factory
    print("Testing model factory...")
    print("=" * 60)

    # Test Boston models
    print("\nBoston HMDA models:")
    boston_models = get_boston_models()
    for name, model in boston_models.items():
        print(f"  {name}: {type(model).__name__}")

    # Test German models
    print("\nGerman Credit models:")
    german_models = get_german_models()
    for name, model in german_models.items():
        print(f"  {name}: {type(model).__name__}")

    # Test RelaxedLasso on synthetic data
    print("\nTesting RelaxedLasso on synthetic data...")
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=20, n_informative=5, random_state=42)

    rlasso = RelaxedLasso()
    rlasso.fit(X, y)
    print(f"  Selected features: {len(rlasso.selected_features_)}/20")
    print(f"  Nonzero coefficients: {np.count_nonzero(rlasso.coef_)}")

    # Test cross-validation
    print("\nTesting cross-validation...")
    results = cross_validate_regression(LinearRegression(), X, y, n_folds=5)
    print(f"  R²: {results['r2'][0]:.3f} ± {results['r2'][1]:.3f}")
    print(f"  MAE: {results['mae'][0]:.3f} ± {results['mae'][1]:.3f}")

    print("\nAll tests passed!")
