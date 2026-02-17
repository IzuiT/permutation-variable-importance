"""
Direct Variable Importance (DVI) using single optimal permutation.

This module implements the core contribution of the paper:
replacing multiple random permutations with a single deterministic
optimal permutation for variable importance estimation.
"""

import numpy as np
from typing import Optional, Literal, Callable


class DirectVariableImportance:
    """
    Direct Variable Importance using single deterministic permutation.

    This class implements model-instance variable importance (VI) using a single
    optimal permutation instead of multiple random permutations. It provides:
    - Deterministic, reproducible scores
    - 10-100x speedup over traditional methods
    - Comparable or better accuracy

    Parameters
    ----------
    permutation_type : {'optimal', 'approximate'}, default='optimal'
        Type of permutation to use:
        - 'optimal': Rank-shift by ⌊n/2⌋ (O(n log n), most accurate)
        - 'approximate': Index-shift by ⌊n/2⌋ (O(n), faster approximation)

    scoring_metric : {'mae', 'mse', 'rmse'}, default='mae'
        Metric to measure prediction change:
        - 'mae': Mean Absolute Error (recommended by paper)
        - 'mse': Mean Squared Error (traditional choice)
        - 'rmse': Root Mean Squared Error

    Attributes
    ----------
    importances_ : np.ndarray of shape (n_features,)
        Normalized importance scores (sum to 1) after calling fit()

    feature_names_ : list of str or None
        Feature names if provided during fit()

    n_features_ : int
        Number of features

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=10,
    ...                        n_informative=5, random_state=42)
    >>> model = RandomForestRegressor(random_state=42).fit(X, y)
    >>> dvi = DirectVariableImportance(permutation_type='optimal')
    >>> scores = dvi.fit(model, X)
    >>> print(f"Top feature importance: {scores.max():.4f}")

    References
    ----------
    Paper: "One Permutation Is All You Need: Fast, Reliable Variable
           Importance and Model Stress-Testing" (arXiv:2512.13892v2)
    Section 2: Theoretical justification (Propositions 1-2)
    Section 3: Implementation details
    """

    def __init__(
        self,
        permutation_type: Literal['optimal', 'approximate'] = 'optimal',
        scoring_metric: Literal['mae', 'mse', 'rmse'] = 'mae'
    ):
        if permutation_type not in ['optimal', 'approximate']:
            raise ValueError(
                f"permutation_type must be 'optimal' or 'approximate', "
                f"got {permutation_type}"
            )
        if scoring_metric not in ['mae', 'mse', 'rmse']:
            raise ValueError(
                f"scoring_metric must be 'mae', 'mse', or 'rmse', "
                f"got {scoring_metric}"
            )

        self.permutation_type = permutation_type
        self.scoring_metric = scoring_metric

        # To be set during fit()
        self.importances_ = None
        self.feature_names_ = None
        self.n_features_ = None

    def _optimal_permutation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply optimal rank-shift permutation by ⌊n/2⌋.

        This is the permutation proven optimal in Proposition 1.
        It maximizes the minimum circular displacement.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples,)
            Feature values

        Returns
        -------
        x_permuted : np.ndarray of shape (n_samples,)
            Permuted feature values

        References
        ----------
        Paper Proposition 1: Proof of optimality
        """
        n = len(x)
        shift = n // 2

        # Get ranks (argsort twice gives ranks)
        ranks = np.argsort(np.argsort(x))

        # Shift ranks circularly
        shifted_ranks = (ranks + shift) % n

        # Map back to original values via sorted positions
        sorted_x = np.sort(x)
        return sorted_x[shifted_ranks]

    def _approximate_permutation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply approximate index-shift permutation by ⌊n/2⌋.

        This is faster than optimal (O(n) vs O(n log n)) but slightly
        less accurate. The paper shows minimal practical difference.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples,)
            Feature values

        Returns
        -------
        x_permuted : np.ndarray of shape (n_samples,)
            Permuted feature values
        """
        n = len(x)
        shift = n // 2
        return np.roll(x, shift)

    def _compute_score(
        self,
        y_pred: np.ndarray,
        y_pred_perm: np.ndarray
    ) -> float:
        """
        Compute prediction difference score.

        Parameters
        ----------
        y_pred : np.ndarray
            Original predictions
        y_pred_perm : np.ndarray
            Predictions after permutation

        Returns
        -------
        score : float
            Non-negative score measuring prediction change

        References
        ----------
        Paper Section 3.1.2: Evaluation metrics
        """
        diff = y_pred - y_pred_perm

        if self.scoring_metric == 'mae':
            return np.mean(np.abs(diff))
        elif self.scoring_metric == 'mse':
            return np.mean(diff ** 2)
        elif self.scoring_metric == 'rmse':
            return np.sqrt(np.mean(diff ** 2))
        else:
            raise ValueError(f"Unknown scoring metric: {self.scoring_metric}")

    def _permute_feature(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the selected permutation strategy.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples,)
            Feature values

        Returns
        -------
        x_permuted : np.ndarray of shape (n_samples,)
            Permuted feature values
        """
        if self.permutation_type == 'optimal':
            return self._optimal_permutation(x)
        else:  # 'approximate'
            return self._approximate_permutation(x)

    def fit(
        self,
        model,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ) -> np.ndarray:
        """
        Compute direct variable importance scores.

        Parameters
        ----------
        model : estimator or callable
            Trained model with predict() or predict_proba(), or a callable
            (X) -> predictions used for ground-truth DVI (e.g. out-of-fold predictor)
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix (typically test set)
        y : np.ndarray of shape (n_samples,), optional
            Target values (not used for model-instance VI, included for API compatibility)
        feature_names : list of str, optional
            Feature names for reference

        Returns
        -------
        importances : np.ndarray of shape (n_features,)
            Normalized importance scores (sum to 1)

        Notes
        -----
        Following the paper's Definition 1, we compute importance as:
        d_k = (1/nq) Σ_i d(f_M(X)_i, f_M(X'_k)_i)
        where X'_k has feature k permuted.

        The paper notes that using test data is conventional but not required -
        we estimate the model's reliance on each feature, which is independent
        of train/test split.

        References
        ----------
        Paper Definition 1: Direct Variable Importance
        Paper Section 3: Methods
        """
        # Input validation
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.feature_names_ = feature_names

        # Support both estimators and callable prediction functions (e.g. for ground-truth DVI)
        use_predict_fn = callable(model) and not hasattr(model, 'predict')

        def get_predictions(X_arr):
            if use_predict_fn:
                out = model(X_arr)
            else:
                if hasattr(model, 'predict_proba'):
                    out = model.predict_proba(X_arr)
                else:
                    out = model.predict(X_arr)
            out = np.asarray(out)
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            return out

        # Get baseline predictions
        y_pred = get_predictions(X)

        # Compute importance for each feature
        scores = np.zeros(n_features)

        for j in range(n_features):
            # Create copy with feature j permuted
            X_perm = X.copy()
            X_perm[:, j] = self._permute_feature(X[:, j])

            # Get perturbed predictions
            y_pred_perm = get_predictions(X_perm)

            # Compute score (handles both regression and classification)
            scores[j] = self._compute_score(y_pred, y_pred_perm)

        # Normalize to sum to 1
        if scores.sum() > 0:
            scores = scores / scores.sum()
        else:
            # All scores are zero (model is constant)
            scores = np.ones(n_features) / n_features

        self.importances_ = scores
        return scores

    def get_feature_importance(self, feature_idx: int) -> float:
        """
        Get importance score for a specific feature.

        Parameters
        ----------
        feature_idx : int
            Feature index

        Returns
        -------
        importance : float
            Normalized importance score
        """
        if self.importances_ is None:
            raise ValueError("Call fit() before accessing importances")
        return self.importances_[feature_idx]

    def get_top_features(self, n: int = 5) -> list:
        """
        Get indices and scores of top n most important features.

        Parameters
        ----------
        n : int, default=5
            Number of top features to return

        Returns
        -------
        top_features : list of tuple
            List of (feature_idx, score) pairs, sorted by importance
        """
        if self.importances_ is None:
            raise ValueError("Call fit() before accessing importances")

        n = min(n, len(self.importances_))
        top_idx = np.argsort(self.importances_)[::-1][:n]

        if self.feature_names_ is not None:
            return [(self.feature_names_[i], self.importances_[i]) for i in top_idx]
        else:
            return [(i, self.importances_[i]) for i in top_idx]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DirectVariableImportance("
            f"permutation_type='{self.permutation_type}', "
            f"scoring_metric='{self.scoring_metric}')"
        )
