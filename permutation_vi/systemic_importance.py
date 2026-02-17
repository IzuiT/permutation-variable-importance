"""
Systemic Variable Importance (SVI) with correlation propagation.

This module extends Direct Variable Importance to account for feature
correlations, enabling stress-testing and fairness auditing.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Optional, Literal, Tuple
from .direct_importance import DirectVariableImportance
from .utils import estimate_correlation_threshold


class SystemicVariableImportance(DirectVariableImportance):
    """
    Systemic Variable Importance with correlation propagation.

    This class extends DirectVariableImportance to account for feature
    correlations. It reveals how perturbations propagate through correlated
    features, enabling:
    - Detection of indirect reliance on protected attributes
    - Model stress-testing under correlated shocks
    - Fairness auditing via proxy variable detection

    Parameters
    ----------
    permutation_type : {'optimal', 'approximate'}, default='optimal'
        Type of permutation (inherited from DirectVariableImportance)

    scoring_metric : {'mae', 'mse', 'rmse'}, default='mae'
        Metric to measure prediction change

    correlation_method : {'spearman', 'pearson'}, default='spearman'
        Method for computing feature correlations:
        - 'spearman': Rank correlation (recommended, handles non-linear monotonic)
        - 'pearson': Linear correlation

    alpha : float, default=0.01
        Significance level for correlation threshold (FWER control)
        Lower values = more conservative (fewer correlations considered significant)

    Attributes
    ----------
    systemic_importances_ : np.ndarray of shape (n_features,)
        Normalized systemic importance scores (sum to 1)

    direct_importances_ : np.ndarray of shape (n_features,)
        Direct importance component (same as DirectVariableImportance)

    indirect_importances_ : np.ndarray of shape (n_features,)
        Indirect importance component (network amplification/dampening)

    correlation_matrix_ : np.ndarray of shape (n_features, n_features)
        Feature correlation matrix

    correlation_threshold_ : float
        Estimated threshold τ for significant correlations

    significant_correlations_ : np.ndarray of shape (n_features, n_features)
        Binary matrix: 1 if |correlation| > threshold, 0 otherwise

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=10,
    ...                            n_informative=5, random_state=42)
    >>> model = LogisticRegression(random_state=42).fit(X, y)
    >>> svi = SystemicVariableImportance(alpha=0.01)
    >>> systemic, direct, indirect = svi.fit(model, X, return_decomposition=True)
    >>> # Check if feature 0 has indirect importance (proxy effects)
    >>> print(f"Feature 0 indirect importance: {indirect[0]:.4f}")

    References
    ----------
    Paper Section 5: Systemic Variable Importance
    Paper Section 5.1: Statistical calibration of propagation threshold
    Paper Sections 5.2-5.3: Case studies (HMDA, German Credit)
    """

    def __init__(
        self,
        permutation_type: Literal['optimal', 'approximate'] = 'optimal',
        scoring_metric: Literal['mae', 'mse', 'rmse'] = 'mae',
        correlation_method: Literal['spearman', 'pearson'] = 'spearman',
        alpha: float = 0.01
    ):
        super().__init__(permutation_type=permutation_type, scoring_metric=scoring_metric)

        if correlation_method not in ['spearman', 'pearson']:
            raise ValueError(
                f"correlation_method must be 'spearman' or 'pearson', "
                f"got {correlation_method}"
            )
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.correlation_method = correlation_method
        self.alpha = alpha

        # To be set during fit()
        self.systemic_importances_ = None
        self.direct_importances_ = None
        self.indirect_importances_ = None
        self.correlation_matrix_ = None
        self.correlation_threshold_ = None
        self.significant_correlations_ = None

    def _compute_correlation_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise feature correlations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix

        Returns
        -------
        corr_matrix : np.ndarray of shape (n_features, n_features)
            Correlation matrix
        """
        n_features = X.shape[1]
        corr_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif i < j:
                    if self.correlation_method == 'spearman':
                        corr, _ = spearmanr(X[:, i], X[:, j])
                    else:  # pearson
                        corr, _ = pearsonr(X[:, i], X[:, j])

                    # Handle NaN (constant columns)
                    if np.isnan(corr):
                        corr = 0.0

                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                else:
                    continue  # Already filled by symmetry

        return corr_matrix

    def _propagate_perturbations(
        self,
        X: np.ndarray,
        model,
        direct_scores: np.ndarray
    ) -> np.ndarray:
        """
        Propagate perturbations through correlation network.

        For each feature k, we compute its systemic importance by:
        1. Starting with its direct perturbation
        2. Adding propagated perturbations from correlated features

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix
        model : estimator
            Trained model
        direct_scores : np.ndarray of shape (n_features,)
            Direct importance scores (unnormalized)

        Returns
        -------
        systemic_scores : np.ndarray of shape (n_features,)
            Raw systemic importance scores (before normalization)

        Notes
        -----
        The propagation rule from the paper (Section 5):
        x'_k = x_k + Σ_j (x'_j - x_j) * cor(x_k, x_j)  for |cor(k,j)| > τ

        References
        ----------
        Paper Section 5, equation (16)
        """
        n_samples, n_features = X.shape
        systemic_scores = np.zeros(n_features)

        # Get baseline predictions
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X)
        else:
            y_pred = model.predict(X)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # For each feature k, compute systemic effect
        for k in range(n_features):
            # Start with original data
            X_systemic = X.copy()

            # Apply direct perturbation to feature k
            X_systemic[:, k] = self._permute_feature(X[:, k])

            # Propagate perturbations from other features
            for j in range(n_features):
                if j == k:
                    continue

                # Check if correlation is significant
                if not self.significant_correlations_[k, j]:
                    continue

                # Compute perturbation for feature j
                perturbation_j = self._permute_feature(X[:, j]) - X[:, j]

                # Propagate to feature k weighted by correlation
                X_systemic[:, k] += self.correlation_matrix_[k, j] * perturbation_j

            # Get predictions with full propagation
            if hasattr(model, 'predict_proba'):
                y_pred_systemic = model.predict_proba(X_systemic)
            else:
                y_pred_systemic = model.predict(X_systemic)

            y_pred_systemic = np.asarray(y_pred_systemic)
            if y_pred_systemic.ndim == 1:
                y_pred_systemic = y_pred_systemic.reshape(-1, 1)

            # Compute systemic score
            systemic_scores[k] = self._compute_score(y_pred, y_pred_systemic)

        return systemic_scores

    def fit(
        self,
        model,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        return_decomposition: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute systemic variable importance scores.

        Parameters
        ----------
        model : estimator
            Trained model with predict() or predict_proba() method
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, optional
            Target values (not used, included for API compatibility)
        feature_names : list of str, optional
            Feature names for reference
        return_decomposition : bool, default=True
            If True, return (systemic, direct, indirect) tuple
            If False, return only systemic scores

        Returns
        -------
        systemic_importances : np.ndarray of shape (n_features,)
            Normalized systemic importance scores

        direct_importances : np.ndarray of shape (n_features,), optional
            Direct component (returned if return_decomposition=True)

        indirect_importances : np.ndarray of shape (n_features,), optional
            Indirect component (returned if return_decomposition=True)

        Notes
        -----
        The decomposition follows equation (18) from the paper:
        s_k = d_k + i_k
        where s_k is systemic, d_k is direct, i_k is indirect importance.

        References
        ----------
        Paper Section 5: Systemic Variable Importance
        Paper equations (16)-(18)
        """
        # Input validation
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.feature_names_ = feature_names

        # Step 1: Compute direct importance (using parent class)
        direct_scores_normalized = super().fit(model, X, y, feature_names)
        self.direct_importances_ = direct_scores_normalized.copy()

        # We need unnormalized scores for propagation
        # Recompute them without normalization
        direct_scores_raw = np.zeros(n_features)
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X)
        else:
            y_pred = model.predict(X)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        for j in range(n_features):
            X_perm = X.copy()
            X_perm[:, j] = self._permute_feature(X[:, j])
            if hasattr(model, 'predict_proba'):
                y_pred_perm = model.predict_proba(X_perm)
            else:
                y_pred_perm = model.predict(X_perm)
            y_pred_perm = np.asarray(y_pred_perm)
            if y_pred_perm.ndim == 1:
                y_pred_perm = y_pred_perm.reshape(-1, 1)
            direct_scores_raw[j] = self._compute_score(y_pred, y_pred_perm)

        # Step 2: Compute correlation matrix
        self.correlation_matrix_ = self._compute_correlation_matrix(X)

        # Step 3: Estimate correlation threshold
        self.correlation_threshold_ = estimate_correlation_threshold(
            X,
            alpha=self.alpha,
            method=self.correlation_method
        )

        # Step 4: Build significant correlation network
        self.significant_correlations_ = (
            np.abs(self.correlation_matrix_) > self.correlation_threshold_
        )
        # Set diagonal to False (no self-correlation propagation)
        np.fill_diagonal(self.significant_correlations_, False)

        # Step 5: Propagate perturbations
        systemic_scores_raw = self._propagate_perturbations(
            X, model, direct_scores_raw
        )

        # Step 6: Normalize systemic scores
        if systemic_scores_raw.sum() > 0:
            systemic_scores = systemic_scores_raw / systemic_scores_raw.sum()
        else:
            systemic_scores = np.ones(n_features) / n_features

        self.systemic_importances_ = systemic_scores

        # Step 7: Compute indirect component
        # Need to align direct and systemic on same scale
        # Since both are normalized, we compute indirect as difference
        # But first denormalize to get proper decomposition
        if direct_scores_raw.sum() > 0:
            direct_normalized = direct_scores_raw / direct_scores_raw.sum()
        else:
            direct_normalized = np.ones(n_features) / n_features

        if systemic_scores_raw.sum() > 0:
            systemic_normalized = systemic_scores_raw / systemic_scores_raw.sum()
        else:
            systemic_normalized = np.ones(n_features) / n_features

        # Indirect = Systemic - Direct (on normalized scale)
        indirect_scores = systemic_normalized - direct_normalized
        self.indirect_importances_ = indirect_scores

        # Update the parent class importances to systemic
        self.importances_ = systemic_scores

        if return_decomposition:
            return systemic_scores, direct_normalized, indirect_scores
        else:
            return systemic_scores

    def get_correlation_network(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Get binary correlation network (adjacency matrix).

        Parameters
        ----------
        threshold : float, optional
            Custom threshold. If None, use self.correlation_threshold_

        Returns
        -------
        network : np.ndarray of shape (n_features, n_features)
            Binary adjacency matrix: 1 if |correlation| > threshold
        """
        if self.correlation_matrix_ is None:
            raise ValueError("Call fit() first")

        if threshold is None:
            threshold = self.correlation_threshold_

        network = np.abs(self.correlation_matrix_) > threshold
        np.fill_diagonal(network, False)
        return network.astype(int)

    def get_feature_proxies(self, feature_idx: int, return_correlations: bool = False):
        """
        Get significant proxy features for a given feature.

        This is useful for fairness auditing: if feature_idx is a protected
        attribute, this returns features that could serve as proxies.

        Parameters
        ----------
        feature_idx : int
            Feature index
        return_correlations : bool, default=False
            If True, return (proxy_indices, correlations) tuple

        Returns
        -------
        proxy_indices : np.ndarray
            Indices of features significantly correlated with feature_idx

        correlations : np.ndarray, optional
            Correlation values (if return_correlations=True)
        """
        if self.significant_correlations_ is None:
            raise ValueError("Call fit() first")

        proxies = np.where(self.significant_correlations_[feature_idx])[0]

        if return_correlations:
            corr_values = self.correlation_matrix_[feature_idx, proxies]
            return proxies, corr_values
        else:
            return proxies

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SystemicVariableImportance("
            f"permutation_type='{self.permutation_type}', "
            f"scoring_metric='{self.scoring_metric}', "
            f"correlation_method='{self.correlation_method}', "
            f"alpha={self.alpha})"
        )
