"""
Visualization utilities for case study results.

This module provides functions to create publication-ready figures
matching the paper's style.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from typing import List, Optional, Dict
from pathlib import Path


def plot_correlation_heatmap(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float,
    save_path: Optional[str] = None,
    title: str = "Feature Correlations",
    figsize: tuple = (10, 8)
):
    """
    Plot Spearman correlation heatmap (Figures 1, 2).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    feature_names : list of str
        Feature names for axes
    threshold : float
        Correlation threshold (from estimate_correlation_threshold)
        Correlations below threshold are considered spurious
    save_path : str, optional
        Path to save figure (if None, display only)
    title : str, default="Feature Correlations"
        Figure title
    figsize : tuple, default=(10, 8)
        Figure size in inches

    Notes
    -----
    Paper format (Figures 1, 2):
    - Spearman correlation matrix
    - Heatmap with viridis colormap
    - Annotated cells
    - Tolerance quantile α=0.01 (threshold at 0.99 quantile)
    """
    # Compute Spearman correlation matrix
    n_features = X.shape[1]
    corr_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr, _ = spearmanr(X[:, i], X[:, j])
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        np.abs(corr_matrix),  # Use absolute values for visualization
        annot=False,  # Don't annotate all cells (too crowded)
        cmap='viridis',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Absolute Spearman Correlation'},
        ax=ax
    )

    # Set labels
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names, rotation=0)
    ax.set_title(title, fontsize=14, pad=20)

    # Add threshold information
    fig.text(
        0.5, 0.02,
        f'Tolerance quantile: {1-threshold:.2f} (significance threshold: {threshold:.2f})',
        ha='center',
        fontsize=10,
        style='italic'
    )

    plt.tight_layout()

    # Save or show
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation heatmap to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_importance_comparison_bars(
    methods_dict: Dict[str, np.ndarray],
    feature_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Variable Importance Comparison",
    figsize: tuple = (12, 6)
):
    """
    Plot grouped bar chart comparing importance across methods.

    Parameters
    ----------
    methods_dict : dict
        Dictionary: {method_name: importance_scores}
        Example: {'DVI-Opt': scores1, 'Breiman (B=10)': scores2}
    feature_names : list of str
        Feature names for x-axis
    save_path : str, optional
        Path to save figure
    title : str, default="Variable Importance Comparison"
        Figure title
    figsize : tuple, default=(12, 6)
        Figure size
    """
    n_features = len(feature_names)
    n_methods = len(methods_dict)

    # Set up bar positions
    x = np.arange(n_features)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars for each method
    for i, (method_name, scores) in enumerate(methods_dict.items()):
        offset = (i - n_methods / 2) * width + width / 2
        ax.bar(x + offset, scores, width, label=method_name, alpha=0.8)

    # Formatting
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved importance comparison to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_svi_decomposition_bars(
    systemic: np.ndarray,
    direct: np.ndarray,
    indirect: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Systemic Variable Importance Decomposition",
    figsize: tuple = (12, 6),
    highlight_features: Optional[List[str]] = None
):
    """
    Plot stacked bar chart showing SVI decomposition.

    Parameters
    ----------
    systemic : np.ndarray
        Systemic importance scores
    direct : np.ndarray
        Direct importance scores
    indirect : np.ndarray
        Indirect importance scores
    feature_names : list of str
        Feature names
    save_path : str, optional
        Path to save figure
    title : str, default="Systemic Variable Importance Decomposition"
        Figure title
    figsize : tuple, default=(12, 6)
        Figure size
    highlight_features : list of str, optional
        Features to highlight (e.g., protected attributes)
    """
    n_features = len(feature_names)
    x = np.arange(n_features)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot stacked bars
    ax.bar(x, direct, label='Direct', color='skyblue', alpha=0.8)
    ax.bar(x, indirect, bottom=direct, label='Indirect', color='coral', alpha=0.8)

    # Highlight protected features if specified
    if highlight_features:
        for i, fname in enumerate(feature_names):
            if fname in highlight_features:
                ax.axvline(x=i, color='red', linestyle='--', alpha=0.5, linewidth=2)

    # Formatting
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add text annotation for highlighted features
    if highlight_features:
        for i, fname in enumerate(feature_names):
            if fname in highlight_features:
                ax.text(
                    i, systemic[i],
                    f'{systemic[i]:.3f}',
                    ha='center', va='bottom',
                    fontweight='bold',
                    fontsize=9
                )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved SVI decomposition to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_speedup_comparison(
    methods: List[str],
    times: List[float],
    baseline_idx: int = -1,
    save_path: Optional[str] = None,
    title: str = "Computation Time Comparison",
    figsize: tuple = (10, 6)
):
    """
    Plot speedup comparison bar chart.

    Parameters
    ----------
    methods : list of str
        Method names
    times : list of float
        Computation times in seconds
    baseline_idx : int, default=-1
        Index of baseline method (usually last, e.g., Breiman B=10)
    save_path : str, optional
        Path to save figure
    title : str, default="Computation Time Comparison"
        Figure title
    figsize : tuple, default=(10, 6)
        Figure size
    """
    baseline_time = times[baseline_idx]
    speedups = [baseline_time / t if t > 0 else 0 for t in times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Absolute times
    colors = ['green' if s > 1 else 'red' for s in speedups]
    ax1.barh(methods, times, color=colors, alpha=0.7)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_title('Computation Time', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)

    # Add time labels
    for i, (method, time) in enumerate(zip(methods, times)):
        ax1.text(time, i, f' {time:.2f}s', va='center', fontsize=9)

    # Plot 2: Speedups relative to baseline
    colors_speedup = ['green' if s > 1 else 'red' for s in speedups]
    ax2.barh(methods, speedups, color=colors_speedup, alpha=0.7)
    ax2.set_xlabel(f'Speedup vs {methods[baseline_idx]}', fontsize=12)
    ax2.set_title('Relative Speedup', fontsize=12)
    ax2.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    ax2.grid(axis='x', alpha=0.3)

    # Add speedup labels
    for i, (method, speedup) in enumerate(zip(methods, speedups)):
        if speedup >= 1:
            ax2.text(speedup, i, f' {speedup:.1f}×', va='center', fontsize=9)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved speedup comparison to: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Test visualizations with synthetic data
    print("Testing visualizations...")
    print("=" * 60)

    # Test correlation heatmap
    print("\nTest 1: Correlation Heatmap")
    rng = np.random.RandomState(42)
    X_test = rng.randn(100, 5)
    # Add some correlations
    X_test[:, 1] = 0.7 * X_test[:, 0] + 0.3 * rng.randn(100)
    X_test[:, 2] = 0.5 * X_test[:, 0] + 0.5 * rng.randn(100)

    feature_names = ['F1', 'F2 (corr F1)', 'F3 (corr F1)', 'F4', 'F5']

    plot_correlation_heatmap(
        X_test,
        feature_names,
        threshold=0.3,
        title="Test Correlation Heatmap",
        save_path=None  # Set to path to save
    )
    print("✓ Correlation heatmap plotted")

    # Test importance comparison
    print("\nTest 2: Importance Comparison")
    methods_dict = {
        'DVI-Optimal': np.array([0.4, 0.3, 0.2, 0.08, 0.02]),
        'DVI-Approx': np.array([0.39, 0.31, 0.19, 0.09, 0.02]),
        'Breiman (B=10)': np.array([0.38, 0.32, 0.21, 0.07, 0.02])
    }

    plot_importance_comparison_bars(
        methods_dict,
        feature_names,
        title="Test Importance Comparison",
        save_path=None
    )
    print("✓ Importance comparison plotted")

    # Test SVI decomposition
    print("\nTest 3: SVI Decomposition")
    systemic = np.array([0.4, 0.3, 0.2, 0.08, 0.02])
    direct = np.array([0.35, 0.28, 0.18, 0.07, 0.02])
    indirect = systemic - direct

    plot_svi_decomposition_bars(
        systemic,
        direct,
        indirect,
        feature_names,
        highlight_features=['F2 (corr F1)'],
        title="Test SVI Decomposition",
        save_path=None
    )
    print("✓ SVI decomposition plotted")

    # Test speedup comparison
    print("\nTest 4: Speedup Comparison")
    methods = ['Direct-Opt', 'Direct-Approx', 'Breiman (B=1)', 'Breiman (B=10)']
    times = [1.63, 1.45, 5.06, 33.84]

    plot_speedup_comparison(
        methods,
        times,
        baseline_idx=-1,
        title="Test Speedup Comparison",
        save_path=None
    )
    print("✓ Speedup comparison plotted")

    print("\n" + "=" * 60)
    print("All visualization tests passed!")
    print("Note: Close plot windows to continue if displayed")
