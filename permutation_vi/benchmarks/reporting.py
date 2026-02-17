"""
Visualization and reporting utilities for benchmarks.

Provides functions to create paper-quality tables, plots, and visualizations
matching the format used in "One Permutation Is All You Need".
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings

# Try to import pandas, but make it optional
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("pandas not available. Some reporting functions will not work.")


def create_performance_table(results: Dict, save_path: Optional[str] = None) -> str:
    """
    Create a formatted performance table like Tables 1-4 in the paper.

    Parameters
    ----------
    results : Dict
        Aggregated results from aggregate_results()
    save_path : str or None, default=None
        If provided, save table to file (txt or latex)

    Returns
    -------
    table_str : str
        Formatted table string

    Examples
    --------
    >>> from permutation_vi.benchmarks import aggregate_results
    >>> summary = aggregate_results(scenario_results)
    >>> table = create_performance_table(summary)
    >>> print(table)
    """
    methods = [
        ('Direct-Opt', 'direct_opt'),
        ('Direct-Approx', 'direct_approx'),
        ('Breiman (B=1)', 'breiman_1'),
        ('Breiman (B=10)', 'breiman_10')
    ]

    # Header
    lines = []
    lines.append("=" * 80)
    lines.append("Performance Comparison")
    lines.append("=" * 80)
    lines.append(f"{'Method':<20} {'Ground-truth cor':<20} {'Max score diff':<20} {'Time (ms)':<20}")
    lines.append("-" * 80)

    # Data rows
    for name, key in methods:
        cor = results[key]['ground_truth_cor']
        max_diff = results[key]['max_score_diff']
        time_ms = results[key]['time_ms']
        lines.append(f"{name:<20} {cor:<20} {max_diff:<20} {time_ms:<20}")

    lines.append("=" * 80)

    table_str = "\n".join(lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_str)

    return table_str


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    feature_names: List[str],
    threshold: float,
    title: str = "Feature Correlations",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Create correlation heatmap like Figures 1-2 in the paper.

    Shows pairwise feature correlations with threshold line for
    systemic importance analysis.

    Parameters
    ----------
    corr_matrix : np.ndarray of shape (p, p)
        Correlation matrix (Spearman or Pearson)
    feature_names : List[str]
        Feature names for axis labels
    threshold : float
        Correlation threshold for systemic importance (typically from
        estimate_correlation_threshold with α=0.01)
    title : str, default="Feature Correlations"
        Plot title
    save_path : str or None, default=None
        If provided, save figure to file
    figsize : tuple, default=(10, 8)
        Figure size in inches

    Examples
    --------
    >>> from scipy.stats import spearmanr
    >>> from permutation_vi.utils import estimate_correlation_threshold
    >>> corr_matrix, _ = spearmanr(X)
    >>> threshold = estimate_correlation_threshold(X, alpha=0.01)
    >>> plot_correlation_heatmap(corr_matrix, feature_names, threshold)
    """
    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cbar_kws={'label': 'Spearman ρ'}
    )

    plt.title(f'{title}\n(tolerance quantile: {threshold:.2f})', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")

    plt.show()


def plot_scenario_comparison(
    scenario_results: List[Dict],
    metric: str = 'cor_mean',
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Create bar plot comparing methods across scenarios.

    Similar to Figure 3 in paper's Appendix C.

    Parameters
    ----------
    scenario_results : List[Dict]
        List of results from run_single_scenario()
    metric : str, default='cor_mean'
        Metric to plot: 'cor_mean', 'max_diff_mean', 'time_mean'
    save_path : str or None, default=None
        If provided, save figure
    figsize : tuple, default=(12, 6)
        Figure size

    Examples
    --------
    >>> from permutation_vi.benchmarks import run_single_scenario
    >>> results = [run_single_scenario(s, n_repetitions=5) for s in scenarios]
    >>> plot_scenario_comparison(results, metric='cor_mean')
    """
    if not HAS_PANDAS:
        warnings.warn("pandas required for plot_scenario_comparison")
        return

    # Extract data
    data = []
    for result in scenario_results:
        scenario_id = result['scenario']['scenario_id']
        for method in ['breiman_1', 'breiman_10', 'direct_opt', 'direct_approx']:
            data.append({
                'Scenario': scenario_id,
                'Method': method,
                'Value': result[method][metric]
            })

    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=figsize)

    methods = ['breiman_1', 'breiman_10', 'direct_opt', 'direct_approx']
    method_labels = ['Breiman (B=1)', 'Breiman (B=10)', 'Direct-Opt', 'Direct-Approx']

    x = np.arange(len(scenario_results))
    width = 0.2

    for i, (method, label) in enumerate(zip(methods, method_labels)):
        values = df[df['Method'] == method]['Value'].values
        plt.bar(x + i * width, values, width, label=label)

    plt.xlabel('Scenario ID')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Method Comparison: {metric.replace("_", " ").title()}')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_stability_analysis(
    top_features_across_runs: List[List[int]],
    n_features: int,
    method_name: str = "Method",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Visualize feature ranking stability across multiple runs.

    Shows how often each feature appears in top-k across independent runs,
    demonstrating determinism advantage of DVI over stochastic Breiman methods.

    Parameters
    ----------
    top_features_across_runs : List[List[int]]
        List of top-k feature lists from multiple runs
        Each sublist contains feature indices
    n_features : int
        Total number of features
    method_name : str, default="Method"
        Name of method for title
    save_path : str or None
        Save path for figure
    figsize : tuple, default=(10, 6)
        Figure size

    Examples
    --------
    >>> # Run method 10 times, collect top-5 features each time
    >>> top_features = []
    >>> for i in range(10):
    >>>     scores = method.fit(model, X, random_state=i)
    >>>     top_5 = np.argsort(scores)[-5:][::-1]
    >>>     top_features.append(top_5)
    >>> plot_stability_analysis(top_features, n_features=X.shape[1], method_name="Breiman")
    """
    # Count frequency of each feature appearing in top-k
    n_runs = len(top_features_across_runs)
    feature_counts = np.zeros(n_features)

    for top_features in top_features_across_runs:
        for feat_idx in top_features:
            feature_counts[feat_idx] += 1

    # Convert to percentages
    feature_percentages = (feature_counts / n_runs) * 100

    # Plot
    plt.figure(figsize=figsize)
    plt.bar(range(n_features), feature_percentages)
    plt.xlabel('Feature Index')
    plt.ylabel('Appearance Frequency (%)')
    plt.title(f'Top-{len(top_features_across_runs[0])} Feature Stability: {method_name}\n'
              f'({n_runs} independent runs)')
    plt.ylim([0, 105])
    plt.axhline(y=100, color='g', linestyle='--', label='100% (deterministic)', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def generate_latex_table(results: Dict, caption: str = "") -> str:
    """
    Generate LaTeX table code for paper-quality output.

    Parameters
    ----------
    results : Dict
        Aggregated results from aggregate_results()
    caption : str, default=""
        Table caption

    Returns
    -------
    latex_code : str
        LaTeX table code

    Examples
    --------
    >>> latex = generate_latex_table(summary, caption="Regression results")
    >>> print(latex)
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Method & Ground-truth cor & Max score diff & Time (ms) \\\\")
    latex.append("\\midrule")

    methods = [
        ('Direct-Opt', 'direct_opt'),
        ('Direct-Approx', 'direct_approx'),
        ('Breiman (B=1)', 'breiman_1'),
        ('Breiman (B=10)', 'breiman_10')
    ]

    for name, key in methods:
        cor = results[key]['ground_truth_cor']
        max_diff = results[key]['max_score_diff']
        time_ms = results[key]['time_ms']
        latex.append(f"{name} & {cor} & {max_diff} & {time_ms} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


if __name__ == "__main__":
    print("Visualization and Reporting Module")
    print("=" * 60)

    # Demo with synthetic data
    print("\nGenerating sample correlation heatmap...")

    np.random.seed(42)
    n_features = 8
    feature_names = [f"F{i}" for i in range(n_features)]

    # Create synthetic correlation matrix
    corr_matrix = np.random.randn(n_features, n_features)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = np.clip(corr_matrix, -1, 1)  # Clip to valid range

    threshold = 0.5

    try:
        plot_correlation_heatmap(
            corr_matrix,
            feature_names,
            threshold,
            title="Example: Feature Correlations"
        )
        print("✓ Heatmap generated successfully")
    except Exception as e:
        print(f"Note: Plotting requires matplotlib/seaborn: {e}")

    print("\n✓ Reporting module loaded successfully")
