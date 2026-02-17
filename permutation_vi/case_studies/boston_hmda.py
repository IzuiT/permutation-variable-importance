"""
Boston HMDA Case Study - Section 5.2

This module implements all experiments from the Boston HMDA case study:
- Master model performance (Table 5)
- Direct VI comparison (Tables 6-7, 15-16)
- Stability testing (Table 19)
- Systemic VI fairness testing
- Correlation heatmap (Figure 1)
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings

# Local imports
try:
    from .data_loaders import load_boston_hmda
    from .models import get_boston_models, cross_validate_regression
    from .cv_utils import train_test_holdout, get_cv_fold_predictions
    from .metrics import compute_vi_comparison_metrics
    from .table_generators import (
        format_master_performance_table,
        format_vi_comparison_table,
        format_stability_table,
        format_svi_decomposition,
        save_table_csv,
        print_table
    )
    from .visualizations import plot_correlation_heatmap
except ImportError:
    from data_loaders import load_boston_hmda
    from models import get_boston_models, cross_validate_regression
    from cv_utils import train_test_holdout, get_cv_fold_predictions
    from metrics import compute_vi_comparison_metrics
    from table_generators import (
        format_master_performance_table,
        format_vi_comparison_table,
        format_stability_table,
        format_svi_decomposition,
        save_table_csv,
        print_table
    )
    from visualizations import plot_correlation_heatmap

# Parent package imports (use package path so relative imports inside permutation_vi work)
try:
    from permutation_vi.direct_importance import DirectVariableImportance
    from permutation_vi.systemic_importance import SystemicVariableImportance
    from permutation_vi.utils import estimate_correlation_threshold
except ImportError:
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parent.parent))
    from direct_importance import DirectVariableImportance
    from systemic_importance import SystemicVariableImportance
    from utils import estimate_correlation_threshold


class BostonHMDAExperiment:
    """
    Orchestrator for Boston HMDA experiments (Section 5.2).

    This class manages all experiments on the Boston HMDA dataset including:
    - Master model evaluation
    - Direct variable importance comparisons
    - Systemic variable importance (fairness testing)
    - Correlation analysis
    """

    def __init__(self, data_path: Optional[str] = None, random_state: int = 42):
        """
        Initialize experiment.

        Parameters
        ----------
        data_path : str, optional
            Path to Boston HMDA dataset CSV
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.data_path = data_path

        # Load data
        print("Loading Boston HMDA dataset...")
        self.X, self.y, self.feature_names = load_boston_hmda(data_path)
        print(f"  Loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"  Features: {', '.join(self.feature_names)}")
        print(f"  Target range: [{self.y.min():.2f}, {self.y.max():.2f}]")

        # Storage for results
        self.results = {}

    def run_master_performance(self) -> Dict:
        """
        Reproduce Table 5: Master model performance.

        Returns
        -------
        results : dict
            Performance metrics for TRUST, OLS, RF
        """
        print("\n" + "="*60)
        print("Running Master Model Performance Evaluation (Table 5)")
        print("="*60)

        models = get_boston_models()
        results = {}

        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            perf = cross_validate_regression(
                model, self.X, self.y,
                n_folds=10,
                random_state=self.random_state
            )
            results[model_name] = perf

            # Print results
            print(f"  R²: {perf['r2'][0]:.3f} ± {perf['r2'][1]:.3f}")
            print(f"  MAE: {perf['mae'][0]:.4f} ± {perf['mae'][1]:.4f}")
            print(f"  Nonzero coef: {perf['nonzero'][0]:.1f} ± {perf['nonzero'][1]:.1f}")

        self.results['master_performance'] = results
        return results

    def run_direct_vi_comparison(
        self,
        master_model: str = 'TRUST',
        scoring: str = 'mse'
    ) -> Dict:
        """
        Reproduce Tables 6-7: Direct VI method comparison.

        Parameters
        ----------
        master_model : str, default='TRUST'
            Which master model to use ('TRUST', 'OLS', or 'RF')
        scoring : str, default='mse'
            Scoring metric ('mse' for main tables, 'mae' for appendix)

        Returns
        -------
        results : dict
            VI comparison metrics for all methods
        """
        print(f"\n" + "="*60)
        print(f"Running Direct VI Comparison ({master_model} master, {scoring} scoring)")
        print("="*60)

        # Train master model on full dataset
        models = get_boston_models()
        model = models[master_model]
        model.fit(self.X, self.y)

        # Get ground truth from RF (averaged over folds for stability)
        print("\nComputing ground truth importance (RF, out-of-fold)...")
        rf_model = models['RF']
        y_pred_rf = get_cv_fold_predictions(
            rf_model, self.X, self.y,
            n_folds=10,
            random_state=self.random_state
        )

        # Compute RF importance using DVI
        dvi_gt = DirectVariableImportance(
            permutation_type='optimal',
            scoring_metric=scoring
        )
        gt_scores = dvi_gt.fit(
            lambda X: get_cv_fold_predictions(rf_model, X, self.y, 10, self.random_state),
            self.X,
            self.y
        )

        # Method 1: Direct-Optimal
        print("\n[1/4] Running Direct-Opt...")
        dvi_opt = DirectVariableImportance(
            permutation_type='optimal',
            scoring_metric=scoring
        )
        start = time.time()
        scores_opt = dvi_opt.fit(model, self.X, self.y)
        time_opt = time.time() - start

        metrics_opt = compute_vi_comparison_metrics(gt_scores, scores_opt, time_opt)
        print(f"  Correlation: {metrics_opt['ground_truth_cor']:.3f}")
        print(f"  Time: {metrics_opt['time_ms']:.2f} ms")

        # Method 2: Direct-Approximate
        print("\n[2/4] Running Direct-Approx...")
        dvi_approx = DirectVariableImportance(
            permutation_type='approximate',
            scoring_metric=scoring
        )
        start = time.time()
        scores_approx = dvi_approx.fit(model, self.X, self.y)
        time_approx = time.time() - start

        metrics_approx = compute_vi_comparison_metrics(gt_scores, scores_approx, time_approx)
        print(f"  Correlation: {metrics_approx['ground_truth_cor']:.3f}")
        print(f"  Time: {metrics_approx['time_ms']:.2f} ms")

        # Method 3: Breiman (B=1)
        print("\n[3/4] Running Breiman (B=1)...")
        from sklearn.inspection import permutation_importance
        start = time.time()
        perm_imp_1 = permutation_importance(
            model, self.X, self.y,
            n_repeats=1,
            random_state=self.random_state,
            scoring='neg_mean_squared_error' if scoring == 'mse' else 'neg_mean_absolute_error'
        )
        time_b1 = time.time() - start

        scores_b1 = perm_imp_1.importances_mean
        scores_b1 = np.maximum(scores_b1, 0)
        scores_b1 = scores_b1 / scores_b1.sum() if scores_b1.sum() > 0 else scores_b1

        metrics_b1 = compute_vi_comparison_metrics(gt_scores, scores_b1, time_b1)
        print(f"  Correlation: {metrics_b1['ground_truth_cor']:.3f}")
        print(f"  Time: {metrics_b1['time_ms']:.2f} ms")

        # Method 4: Breiman (B=10)
        print("\n[4/4] Running Breiman (B=10)...")
        start = time.time()
        perm_imp_10 = permutation_importance(
            model, self.X, self.y,
            n_repeats=10,
            random_state=self.random_state,
            scoring='neg_mean_squared_error' if scoring == 'mse' else 'neg_mean_absolute_error'
        )
        time_b10 = time.time() - start

        scores_b10 = perm_imp_10.importances_mean
        scores_b10 = np.maximum(scores_b10, 0)
        scores_b10 = scores_b10 / scores_b10.sum() if scores_b10.sum() > 0 else scores_b10

        metrics_b10 = compute_vi_comparison_metrics(gt_scores, scores_b10, time_b10)
        print(f"  Correlation: {metrics_b10['ground_truth_cor']:.3f}")
        print(f"  Time: {metrics_b10['time_ms']:.2f} ms")

        # Compile results
        results = {
            'Direct-Opt': metrics_opt,
            'Direct-Approx': metrics_approx,
            'Breiman (B=1)': metrics_b1,
            'Breiman (B=10)': metrics_b10
        }

        # Compute speedups
        print("\n" + "-"*60)
        print("Speedup vs Breiman (B=10):")
        print(f"  Direct-Opt: {time_b10/time_opt:.1f}×")
        print(f"  Direct-Approx: {time_b10/time_approx:.1f}×")
        print(f"  Breiman (B=1): {time_b10/time_b1:.1f}×")

        key = f'dvi_comparison_{master_model.lower()}_{scoring}'
        self.results[key] = results
        return results

    def run_stability_test(self, n_runs: int = 10) -> Dict:
        """
        Reproduce Table 19: Stability across repeated runs.

        Parameters
        ----------
        n_runs : int, default=10
            Number of independent runs

        Returns
        -------
        results : dict
            Top-5 feature sets for each method across runs
        """
        print("\n" + "="*60)
        print(f"Running Stability Test ({n_runs} runs)")
        print("="*60)

        # Train RF model
        models = get_boston_models()
        rf_model = models['RF']
        rf_model.fit(self.X, self.y)

        # Storage for top-5 sets
        top5_sets = {
            'Direct-Opt': [],
            'Direct-Approx': [],
            'Breiman (B=1)': [],
            'Breiman (B=10)': []
        }

        for run in range(n_runs):
            print(f"\nRun {run+1}/{n_runs}...")

            # Direct-Opt (deterministic, should be identical)
            dvi_opt = DirectVariableImportance(permutation_type='optimal')
            scores_opt = dvi_opt.fit(rf_model, self.X)
            top5_opt = tuple(np.argsort(scores_opt)[::-1][:5])
            top5_sets['Direct-Opt'].append(top5_opt)

            # Direct-Approx (also deterministic)
            dvi_approx = DirectVariableImportance(permutation_type='approximate')
            scores_approx = dvi_approx.fit(rf_model, self.X)
            top5_approx = tuple(np.argsort(scores_approx)[::-1][:5])
            top5_sets['Direct-Approx'].append(top5_approx)

            # Breiman (B=1) with different seed
            from sklearn.inspection import permutation_importance
            perm_imp_1 = permutation_importance(
                rf_model, self.X, self.y,
                n_repeats=1,
                random_state=self.random_state + run
            )
            scores_b1 = perm_imp_1.importances_mean
            top5_b1 = tuple(np.argsort(scores_b1)[::-1][:5])
            top5_sets['Breiman (B=1)'].append(top5_b1)

            # Breiman (B=10) with different seed
            perm_imp_10 = permutation_importance(
                rf_model, self.X, self.y,
                n_repeats=10,
                random_state=self.random_state + run
            )
            scores_b10 = perm_imp_10.importances_mean
            top5_b10 = tuple(np.argsort(scores_b10)[::-1][:5])
            top5_sets['Breiman (B=10)'].append(top5_b10)

        # Count unique top-5 sets
        print("\n" + "-"*60)
        print("Unique top-5 sets across runs:")
        for method, sets in top5_sets.items():
            unique_sets = len(set(sets))
            print(f"  {method}: {unique_sets} unique set(s)")

        self.results['stability'] = top5_sets
        return top5_sets

    def run_systemic_vi_ols(self) -> Dict:
        """
        Reproduce Section 5.2.2 Scenario 1: OLS with all features.

        Tests systemic importance of 'black' attribute when using all features.

        Returns
        -------
        results : dict
            Systemic, direct, indirect importance for 'black'
        """
        print("\n" + "="*60)
        print("Running Systemic VI: OLS with all features")
        print("="*60)

        # Train OLS on all features
        models = get_boston_models()
        ols_model = models['OLS']
        ols_model.fit(self.X, self.y)

        # Compute SVI
        print("\nComputing systemic importance...")
        svi = SystemicVariableImportance(
            permutation_type='optimal',
            scoring_metric='mae',
            correlation_method='spearman',
            alpha=0.01
        )

        systemic, direct, indirect = svi.fit(
            ols_model, self.X,
            feature_names=self.feature_names,
            return_decomposition=True
        )

        # Extract 'black' importance
        black_idx = self.feature_names.index('black')

        results = {
            'systemic': systemic[black_idx],
            'direct': direct[black_idx],
            'indirect': indirect[black_idx]
        }

        print(f"\n{format_svi_decomposition('black', **results)}")

        self.results['svi_ols'] = results
        return results

    def run_systemic_vi_trust_no_black(self) -> Dict:
        """
        Reproduce Section 5.2.2 Scenario 2: TRUST without 'black'.

        Tests indirect reliance on 'black' when it's excluded from training.

        Returns
        -------
        results : dict
            Systemic, direct, indirect importance for 'black'
        """
        print("\n" + "="*60)
        print("Running Systemic VI: TRUST without 'black'")
        print("="*60)

        # Remove 'black' from features
        black_idx = self.feature_names.index('black')
        X_no_black = np.delete(self.X, black_idx, axis=1)
        features_no_black = [f for i, f in enumerate(self.feature_names) if i != black_idx]

        # Train TRUST without 'black'
        models = get_boston_models()
        trust_model = models['TRUST']
        trust_model.fit(X_no_black, self.y)

        print(f"Model trained on {len(features_no_black)} features (excluded: 'black')")

        # Wrap model to handle missing feature
        class ModelWrapper:
            def __init__(self, model, exclude_idx):
                self.model = model
                self.exclude_idx = exclude_idx

            def predict(self, X):
                X_reduced = np.delete(X, self.exclude_idx, axis=1)
                return self.model.predict(X_reduced)

        wrapped_model = ModelWrapper(trust_model, black_idx)

        # Compute SVI on full feature set (including 'black' for correlation analysis)
        print("\nComputing systemic importance on full feature set...")
        svi = SystemicVariableImportance(
            permutation_type='optimal',
            scoring_metric='mae',
            correlation_method='spearman',
            alpha=0.01
        )

        systemic, direct, indirect = svi.fit(
            wrapped_model, self.X,
            feature_names=self.feature_names,
            return_decomposition=True
        )

        results = {
            'systemic': systemic[black_idx],
            'direct': direct[black_idx],
            'indirect': indirect[black_idx]
        }

        print(f"\n{format_svi_decomposition('black', **results)}")
        print("\nInterpretation: Model relies on 'black' indirectly through proxies!")

        # Show proxy features
        print("\nProxy features for 'black' (|correlation| > threshold):")
        proxy_indices, corr_vals = svi.get_feature_proxies(black_idx, return_correlations=True)
        for idx, corr_val in zip(proxy_indices, corr_vals):
            print(f"  {self.feature_names[idx]}: ρ = {corr_val:.3f}")

        self.results['svi_trust_no_black'] = results
        return results

    def generate_correlation_heatmap(self, save_path: Optional[str] = None):
        """
        Reproduce Figure 1: Correlation heatmap.

        Parameters
        ----------
        save_path : str, optional
            Path to save figure
        """
        print("\n" + "="*60)
        print("Generating Correlation Heatmap (Figure 1)")
        print("="*60)

        # Compute correlation threshold
        threshold = estimate_correlation_threshold(
            self.X,
            alpha=0.01,
            method='spearman',
            random_state=self.random_state
        )
        print(f"Correlation threshold (α=0.01): {threshold:.4f}")

        # Plot heatmap
        plot_correlation_heatmap(
            self.X,
            self.feature_names,
            threshold=threshold,
            title="Feature Correlations - Boston HMDA Dataset",
            save_path=save_path
        )

        if save_path:
            print(f"Saved to: {save_path}")

    def run_all_experiments(self, output_dir: str = 'results/boston_hmda'):
        """
        Run complete experimental pipeline.

        Parameters
        ----------
        output_dir : str
            Directory to save results
        """
        output_path = Path(output_dir)
        tables_path = output_path / 'tables'
        figures_path = output_path / 'figures'

        tables_path.mkdir(parents=True, exist_ok=True)
        figures_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("BOSTON HMDA CASE STUDY - COMPLETE PIPELINE")
        print("="*60)

        # 1. Master performance
        print("\n[1/7] Master Model Performance...")
        master_perf = self.run_master_performance()
        table5 = format_master_performance_table(master_perf, 'regression')
        save_table_csv(table5, tables_path / 'table5_master_performance.csv')
        print_table(table5, "Table 5: Master Model Performance")

        # 2. Direct VI - TRUST (MSE scoring)
        print("\n[2/7] Direct VI - TRUST Master (MSE)...")
        dvi_trust = self.run_direct_vi_comparison('TRUST', 'mse')
        table6 = format_vi_comparison_table(dvi_trust, 'TRUST')
        save_table_csv(table6, tables_path / 'table6_dvi_trust.csv')
        print_table(table6, "Table 6: Direct VI (TRUST, MSE)")

        # 3. Direct VI - OLS (MSE scoring)
        print("\n[3/7] Direct VI - OLS Master (MSE)...")
        dvi_ols = self.run_direct_vi_comparison('OLS', 'mse')
        table7 = format_vi_comparison_table(dvi_ols, 'OLS')
        save_table_csv(table7, tables_path / 'table7_dvi_ols.csv')
        print_table(table7, "Table 7: Direct VI (OLS, MSE)")

        # 4. Direct VI - TRUST (MAE scoring, Appendix)
        print("\n[4/7] Direct VI - TRUST Master (MAE)...")
        dvi_trust_mae = self.run_direct_vi_comparison('TRUST', 'mae')
        table15 = format_vi_comparison_table(dvi_trust_mae, 'TRUST')
        save_table_csv(table15, tables_path / 'table15_dvi_trust_mae.csv')
        print_table(table15, "Table 15: Direct VI (TRUST, MAE)")

        # 5. Stability test
        print("\n[5/7] Stability Test...")
        stability = self.run_stability_test(n_runs=10)
        table19 = format_stability_table(stability, self.feature_names)
        save_table_csv(table19, tables_path / 'table19_stability.csv')
        print_table(table19, "Table 19: Stability Test")

        # 6. Systemic VI scenarios
        print("\n[6/7] Systemic VI Scenarios...")
        svi_ols = self.run_systemic_vi_ols()
        svi_trust = self.run_systemic_vi_trust_no_black()

        # 7. Correlation heatmap
        print("\n[7/7] Correlation Heatmap...")
        self.generate_correlation_heatmap(
            save_path=str(figures_path / 'figure1_correlation_heatmap.png')
        )

        # Summary
        print("\n" + "="*60)
        print("BOSTON HMDA EXPERIMENTS COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {output_path}")
        print(f"  Tables: {tables_path}")
        print(f"  Figures: {figures_path}")


if __name__ == "__main__":
    # Run Boston HMDA experiments
    exp = BostonHMDAExperiment()
    exp.run_all_experiments()
