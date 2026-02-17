"""
German Credit Case Study - Section 5.3

This module implements all experiments from the German Credit case study:
- Master model performance (Table 8)
- Direct VI comparison (Tables 9-10, 17-18)
- Stability testing (Table 20)
- Systemic VI fairness testing
- Correlation heatmap (Figure 2)
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Local imports
try:
    from .data_loaders import load_german_credit
    from .models import get_german_models, cross_validate_classification
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
    from data_loaders import load_german_credit
    from models import get_german_models, cross_validate_classification
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


class GermanCreditExperiment:
    """
    Orchestrator for German Credit experiments (Section 5.3).

    This class manages all experiments on the German Credit dataset including:
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
            Path to German Credit dataset CSV
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.data_path = data_path

        # Load data
        print("Loading German Credit dataset...")
        self.X, self.y, self.feature_names = load_german_credit(data_path)
        print(f"  Loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"  Features: {', '.join(self.feature_names[:5])} ...")
        print(f"  Class distribution: {np.bincount(self.y.astype(int))}")

        # Storage for results
        self.results = {}

    def run_master_performance(self) -> Dict:
        """
        Reproduce Table 8: Master model performance.

        Returns
        -------
        results : dict
            Performance metrics for l1_logistic, logistic, RF
        """
        print("\n" + "="*60)
        print("Running Master Model Performance Evaluation (Table 8)")
        print("="*60)

        models = get_german_models()
        results = {}

        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            perf = cross_validate_classification(
                model, self.X, self.y,
                n_folds=10,
                random_state=self.random_state
            )
            results[model_name] = perf

            # Print results
            print(f"  PR-AUC: {perf['pr_auc'][0]:.3f} ± {perf['pr_auc'][1]:.3f}")
            print(f"  ROC-AUC: {perf['roc_auc'][0]:.3f} ± {perf['roc_auc'][1]:.3f}")
            print(f"  Nonzero coef: {perf['nonzero'][0]:.1f} ± {perf['nonzero'][1]:.1f}")

        self.results['master_performance'] = results
        return results

    def run_direct_vi_comparison(
        self,
        master_model: str = 'l1_logistic',
        scoring: str = 'mse'
    ) -> Dict:
        """
        Reproduce Tables 9-10: Direct VI method comparison.

        Parameters
        ----------
        master_model : str, default='l1_logistic'
            Which master model to use ('l1_logistic', 'logistic', or 'RF')
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
        models = get_german_models()
        model = models[master_model]
        model.fit(self.X, self.y)

        # Get ground truth from RF (averaged over folds)
        print("\nComputing ground truth importance (RF, out-of-fold)...")
        rf_model = models['RF']

        # For classification, we need to use probabilities
        # Create a wrapper that returns class 1 probabilities
        class ProbWrapper:
            def __init__(self, clf_model):
                self.clf = clf_model

            def predict(self, X):
                return self.clf.predict_proba(X)[:, 1]

        # Compute ground truth using DVI on RF probabilities
        dvi_gt = DirectVariableImportance(
            permutation_type='optimal',
            scoring_metric=scoring
        )

        # Fit DVI using out-of-fold RF predictions
        y_pred_rf = get_cv_fold_predictions(
            rf_model, self.X, self.y,
            n_folds=10,
            random_state=self.random_state,
            return_proba=True
        )[:, 1]  # Get class 1 probabilities

        # Use a simple lambda that returns the cached predictions
        gt_scores = dvi_gt.fit(lambda X: y_pred_rf, self.X, self.y)

        # For actual model testing, wrap to return probabilities
        model_wrapper = ProbWrapper(model)

        # Method 1: Direct-Optimal
        print("\n[1/4] Running Direct-Opt...")
        dvi_opt = DirectVariableImportance(
            permutation_type='optimal',
            scoring_metric=scoring
        )
        start = time.time()
        scores_opt = dvi_opt.fit(model_wrapper, self.X, self.y)
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
        scores_approx = dvi_approx.fit(model_wrapper, self.X, self.y)
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
            scoring='neg_brier_score'  # For classification
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
            scoring='neg_brier_score'
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

        key = f'dvi_comparison_{master_model}_{scoring}'
        self.results[key] = results
        return results

    def run_stability_test(self, n_runs: int = 10) -> Dict:
        """
        Reproduce Table 20: Stability across repeated runs.

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
        models = get_german_models()
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

            # Wrapper for probabilities
            class ProbWrapper:
                def __init__(self, clf):
                    self.clf = clf
                def predict(self, X):
                    return self.clf.predict_proba(X)[:, 1]

            model_wrapper = ProbWrapper(rf_model)

            # Direct-Opt (deterministic)
            dvi_opt = DirectVariableImportance(permutation_type='optimal')
            scores_opt = dvi_opt.fit(model_wrapper, self.X, self.y)
            top5_opt = tuple(np.argsort(scores_opt)[::-1][:5])
            top5_sets['Direct-Opt'].append(top5_opt)

            # Direct-Approx (also deterministic)
            dvi_approx = DirectVariableImportance(permutation_type='approximate')
            scores_approx = dvi_approx.fit(model_wrapper, self.X, self.y)
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

    def run_systemic_vi_unregularized(self) -> Dict:
        """
        Reproduce Section 5.3.2 Scenario 1: Unregularized logistic with all features.

        Tests systemic importance of 'Sex-Marital_status' attribute.

        Returns
        -------
        results : dict
            Systemic, direct, indirect importance for protected attribute
        """
        print("\n" + "="*60)
        print("Running Systemic VI: Unregularized Logistic with all features")
        print("="*60)

        # Train unregularized logistic on all features
        models = get_german_models()
        logistic_model = models['logistic']
        logistic_model.fit(self.X, self.y)

        # Wrapper for probabilities
        class ProbWrapper:
            def __init__(self, clf):
                self.clf = clf
            def predict(self, X):
                return self.clf.predict_proba(X)[:, 1]

        model_wrapper = ProbWrapper(logistic_model)

        # Compute SVI
        print("\nComputing systemic importance...")
        svi = SystemicVariableImportance(
            permutation_type='optimal',
            scoring_metric='mae',
            correlation_method='spearman',
            alpha=0.01
        )

        systemic, direct, indirect = svi.fit(
            model_wrapper, self.X,
            feature_names=self.feature_names,
            return_decomposition=True
        )

        # Extract protected attribute importance
        protected_idx = self.feature_names.index('Sex-Marital_status')

        results = {
            'systemic': systemic[protected_idx],
            'direct': direct[protected_idx],
            'indirect': indirect[protected_idx]
        }

        print(f"\n{format_svi_decomposition('Sex-Marital_status', **results)}")

        self.results['svi_unregularized'] = results
        return results

    def run_systemic_vi_sparse_no_protected(self) -> Dict:
        """
        Reproduce Section 5.3.2 Scenario 2: Sparse logistic without protected attribute.

        Tests safeguarding by exclusion.

        Returns
        -------
        results : dict
            Systemic, direct, indirect importance for protected attribute
        """
        print("\n" + "="*60)
        print("Running Systemic VI: Sparse Logistic without 'Sex-Marital_status'")
        print("="*60)

        # Remove protected attribute
        protected_idx = self.feature_names.index('Sex-Marital_status')
        X_no_protected = np.delete(self.X, protected_idx, axis=1)
        features_no_protected = [f for i, f in enumerate(self.feature_names) if i != protected_idx]

        # Train sparse logistic without protected attribute
        models = get_german_models()
        sparse_model = models['l1_logistic']
        sparse_model.fit(X_no_protected, self.y)

        print(f"Model trained on {len(features_no_protected)} features (excluded: 'Sex-Marital_status')")

        # Wrap model to handle missing feature
        class ModelWrapper:
            def __init__(self, model, exclude_idx):
                self.model = model
                self.exclude_idx = exclude_idx

            def predict(self, X):
                X_reduced = np.delete(X, self.exclude_idx, axis=1)
                return self.model.predict_proba(X_reduced)[:, 1]

        wrapped_model = ModelWrapper(sparse_model, protected_idx)

        # Compute SVI on full feature set
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
            'systemic': systemic[protected_idx],
            'direct': direct[protected_idx],
            'indirect': indirect[protected_idx]
        }

        print(f"\n{format_svi_decomposition('Sex-Marital_status', **results)}")

        if results['systemic'] < 0.001:
            print("\nInterpretation: Exclusion successfully prevents indirect reliance!")
        else:
            print("\nWarning: Model may still rely on protected attribute via proxies")

        self.results['svi_sparse_no_protected'] = results
        return results

    def generate_correlation_heatmap(self, save_path: Optional[str] = None):
        """
        Reproduce Figure 2: Correlation heatmap.

        Parameters
        ----------
        save_path : str, optional
            Path to save figure
        """
        print("\n" + "="*60)
        print("Generating Correlation Heatmap (Figure 2)")
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
            title="Feature Correlations - German Credit Dataset",
            save_path=save_path
        )

        if save_path:
            print(f"Saved to: {save_path}")

    def run_all_experiments(self, output_dir: str = 'results/german_credit'):
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
        print("GERMAN CREDIT CASE STUDY - COMPLETE PIPELINE")
        print("="*60)

        # 1. Master performance
        print("\n[1/7] Master Model Performance...")
        master_perf = self.run_master_performance()
        table8 = format_master_performance_table(master_perf, 'classification')
        save_table_csv(table8, tables_path / 'table8_master_performance.csv')
        print_table(table8, "Table 8: Master Model Performance")

        # 2. Direct VI - l1_logistic (MSE scoring)
        print("\n[2/7] Direct VI - l1_logistic Master (MSE)...")
        dvi_l1 = self.run_direct_vi_comparison('l1_logistic', 'mse')
        table9 = format_vi_comparison_table(dvi_l1, 'l1_logistic')
        save_table_csv(table9, tables_path / 'table9_dvi_l1logistic.csv')
        print_table(table9, "Table 9: Direct VI (l1_logistic, MSE)")

        # 3. Direct VI - logistic (MSE scoring)
        print("\n[3/7] Direct VI - Logistic Master (MSE)...")
        dvi_log = self.run_direct_vi_comparison('logistic', 'mse')
        table10 = format_vi_comparison_table(dvi_log, 'logistic')
        save_table_csv(table10, tables_path / 'table10_dvi_logistic.csv')
        print_table(table10, "Table 10: Direct VI (Logistic, MSE)")

        # 4. Direct VI - l1_logistic (MAE scoring, Appendix)
        print("\n[4/7] Direct VI - l1_logistic Master (MAE)...")
        dvi_l1_mae = self.run_direct_vi_comparison('l1_logistic', 'mae')
        table17 = format_vi_comparison_table(dvi_l1_mae, 'l1_logistic')
        save_table_csv(table17, tables_path / 'table17_dvi_l1logistic_mae.csv')
        print_table(table17, "Table 17: Direct VI (l1_logistic, MAE)")

        # 5. Stability test
        print("\n[5/7] Stability Test...")
        stability = self.run_stability_test(n_runs=10)
        table20 = format_stability_table(stability, self.feature_names)
        save_table_csv(table20, tables_path / 'table20_stability.csv')
        print_table(table20, "Table 20: Stability Test")

        # 6. Systemic VI scenarios
        print("\n[6/7] Systemic VI Scenarios...")
        svi_unreg = self.run_systemic_vi_unregularized()
        svi_sparse = self.run_systemic_vi_sparse_no_protected()

        # 7. Correlation heatmap
        print("\n[7/7] Correlation Heatmap...")
        self.generate_correlation_heatmap(
            save_path=str(figures_path / 'figure2_correlation_heatmap.png')
        )

        # Summary
        print("\n" + "="*60)
        print("GERMAN CREDIT EXPERIMENTS COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {output_path}")
        print(f"  Tables: {tables_path}")
        print(f"  Figures: {figures_path}")


if __name__ == "__main__":
    # Run German Credit experiments
    exp = GermanCreditExperiment()
    exp.run_all_experiments()
