#!/usr/bin/env python3
"""
Main script to reproduce all Boston HMDA results (Section 5.2).

Usage:
    uv run python permutation_vi/case_studies/run_boston_hmda.py
    uv run python permutation_vi/case_studies/run_boston_hmda.py --data-path data/boston_hmda/hmda.csv
    uv run python permutation_vi/case_studies/run_boston_hmda.py --output-dir custom_results

Outputs:
    - results/boston_hmda/tables/*.csv - All result tables
    - results/boston_hmda/figures/*.png - Correlation heatmap
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from permutation_vi.case_studies.boston_hmda import BostonHMDAExperiment


def main():
    """Run Boston HMDA case study."""
    parser = argparse.ArgumentParser(
        description="Reproduce Boston HMDA case study results (Paper Section 5.2)"
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to Boston HMDA dataset CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/boston_hmda',
        help='Directory to save results (default: results/boston_hmda)'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['master', 'dvi', 'stability', 'svi', 'heatmap', 'all'],
        default=['all'],
        help='Which experiments to run (default: all)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Initialize experiment
    try:
        exp = BostonHMDAExperiment(
            data_path=args.data_path,
            random_state=args.random_state
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease download the dataset from:")
        print("  https://github.com/adc-trust-ai/trust-free")
        print("\nOr specify the path with --data-path")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        sys.exit(1)

    # Run experiments
    print("\n" + "="*70)
    print("BOSTON HMDA CASE STUDY - Paper Section 5.2")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"Experiments: {', '.join(args.experiments)}")
    print()

    try:
        if 'all' in args.experiments:
            # Run complete pipeline
            exp.run_all_experiments(output_dir=args.output_dir)
        else:
            # Run selected experiments
            from pathlib import Path as P
            output_path = P(args.output_dir)
            tables_path = output_path / 'tables'
            figures_path = output_path / 'figures'
            tables_path.mkdir(parents=True, exist_ok=True)
            figures_path.mkdir(parents=True, exist_ok=True)

            if 'master' in args.experiments:
                print("\nRunning Master Model Performance...")
                results = exp.run_master_performance()

            if 'dvi' in args.experiments:
                print("\nRunning Direct VI Comparisons...")
                exp.run_direct_vi_comparison('TRUST', 'mse')
                exp.run_direct_vi_comparison('OLS', 'mse')

            if 'stability' in args.experiments:
                print("\nRunning Stability Test...")
                exp.run_stability_test(n_runs=10)

            if 'svi' in args.experiments:
                print("\nRunning Systemic VI...")
                exp.run_systemic_vi_ols()
                exp.run_systemic_vi_trust_no_black()

            if 'heatmap' in args.experiments:
                print("\nGenerating Correlation Heatmap...")
                exp.generate_correlation_heatmap(
                    save_path=str(figures_path / 'figure1_correlation_heatmap.png')
                )

        print("\n" + "="*70)
        print("✅ BOSTON HMDA EXPERIMENTS COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {args.output_dir}/")
        print("  📊 Tables: tables/")
        print("  📈 Figures: figures/")
        print()

    except Exception as e:
        print(f"\n❌ Error during experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
