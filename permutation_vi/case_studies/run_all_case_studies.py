#!/usr/bin/env python3
"""
Run both case studies sequentially.

Usage:
    uv run python permutation_vi/case_studies/run_all_case_studies.py
    uv run python permutation_vi/case_studies/run_all_case_studies.py --output-dir custom_results

This script runs:
1. Boston HMDA case study (Section 5.2)
2. German Credit case study (Section 5.3)

Outputs:
    - results/boston_hmda/ - Boston HMDA results
    - results/german_credit/ - German Credit results
"""

import argparse
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from permutation_vi.case_studies.boston_hmda import BostonHMDAExperiment
from permutation_vi.case_studies.german_credit import GermanCreditExperiment


def print_banner(text):
    """Print a nice banner."""
    width = 70
    print("\n" + "="*width)
    print(f"  {text}")
    print("="*width + "\n")


def main():
    """Run both case studies."""
    parser = argparse.ArgumentParser(
        description="Reproduce all case study results from Paper Sections 5.2 and 5.3"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Base directory for results (default: results/)'
    )
    parser.add_argument(
        '--boston-data',
        type=str,
        default=None,
        help='Path to Boston HMDA dataset'
    )
    parser.add_argument(
        '--german-data',
        type=str,
        default=None,
        help='Path to German Credit dataset'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--skip-boston',
        action='store_true',
        help='Skip Boston HMDA case study'
    )
    parser.add_argument(
        '--skip-german',
        action='store_true',
        help='Skip German Credit case study'
    )

    args = parser.parse_args()

    print_banner("CASE STUDIES REPRODUCTION - Paper Sections 5.2 & 5.3")
    print(f"Output directory: {args.output_dir}/")
    print(f"Random seed: {args.random_state}")
    print()

    overall_start = time.time()
    results_summary = []

    # =================================================================
    # CASE STUDY A: Boston HMDA
    # =================================================================
    if not args.skip_boston:
        print_banner("CASE STUDY A: Boston HMDA (Section 5.2)")

        try:
            exp_boston = BostonHMDAExperiment(
                data_path=args.boston_data,
                random_state=args.random_state
            )

            boston_start = time.time()
            exp_boston.run_all_experiments(
                output_dir=f"{args.output_dir}/boston_hmda"
            )
            boston_time = time.time() - boston_start

            results_summary.append({
                'study': 'Boston HMDA',
                'status': '✅ Complete',
                'time': boston_time
            })

        except FileNotFoundError as e:
            print(f"\n⚠️  Skipping Boston HMDA: Dataset not found")
            print(f"    Download from: https://github.com/adc-trust-ai/trust-free")
            results_summary.append({
                'study': 'Boston HMDA',
                'status': '⚠️  Skipped (no data)',
                'time': 0
            })
        except Exception as e:
            print(f"\n❌ Error in Boston HMDA: {e}")
            results_summary.append({
                'study': 'Boston HMDA',
                'status': '❌ Failed',
                'time': 0
            })
            import traceback
            traceback.print_exc()
    else:
        print("\n⏭️  Skipping Boston HMDA (--skip-boston)")
        results_summary.append({
            'study': 'Boston HMDA',
            'status': '⏭️  Skipped (user)',
            'time': 0
        })

    # =================================================================
    # CASE STUDY B: German Credit
    # =================================================================
    if not args.skip_german:
        print_banner("CASE STUDY B: German Credit (Section 5.3)")

        try:
            exp_german = GermanCreditExperiment(
                data_path=args.german_data,
                random_state=args.random_state
            )

            german_start = time.time()
            exp_german.run_all_experiments(
                output_dir=f"{args.output_dir}/german_credit"
            )
            german_time = time.time() - german_start

            results_summary.append({
                'study': 'German Credit',
                'status': '✅ Complete',
                'time': german_time
            })

        except FileNotFoundError as e:
            print(f"\n⚠️  Skipping German Credit: Dataset not found")
            print(f"    Download from: https://github.com/adc-trust-ai/trust-free")
            results_summary.append({
                'study': 'German Credit',
                'status': '⚠️  Skipped (no data)',
                'time': 0
            })
        except Exception as e:
            print(f"\n❌ Error in German Credit: {e}")
            results_summary.append({
                'study': 'German Credit',
                'status': '❌ Failed',
                'time': 0
            })
            import traceback
            traceback.print_exc()
    else:
        print("\n⏭️  Skipping German Credit (--skip-german)")
        results_summary.append({
            'study': 'German Credit',
            'status': '⏭️  Skipped (user)',
            'time': 0
        })

    # =================================================================
    # SUMMARY
    # =================================================================
    overall_time = time.time() - overall_start

    print_banner("SUMMARY")

    print("Results by Case Study:")
    print("-" * 70)
    for result in results_summary:
        if result['time'] > 0:
            time_str = f"{result['time']:.1f}s"
        else:
            time_str = "-"
        print(f"  {result['study']:<20} {result['status']:<20} {time_str:>10}")

    print()
    print(f"Total time: {overall_time:.1f}s")
    print()
    print(f"📁 Results saved to: {args.output_dir}/")
    print(f"   - boston_hmda/tables/    (Tables 5-7, 15-16, 19)")
    print(f"   - boston_hmda/figures/   (Figure 1)")
    print(f"   - german_credit/tables/  (Tables 8-10, 17-18, 20)")
    print(f"   - german_credit/figures/ (Figure 2)")
    print()
    print("✨ All case studies complete!")
    print()


if __name__ == "__main__":
    main()
