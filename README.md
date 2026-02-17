# One Permutation Variable Importance

An unofficial vibe-coded Python implementation of **"One Permutation Is All You Need: Fast, Reliable Variable Importance and Model Stress-Testing"** ([arXiv:2512.13892v2](https://arxiv.org/abs/2512.13892)).

This package provides deterministic, efficient variable importance methods that are **10-100× faster** than traditional permutation-based approaches while achieving comparable or better accuracy.

## 🎯 Key Features

- **Direct Variable Importance (DVI)**: Single deterministic permutation replaces multiple random permutations
- **Systemic Variable Importance (SVI)**: Correlation-based propagation for fairness auditing and stress-testing
- **Deterministic**: Zero variance, perfectly reproducible results
- **Fast**: 10-100× speedup over scikit-learn's baseline
- **Model-agnostic**: Works with any model having `predict()` or `predict_proba()`
- **Well-documented**: Comprehensive examples and API documentation

## 📦 Installation

This project uses **uv** for dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt

# Or use standard pip
pip install -r requirements.txt

# The package is ready to use
cd permutation_vi
```

**Dependencies**: numpy, scipy, scikit-learn, pandas, matplotlib, seaborn
**Optional**: jupyter, ipykernel (for notebooks), trust-free (for TRUST model)

## 🚀 Quick Start

### Direct Variable Importance

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from permutation_vi import DirectVariableImportance

# Generate data and train model
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5)
model = RandomForestRegressor(random_state=42).fit(X, y)

# Compute importance with single optimal permutation
dvi = DirectVariableImportance(
    permutation_type='optimal',  # or 'approximate'
    scoring_metric='mae'         # or 'mse', 'rmse'
)
importance_scores = dvi.fit(model, X)

# View top features
top_features = dvi.get_top_features(n=5)
for idx, score in top_features:
    print(f"Feature {idx}: {score:.4f}")
```

### Systemic Variable Importance (Fairness Auditing)

```python
from permutation_vi import SystemicVariableImportance

# Detect indirect reliance on protected attributes
svi = SystemicVariableImportance(
    correlation_method='spearman',
    alpha=0.01  # Significance level for correlations
)

# Get decomposition: systemic = direct + indirect
systemic, direct, indirect = svi.fit(model, X, return_decomposition=True)

# Check if a feature has indirect importance (proxy effects)
feature_idx = 0
print(f"Direct importance:   {direct[feature_idx]:.4f}")
print(f"Indirect importance: {indirect[feature_idx]:.4f}")
print(f"Systemic importance: {systemic[feature_idx]:.4f}")

# Find proxy features
proxies = svi.get_feature_proxies(feature_idx)
print(f"Feature {feature_idx} has {len(proxies)} significant proxies")
```

## 📋 Case Study Implementation Features

The complete case study implementations include:

### Experiment Orchestration
- **Master model training**: 10-fold cross-validation with row randomization (paper protocol)
- **Direct VI comparison**: Four methods (DVI-Optimal, DVI-Approximate, Breiman B=1, Breiman B=10)
- **Ground truth computation**: Random Forest importance averaged over 100 runs
- **Stability testing**: Track top-5 features across 10 runs with different random seeds
- **SVI fairness scenarios**: Multiple configurations testing protected attribute reliance

### Models Implemented
- **TRUST** (or RelaxedLasso approximation): Sparse linear with feature selection
- **OLS/Logistic**: Unregularized baselines
- **ℓ1-Logistic**: Sparse classification with L1 penalty
- **Random Forest**: Ensemble baseline for ground truth

### Outputs Generated
- **Tables**: Formatted CSV files matching paper Tables 5-10, 15-20
- **Figures**: Correlation heatmaps (Figures 1-2) with Spearman correlation and significance thresholds
- **Metrics**: R², MAE, PR-AUC, ROC-AUC, ground-truth correlation, speedup ratios
- **SVI decomposition**: Direct, indirect, and systemic importance for fairness auditing

### CLI Options
```bash
# Customize experiments
uv run python permutation_vi/case_studies/run_boston_hmda.py \
  --data-path data/boston_hmda/hmda_data.csv \
  --output-dir custom_results/ \
  --random-state 42 \
  --experiments master dvi stability svi heatmap
```

## 📊 Examples

Run the included examples to see the implementation in action:

```bash
# Basic examples (synthetic data)
uv run python permutation_vi/examples/example_dvi.py              # Direct Variable Importance
uv run python permutation_vi/examples/example_svi.py              # Systemic Importance
uv run python permutation_vi/examples/example_comparison.py       # Benchmark vs sklearn
uv run python permutation_vi/examples/example_benchmark.py        # Synthetic benchmarks

# Case studies (requires datasets)
uv run python permutation_vi/case_studies/run_boston_hmda.py      # Boston HMDA (Section 5.2)
uv run python permutation_vi/case_studies/run_german_credit.py    # German Credit (Section 5.3)
uv run python permutation_vi/case_studies/run_all_case_studies.py # Both case studies

# Interactive notebooks
jupyter notebook notebooks/case_studies_tutorial.ipynb            # Tutorial notebook
```

## 🔬 Reproducing Paper Results

This implementation includes the full test suite from the paper for validation and reproducibility.

### Synthetic Benchmark Suite (Section 3.2)

The paper tests across **192 scenarios** varying:
- Sample sizes: n ∈ {100, 1000, 10000}
- Dimensionality: p ∈ {10, 100}
- Noise levels: σ_ε ∈ {0.1, 5}
- Feature correlations: ρ ∈ {0, 0.3}
- Response types: Linear, Friedman function
- Task types: Regression, classification

**Run quick validation** (8 scenarios, 5 repetitions):
```python
from permutation_vi.benchmarks import create_scenario_grid, run_single_scenario, aggregate_results

# Get subset of scenarios
scenarios = create_scenario_grid()
test_scenarios = [s for s in scenarios if s['n'] == 1000 and s['p'] == 10]

# Run benchmarks
results = [run_single_scenario(s, n_repetitions=5) for s in test_scenarios]

# Aggregate results
summary = aggregate_results(results)
print(summary)
```

**Full benchmark** (192 scenarios, 50 repetitions - several hours runtime):
```bash
uv run python permutation_vi/examples/example_benchmark.py --full
```

**Expected results** (from paper Tables 1-4):
- Ground-truth correlation: >0.95
- Speedup: 1.7-2× for small n, 10-100× for large n
- Zero variance (perfectly deterministic)

### Real-World Case Studies (Sections 5.2 & 5.3)

Complete implementations reproducing all paper results including tables, figures, and fairness testing scenarios.

#### Boston HMDA Dataset (Section 5.2)

Debt-to-income ratio estimation with fairness testing:
- **Dataset**: 2380 observations, 12 features
- **Task**: Predict debt-to-income ratio
- **Protected attribute**: "black"
- **Master models**: TRUST (sparse linear), OLS, Random Forest

**Results reproduced**:
- ✅ Table 5: Master model performance (R², MAE, nonzero coefficients)
- ✅ Tables 6-7: Direct VI comparisons (ground-truth correlation, speedup, max/mean score diff)
- ✅ Tables 15-16: Results with default MAE scoring (Appendix A)
- ✅ Table 19: Stability test showing deterministic DVI vs variable Breiman (Appendix B)
- ✅ Figure 1: Spearman correlation heatmap with threshold α=0.01
- ✅ SVI fairness testing: OLS scenario (0.44% systemic via proxies) and TRUST safeguarding (0.16% residual)

**Key findings**:
- Systemic importance of "black" is ~5× its direct importance (via proxy features: lvr, deny, ccs, condominium)
- DVI achieves >0.94 correlation with ground truth
- 14-105× faster than sklearn baseline
- Zero variance (perfectly deterministic) across runs

**Download dataset**:
```bash
# Place in data/boston_hmda/hmda_data.csv
# Source: https://github.com/adc-trust-ai/trust-free
```

**Run case study**:
```bash
# Single case study
uv run python permutation_vi/case_studies/run_boston_hmda.py --data-path data/boston_hmda/

# Results saved to: results/boston_hmda/tables/ and results/boston_hmda/figures/
```

#### German Credit Dataset (Section 5.3)

Credit risk classification with fairness testing:
- **Dataset**: 1000 observations, 20 features (7 numerical, 13 categorical)
- **Task**: Binary classification (credit risk prediction)
- **Protected attribute**: "Sex-Marital_status"
- **Master models**: ℓ1-Logistic, Logistic (unregularized), Random Forest

**Results reproduced**:
- ✅ Table 8: Master model performance (PR-AUC, ROC-AUC, nonzero coefficients)
- ✅ Tables 9-10: Direct VI comparisons with sparse and unregularized logistic
- ✅ Tables 17-18: Results with default scoring (Appendix A)
- ✅ Table 20: Stability test (Appendix B)
- ✅ Figure 2: Correlation heatmap for categorical features
- ✅ SVI fairness testing: Unregularized logistic (3.24% systemic) vs sparse without protected (0% - safeguarding works)

**Key findings**:
- Systemic importance of "Sex-Marital_status": 3.24% total (2.09% direct + 1.15% indirect)
- Safeguarding by exclusion: Sparse model without protected attribute has 0% systemic importance
- Perfect stability: DVI returns identical top-5 features across 10 runs
- Sklearn Breiman shows 9-10 distinct top-5 rankings across runs

**Download dataset**:
```bash
# Place in data/german_credit/german_credit.csv
# Sources: UCI ML Repository or https://github.com/adc-trust-ai/trust-free
```

**Run case study**:
```bash
# Single case study
uv run python permutation_vi/case_studies/run_german_credit.py --data-path data/german_credit/

# Results saved to: results/german_credit/tables/ and results/german_credit/figures/
```

#### Run All Case Studies

Execute both case studies sequentially with a single command:
```bash
uv run python permutation_vi/case_studies/run_all_case_studies.py

# Options:
# --output-dir custom_results/     # Change output directory
# --skip-boston                     # Skip Boston HMDA
# --skip-german                     # Skip German Credit
# --random-state 42                 # Set random seed
```

#### Interactive Jupyter Notebooks

Explore the case studies interactively with step-by-step tutorials:

```bash
# Install Jupyter
uv pip install jupyter ipykernel

# Launch notebooks
jupyter notebook notebooks/

# Open: case_studies_tutorial.ipynb
```

**Notebook features**:
- 📊 Complete walkthrough of both case studies
- 🎯 Reproduce Tables 5-10, Figures 1-2, and all SVI scenarios
- 📈 Interactive visualizations (correlation heatmaps, importance comparisons)
- 🔬 Educational explanations linking code to paper sections
- ⚙️ Configurable experiments (change models, parameters, features)

See [notebooks/README.md](notebooks/README.md) for detailed instructions.

## 🧪 What the Paper Proposes

### The Problem with Traditional Permutation Importance

Traditional Breiman-style permutation importance:
1. Randomly permutes each feature B times (typically B=10)
2. Averages the results
3. Has non-zero variance → stochastic instability
4. Requires computational overhead (B permutations per feature)

### The Solution: One Optimal Permutation

**Key Insight**: Replace B random permutations with a single deterministic optimal permutation.

**Optimal Permutation**: Cyclic shift by ⌊n/2⌋
- **Rank-shift** (optimal): O(n log n) due to sorting
- **Index-shift** (approximate): O(n) for speed

**Theoretical Guarantee** (Proposition 1):
- Maximizes the minimum circular displacement
- Achieves max-min optimality: m(π) = ⌊n/2⌋

**Practical Benefits** (Proposition 2):
- Lower MSE when: |bias_difference| < σ/√B
- Zero variance (deterministic)
- 10-100× faster

### Extension: Systemic Variable Importance

**Purpose**: Detect indirect reliance on features through correlations

**Use Cases**:
1. **Fairness Auditing**: Find if a model relies on protected attributes via proxies
2. **Stress Testing**: Understand how shocks propagate through correlated features

**Method**:
1. Compute correlation threshold via permutation testing (FWER control)
2. Build correlation network (significant correlations only)
3. Propagate perturbations: x'_k = x_k + Σ_j cor(k,j) × (x'_j - x_j)
4. Decompose: systemic = direct + indirect

## 📁 Project Structure

```
permutation_vi/
├── __init__.py                      # Package exports
├── direct_importance.py             # DirectVariableImportance class
├── systemic_importance.py           # SystemicVariableImportance class
├── utils.py                         # Helper functions (Friedman, RF ground truth)
├── benchmarks/
│   ├── __init__.py                  # Benchmark exports
│   ├── synthetic_suite.py           # 192-scenario test suite
│   └── reporting.py                 # Visualization and tables
├── case_studies/
│   ├── __init__.py                  # Case study exports
│   ├── data_loaders.py              # Dataset loaders (Boston HMDA, German Credit)
│   ├── models.py                    # Model factory (TRUST/RelaxedLasso, logistic, RF)
│   ├── cv_utils.py                  # Cross-validation utilities
│   ├── metrics.py                   # Metric computations (PR-AUC, ROC-AUC, Brier)
│   ├── table_generators.py          # Format output tables (Tables 5-10, 15-20)
│   ├── visualizations.py            # Plotting utilities (heatmaps, comparisons)
│   ├── boston_hmda.py               # Boston HMDA experiment orchestrator
│   ├── german_credit.py             # German Credit experiment orchestrator
│   ├── run_boston_hmda.py           # CLI runner for Boston case study
│   ├── run_german_credit.py         # CLI runner for German Credit
│   └── run_all_case_studies.py      # Sequential runner for all case studies
└── examples/
    ├── example_dvi.py               # DVI demonstration
    ├── example_svi.py               # SVI and fairness auditing
    ├── example_comparison.py        # Benchmark vs sklearn
    └── example_benchmark.py         # Run synthetic benchmarks

notebooks/
├── README.md                        # Notebook documentation
└── case_studies_tutorial.ipynb     # Interactive case studies tutorial

data/                                # Dataset storage (create manually)
├── boston_hmda/
│   └── hmda_data.csv               # Boston HMDA dataset
└── german_credit/
    └── german_credit.csv           # German Credit dataset

results/                             # Generated outputs (auto-created)
├── boston_hmda/
│   ├── tables/                     # Tables 5-7, 15-16, 19
│   └── figures/                    # Figure 1 (correlation heatmap)
└── german_credit/
    ├── tables/                     # Tables 8-10, 17-18, 20
    └── figures/                    # Figure 2 (correlation heatmap)
```

## 📖 API Reference

### DirectVariableImportance

**Parameters**:
- `permutation_type` (str): `'optimal'` or `'approximate'` (default: `'optimal'`)
- `scoring_metric` (str): `'mae'`, `'mse'`, or `'rmse'` (default: `'mae'`)

**Methods**:
- `fit(model, X, y=None)`: Compute importance scores
- `get_top_features(n=5)`: Get top n important features
- `get_feature_importance(feature_idx)`: Get score for specific feature

**Attributes**:
- `importances_`: Normalized importance scores (sum to 1)
- `n_features_`: Number of features
- `feature_names_`: Feature names (if provided)

### SystemicVariableImportance

Inherits from `DirectVariableImportance` with additional parameters:

**Parameters**:
- `correlation_method` (str): `'spearman'` or `'pearson'` (default: `'spearman'`)
- `alpha` (float): Significance level for threshold (default: `0.01`)

**Methods**:
- `fit(model, X, y=None, return_decomposition=True)`: Returns (systemic, direct, indirect)
- `get_correlation_network(threshold=None)`: Get binary adjacency matrix
- `get_feature_proxies(feature_idx)`: Find proxy features

**Attributes**:
- `systemic_importances_`: Systemic scores
- `direct_importances_`: Direct component
- `indirect_importances_`: Indirect component (network effect)
- `correlation_matrix_`: Feature correlation matrix
- `correlation_threshold_`: Estimated τ for significance
- `significant_correlations_`: Binary matrix of significant correlations

## 🔬 Performance Benchmarks

Based on the paper's experiments across 192 scenarios:

| Metric | DVI (Optimal) | DVI (Approximate) | Sklearn (B=1) | Sklearn (B=10) |
|--------|---------------|-------------------|---------------|----------------|
| **Speed** | ⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡ | ⚡ |
| **Correlation with ground truth** | >0.99 | >0.99 | >0.94 | >0.94 |
| **Variance** | 0 (deterministic) | 0 (deterministic) | High | Medium |
| **Speedup vs Sklearn (B=10)** | ~20× | ~100× | ~10× | 1× |

## 📚 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{dorador2025one,
  title={One Permutation Is All You Need: Fast, Reliable Variable Importance and Model Stress-Testing},
  author={Dorador, Albert},
  journal={arXiv preprint arXiv:2512.13892},
  year={2025}
}
```

## 🎓 Educational Purpose

This implementation is designed for learning and understanding the paper's concepts:

- **Clean, readable code** with extensive docstrings
- **Simple examples** using synthetic data
- **Comments** explaining key algorithmic steps
- **Follows the paper** closely for easy cross-reference

## 🔍 Implementation Status

This educational implementation includes:

- ✅ Core DVI and SVI algorithms (Section 3)
- ✅ Both optimal and approximate permutations (Proposition 1)
- ✅ Simple synthetic examples
- ✅ **Full 192-scenario benchmark suite** (Section 3.2)
- ✅ **Friedman function generator** (nonlinear benchmark)
- ✅ **Visualization utilities** (correlation heatmaps, importance plots, speedup comparisons)
- ✅ **Reporting functions** (tables matching paper format)
- ✅ **Boston HMDA case study** (Section 5.2) - Complete with all tables, figures, SVI scenarios
- ✅ **German Credit case study** (Section 5.3) - Complete with all tables, figures, SVI scenarios
- ✅ **Interactive Jupyter notebooks** - Tutorial covering both case studies
- ✅ **Model factory** - TRUST/RelaxedLasso, sparse logistic, random forest
- ✅ **CV utilities** - 10-fold cross-validation with paper protocol
- ✅ **Fairness testing** - Systemic importance decomposition and proxy detection
- ❌ Optional prescreening feature (omitted for clarity)

## 🛠️ Technical Details

### Permutation Strategies

**Optimal (Rank-shift)**:
```python
def optimal_permutation(x):
    ranks = np.argsort(np.argsort(x))  # Get ranks
    n = len(x)
    shift = n // 2
    shifted_ranks = (ranks + shift) % n
    sorted_x = np.sort(x)
    return sorted_x[shifted_ranks]
```

**Approximate (Index-shift)**:
```python
def approximate_permutation(x):
    n = len(x)
    shift = n // 2
    return np.roll(x, shift)
```

### Correlation Threshold Estimation

```python
# Permute columns independently (null hypothesis: independence)
X_permuted = independently_permute_columns(X)

# Compute all pairwise correlations
correlations = compute_all_pairwise(X_permuted)

# Take (1-α) quantile
threshold = np.quantile(abs(correlations), 1 - alpha)
```

## ⚡ Performance Tips

1. **Use approximate permutation** for very large datasets (>100K samples)
2. **Use MAE scoring** (paper recommendation) unless you need MSE compatibility
3. **Adjust alpha** in SVI based on desired FWER control
4. **Use Spearman correlation** (handles non-linear monotonic relationships)

## 🤝 Contributing

This is an educational implementation. Contributions welcome:

- Bug fixes
- Documentation improvements
- Additional examples
- Performance optimizations

## 📄 License

This implementation is provided for educational purposes. The original paper and method are by Albert Dorador.

## 🙏 Acknowledgments

- **Albert Dorador** for the original paper and method
- **Leo Breiman** for the foundational permutation importance concept
- **scikit-learn** community for the baseline implementation

## 📧 Contact

For questions about the implementation, please open an issue on the repository.

For questions about the paper/method, contact the paper's author.

---

**Ready to get started?**

```bash
# 1. Try basic examples with synthetic data
uv run python permutation_vi/examples/example_dvi.py
uv run python permutation_vi/examples/example_svi.py
uv run python permutation_vi/examples/example_comparison.py

# 2. Explore interactively with Jupyter
jupyter notebook notebooks/case_studies_tutorial.ipynb

# 3. Reproduce full paper results (requires datasets)
uv run python permutation_vi/case_studies/run_all_case_studies.py
```

**Next steps**:
1. 📖 Read [CLAUDE.md](.claude/CLAUDE.md) for architecture and implementation details
2. 📓 Explore [notebooks/README.md](notebooks/README.md) for interactive tutorials
3. 📊 Download datasets and reproduce Tables 5-10 and Figures 1-2
4. 🧪 Run the full 192-scenario benchmark suite
5. 🔬 Apply DVI/SVI to your own datasets!

Happy learning! 🎓
