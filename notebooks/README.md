# Jupyter Notebooks for Case Studies

This directory contains interactive Jupyter notebooks for exploring the case study results from the paper "One Permutation Is All You Need" (arXiv:2512.13892v2).

## Available Notebooks

### 1. `case_studies_tutorial.ipynb` - **Start Here!**
Complete tutorial covering both case studies:
- Boston HMDA (Section 5.2) - Mortgage lending with fairness testing
- German Credit (Section 5.3) - Credit risk with protected attributes
- Step-by-step reproduction of all key results
- Interactive visualizations and explanations

### Features

✅ **Interactive exploration** - Run experiments cell-by-cell
✅ **Visualizations** - Correlation heatmaps, importance comparisons
✅ **Educational** - Detailed explanations and interpretations
✅ **Reproducible** - Fixed random seeds for exact paper reproduction

## Getting Started

### 1. Install Jupyter

```bash
# Using uv (recommended)
uv pip install jupyter ipykernel

# Or using pip
pip install jupyter ipykernel
```

### 2. Download Datasets

Download the required datasets:

**Boston HMDA:**
- Source: https://github.com/adc-trust-ai/trust-free
- Place in: `data/boston_hmda/hmda_data.csv`

**German Credit:**
- Source: https://github.com/adc-trust-ai/trust-free (or UCI ML Repository)
- Place in: `data/german_credit/german_credit.csv`

### 3. Launch Jupyter

```bash
# From project root
jupyter notebook notebooks/
```

Then open `case_studies_tutorial.ipynb` and run the cells!

## What You'll Learn

### Boston HMDA Case Study (Section 5.2)
- **Direct VI**: Compare optimal vs approximate vs Breiman methods
- **Performance**: See 14-105× speedup with zero variance
- **Fairness**: Discover indirect reliance on 'black' attribute (0.16%-0.44%)
- **Proxies**: Identify correlated features (lvr, deny, ccs, condominium)

### German Credit Case Study (Section 5.3)
- **Classification**: Apply DVI to binary classification with probabilities
- **SVI**: Measure systemic importance of 'Sex-Marital_status' (3.24%)
- **Safeguarding**: Verify that exclusion prevents indirect reliance (0%)
- **Comparison**: ℓ1-Logistic vs unregularized Logistic vs RF

## Results Reproduced

The notebooks reproduce:
- ✅ **Tables 5-10**: Master performance and Direct VI comparisons
- ✅ **Tables 15-18**: Results with default scoring (Appendix A)
- ✅ **Tables 19-20**: Stability tests (Appendix B)
- ✅ **Figures 1-2**: Correlation heatmaps
- ✅ **SVI Results**: All fairness testing scenarios

## Tips

1. **Run cells in order** - Each cell builds on previous ones
2. **Check dataset paths** - Adjust if your data is in a different location
3. **Experiment** - Try different models, parameters, or features
4. **Save outputs** - Uncomment save_path arguments to save figures

## Troubleshooting

**Import errors:**
```python
# Make sure project root is in path
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
```

**Dataset not found:**
- Download from https://github.com/adc-trust-ai/trust-free
- Or specify custom path: `load_boston_hmda(data_path='your/path/here.csv')`

**Slow performance:**
- Reduce n_folds (e.g., 5 instead of 10)
- Use smaller n_repeats for stability tests
- Use approximate permutation instead of optimal

## Next Steps

After exploring the notebooks:
1. Run full experiments: `uv run python permutation_vi/case_studies/run_all_case_studies.py`
2. Read the paper: https://arxiv.org/abs/2512.13892
3. Try on your own datasets!

## Questions?

- Check the main [README](../README.md) for project overview
- See [CLAUDE.md](../.claude/CLAUDE.md) for code architecture
- Read the paper for theoretical details
