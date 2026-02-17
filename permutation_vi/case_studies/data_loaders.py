"""
Data loaders for real-world case studies.

This module provides functions to load and preprocess:
- Boston HMDA dataset (mortgage lending, fairness)
- German Credit dataset (credit risk, fairness)
"""

import numpy as np
from typing import Tuple, List, Optional
import warnings


def load_boston_hmda(data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load Boston HMDA dataset for debt-to-income ratio estimation.

    The dataset has 2380 observations and 12 features collected in the 1990s
    from the Home Mortgage Disclosure Act (HMDA) data.

    Parameters
    ----------
    data_path : str or None, default=None
        Path to HMDA dataset file (CSV or similar)
        If None, attempts to download from trust-free repository

    Returns
    -------
    X : np.ndarray of shape (2380, 12)
        Feature matrix
    y : np.ndarray of shape (2380,)
        Target variable (debt-to-income ratio)
    feature_names : List[str]
        List of feature names

    Notes
    -----
    Features (paper Section 5.2):
    - hir: Housing expense to income ratio
    - lvr: Loan-to-value ratio
    - ccs: Consumer credit score
    - mcs: Mortgage credit score
    - pbcr: Public bad credit record
    - dmi: Denied mortgage insurance
    - self: Self-employed
    - single: Single applicant
    - uria: Unemployment rate in area
    - condominium: Property type
    - black: Race (protected attribute)
    - deny: Application denied

    Target variable (dir): Debt-to-income ratio
    - Distribution: min=0, Q1=0.28, median=0.33, Q3=0.37, max=3
    - Right-skewed, challenging for standard regression

    References
    ----------
    Paper Section 5.2
    Munnell et al. (1996). "Mortgage lending in boston: Interpreting hmda data."
    Dataset: https://github.com/adc-trust-ai/trust-free
    """
    import pandas as pd
    from pathlib import Path

    feature_names = [
        'hir',          # Housing expense to income ratio
        'lvr',          # Loan-to-value ratio
        'ccs',          # Consumer credit score
        'mcs',          # Mortgage credit score
        'pbcr',         # Public bad credit record
        'dmi',          # Denied mortgage insurance
        'self',         # Self-employed
        'single',       # Single applicant
        'uria',         # Unemployment rate in area
        'condominium',  # Property type
        'black',        # Race (protected attribute)
        'deny'          # Application denied
    ]

    target_name = 'dir'  # Debt-to-income ratio

    # Try to load from provided path
    if data_path is not None:
        path = Path(data_path)
        if path.is_file():
            try:
                df = pd.read_csv(path)
            except Exception as e:
                raise ValueError(f"Failed to load dataset from {path}: {e}")
        else:
            raise FileNotFoundError(f"Dataset file not found: {path}")
    else:
        # Try default locations
        project_root = Path(__file__).resolve().parent.parent.parent
        default_paths = [
            project_root / 'data' / 'boston_hmda' / 'hmda_data.csv',
            project_root / 'data' / 'boston_hmda' / 'boston_hmda.csv',
            project_root / 'data' / 'hmda.csv',
        ]

        df = None
        for default_path in default_paths:
            if default_path.exists():
                try:
                    df = pd.read_csv(default_path)
                    break
                except Exception:
                    continue

        if df is None:
            raise FileNotFoundError(
                f"Boston HMDA dataset not found. Tried locations:\n" +
                "\n".join(f"  - {p}" for p in default_paths) +
                "\n\nPlease download from: https://github.com/adc-trust-ai/trust-free\n" +
                "Or specify data_path parameter with the file location."
            )

    # Validate dataset
    required_cols = feature_names + [target_name]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # Extract features and target
    X = df[feature_names].values
    y = df[target_name].values

    # Validate shape
    if X.shape[1] != 12:
        warnings.warn(f"Expected 12 features, got {X.shape[1]}")

    if X.shape[0] != 2380:
        warnings.warn(f"Expected 2380 observations, got {X.shape[0]}")

    return X, y, feature_names


def load_german_credit(data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load German Credit dataset for credit risk classification.

    The dataset has 1000 observations and 20 features (7 numerical, 13 categorical)
    used to classify loan applicants as low or high credit risk.

    Parameters
    ----------
    data_path : str or None, default=None
        Path to German Credit dataset
        If None, attempts to load from sklearn or UCI repository

    Returns
    -------
    X : np.ndarray of shape (1000, 20)
        Feature matrix (preprocessed/encoded)
    y : np.ndarray of shape (1000,)
        Binary target (0=low risk, 1=high risk)
    feature_names : List[str]
        List of feature names

    Notes
    -----
    Features (paper Section 5.3):
    - Numerical (7): Checking account, duration, credit history, purpose, credit amount,
      savings, employment duration
    - Categorical (13): Payment history, residence time, other payments, housing,
      number of credits, job, dependents, phone, foreign worker, etc.

    Target: Credit risk (binary)
    - Class imbalance: 70% low risk, 30% high risk
    - Cost-sensitive: FN costs 5× more than FP

    Protected attribute: Sex-Marital_status
    - Encoded from marital status and gender
    - Used for fairness testing in paper

    References
    ----------
    Paper Section 5.3
    UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    Alternative: https://github.com/adc-trust-ai/trust-free (with convenient encoding)
    """
    import pandas as pd
    from pathlib import Path
    from sklearn.preprocessing import OrdinalEncoder

    feature_names = [
        'Checking_acc',
        'Duration',
        'Credit_hist',
        'Purpose',
        'Credit_amount',
        'Savings',
        'Employment_time',
        'Payment_rt',
        'Sex-Marital_status',  # Protected attribute
        'Debtors/Guarantors',
        'Residence_time',
        'Properties',
        'Age',
        'Other_payments',
        'Housing',
        'Num_credits_w/this_bank',
        'Job',
        'Num_dependants',
        'Has_phone',
        'Is_foreigner'
    ]

    target_name = 'class'  # Credit risk (0=low, 1=high)

    # Try to load from provided path
    if data_path is not None:
        path = Path(data_path)
        if path.is_file():
            try:
                df = pd.read_csv(path)
            except Exception as e:
                raise ValueError(f"Failed to load dataset from {path}: {e}")
        else:
            raise FileNotFoundError(f"Dataset file not found: {path}")
    else:
        # Try default locations
        project_root = Path(__file__).resolve().parent.parent.parent
        default_paths = [
            project_root / 'data' / 'german_credit' / 'german_credit.csv',
            project_root / 'data' / 'german_credit' / 'german.csv',
            project_root / 'data' / 'german.csv',
        ]

        df = None
        for default_path in default_paths:
            if default_path.exists():
                try:
                    df = pd.read_csv(default_path)
                    break
                except Exception:
                    continue

        if df is None:
            raise FileNotFoundError(
                f"German Credit dataset not found. Tried locations:\n" +
                "\n".join(f"  - {p}" for p in default_paths) +
                "\n\nPlease download from: https://github.com/adc-trust-ai/trust-free\n" +
                "Or UCI: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)\n" +
                "Or specify data_path parameter with the file location."
            )

    # Validate dataset has required columns
    required_cols = feature_names + [target_name]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # Preprocess: Use ordinal encoding for categorical features
    # This preserves rank-based relationships needed for Spearman correlation
    X_raw = df[feature_names].copy()

    # Identify categorical columns (non-numeric; includes object, category, string)
    categorical_cols = [
        col for col in feature_names
        if not pd.api.types.is_numeric_dtype(X_raw[col])
    ]

    # Apply ordinal encoding to categorical columns
    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_raw[categorical_cols] = encoder.fit_transform(X_raw[categorical_cols])

    X = X_raw.values
    y = df[target_name].values

    # Ensure binary target (0 or 1)
    if set(np.unique(y)) == {1, 2}:
        y = y - 1  # Convert 1,2 to 0,1
    elif set(np.unique(y)) != {0, 1}:
        warnings.warn(f"Unexpected target values: {np.unique(y)}. Expected 0/1 or 1/2.")

    # Validate shape
    if X.shape[1] != 20:
        warnings.warn(f"Expected 20 features, got {X.shape[1]}")

    if X.shape[0] != 1000:
        warnings.warn(f"Expected 1000 observations, got {X.shape[0]}")

    return X, y, feature_names


def preprocess_german_credit_features(X_raw: np.ndarray, categorical_indices: List[int]) -> np.ndarray:
    """
    Preprocess German Credit features (encode categoricals).

    Parameters
    ----------
    X_raw : np.ndarray
        Raw feature matrix with mixed types
    categorical_indices : List[int]
        Indices of categorical features

    Returns
    -------
    X_processed : np.ndarray
        Processed feature matrix suitable for modeling

    Notes
    -----
    The paper uses rank-based correlations (Spearman) for SVI analysis,
    so ordinal encoding is appropriate. For fairness testing, the protected
    attribute should be kept in a meaningful encoding.
    """
    from sklearn.preprocessing import OrdinalEncoder

    X_processed = X_raw.copy()

    if len(categorical_indices) > 0:
        # Extract categorical columns
        X_cat = X_raw[:, categorical_indices]

        # Apply ordinal encoding
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_cat_encoded = encoder.fit_transform(X_cat)

        # Replace in processed array
        X_processed[:, categorical_indices] = X_cat_encoded

    return X_processed


if __name__ == "__main__":
    print("Data loader module for case studies")
    print("=" * 60)

    print("\nAvailable datasets:")
    print("1. Boston HMDA (mortgage lending)")
    print("   - 2380 observations, 12 features")
    print("   - Task: Debt-to-income ratio estimation (regression)")
    print("   - Fairness: Protected attribute 'black'")

    print("\n2. German Credit (credit risk)")
    print("   - 1000 observations, 20 features")
    print("   - Task: Credit risk classification (binary)")
    print("   - Fairness: Protected attribute 'Sex-Marital_status'")

    print("\nNote: Data loaders require actual dataset files.")
    print("Download from: https://github.com/adc-trust-ai/trust-free")
