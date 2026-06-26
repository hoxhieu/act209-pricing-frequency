"""
Data preparation utilities for the ACT209 pricing frequency project.

This module contains reusable functions to:

1. load the raw insurance frequency dataset;
2. check the actuarial target variables;
3. clean invalid observations;
4. create the empirical frequency target;
5. identify feature columns;
6. save and reload the cleaned dataset.

Actuarial notation
------------------
For each policy i:

    N_i = observed claim count
    E_i = exposure
    X_i = explanatory variables

For frequency modeling:

    empirical_frequency_i = N_i / E_i

The expected claim count is:

    mu_i = E_i * lambda_i
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CLAIM_COL = "ClaimNb"
DEFAULT_EXPOSURE_COL = "Exposure"
DEFAULT_FREQUENCY_COL = "claim_frequency"
DEFAULT_ID_COL = "IDpol"


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Load the raw CSV dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset loaded as a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the loaded dataset is empty.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"The raw data file is empty: {path}")

    return df


def check_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """
    Check that required columns are present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    required_columns : iterable of str
        Required column names.

    Raises
    ------
    ValueError
        If at least one required column is missing.
    """
    required_columns = list(required_columns)
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def check_claim_and_exposure(
    df: pd.DataFrame,
    claim_col: str = DEFAULT_CLAIM_COL,
    exposure_col: str = DEFAULT_EXPOSURE_COL,
) -> None:
    """
    Check the basic actuarial validity of claim count and exposure.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    claim_col : str, default="ClaimNb"
        Name of the observed claim count column.
    exposure_col : str, default="Exposure"
        Name of the exposure column.

    Raises
    ------
    ValueError
        If required columns are missing, contain missing values, or violate
        actuarial constraints.

    Notes
    -----
    The required constraints are:

        Exposure > 0
        ClaimNb >= 0

    Claim count should also be integer-valued.
    """
    check_required_columns(df, [claim_col, exposure_col])

    if df[claim_col].isna().any():
        n_missing = int(df[claim_col].isna().sum())
        raise ValueError(f"Missing values detected in {claim_col}: {n_missing}")

    if df[exposure_col].isna().any():
        n_missing = int(df[exposure_col].isna().sum())
        raise ValueError(f"Missing values detected in {exposure_col}: {n_missing}")

    if (df[exposure_col] <= 0).any():
        n_bad = int((df[exposure_col] <= 0).sum())
        raise ValueError(f"Exposure must be strictly positive. Invalid rows: {n_bad}")

    if (df[claim_col] < 0).any():
        n_bad = int((df[claim_col] < 0).sum())
        raise ValueError(f"Claim count must be non-negative. Invalid rows: {n_bad}")

    non_integer_claims = ((df[claim_col] % 1) != 0).sum()

    if non_integer_claims > 0:
        raise ValueError(
            "Claim count should be integer-valued. "
            f"Non-integer values detected: {int(non_integer_claims)}"
        )


def summarize_data_quality(
    df: pd.DataFrame,
    claim_col: str = DEFAULT_CLAIM_COL,
    exposure_col: str = DEFAULT_EXPOSURE_COL,
) -> dict:
    """
    Return basic data quality diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    claim_col : str, default="ClaimNb"
        Claim count column.
    exposure_col : str, default="Exposure"
        Exposure column.

    Returns
    -------
    dict
        Data quality summary.
    """
    check_required_columns(df, [claim_col, exposure_col])

    total_claim_count = float(df[claim_col].sum(skipna=True))
    total_exposure = float(df[exposure_col].sum(skipna=True))

    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "missing_claim_count": int(df[claim_col].isna().sum()),
        "missing_exposure": int(df[exposure_col].isna().sum()),
        "exposure_leq_zero": int((df[exposure_col] <= 0).sum()),
        "claim_count_negative": int((df[claim_col] < 0).sum()),
        "total_claim_count": total_claim_count,
        "total_exposure": total_exposure,
        "global_claim_frequency": (
            total_claim_count / total_exposure if total_exposure > 0 else None
        ),
    }

    return summary


def clean_frequency_data(
    df: pd.DataFrame,
    claim_col: str = DEFAULT_CLAIM_COL,
    exposure_col: str = DEFAULT_EXPOSURE_COL,
    frequency_col: str = DEFAULT_FREQUENCY_COL,
) -> pd.DataFrame:
    """
    Clean the frequency dataset and create the empirical frequency target.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataset.
    claim_col : str, default="ClaimNb"
        Observed claim count column.
    exposure_col : str, default="Exposure"
        Exposure column.
    frequency_col : str, default="claim_frequency"
        Name of the empirical frequency column to create.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with an additional frequency column:

            claim_frequency = ClaimNb / Exposure

    Notes
    -----
    Invalid observations are removed only for the core actuarial constraints:

        Exposure > 0
        ClaimNb >= 0

    More advanced preprocessing decisions, such as handling rare categories,
    must be performed later inside the preprocessing pipeline to avoid
    data leakage.
    """
    check_required_columns(df, [claim_col, exposure_col])

    df_clean = df.copy()

    df_clean = df_clean[df_clean[exposure_col].notna()].copy()
    df_clean = df_clean[df_clean[claim_col].notna()].copy()
    df_clean = df_clean[df_clean[exposure_col] > 0].copy()
    df_clean = df_clean[df_clean[claim_col] >= 0].copy()

    check_claim_and_exposure(
        df_clean,
        claim_col=claim_col,
        exposure_col=exposure_col,
    )

    df_clean[frequency_col] = df_clean[claim_col] / df_clean[exposure_col]

    return df_clean


def get_feature_columns(
    df: pd.DataFrame,
    claim_col: str = DEFAULT_CLAIM_COL,
    exposure_col: str = DEFAULT_EXPOSURE_COL,
    frequency_col: str = DEFAULT_FREQUENCY_COL,
    id_col: str = DEFAULT_ID_COL,
) -> list[str]:
    """
    Return explanatory feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing target, exposure, and explanatory variables.
    claim_col : str, default="ClaimNb"
        Claim count column.
    exposure_col : str, default="Exposure"
        Exposure column.
    frequency_col : str, default="claim_frequency"
        Empirical frequency column.
    id_col : str, default="IDpol"
        Policy identifier column.

    Returns
    -------
    list[str]
        Feature column names.

    Notes
    -----
    The following columns are excluded from the feature set:

        - policy identifier,
        - observed claim count,
        - exposure,
        - empirical frequency target.

    Exposure is not used as an ordinary feature in the recommended frequency
    modeling approach. Instead, it is used as sample weight.
    """
    excluded = {claim_col, exposure_col, frequency_col, id_col}
    return [col for col in df.columns if col not in excluded]


def split_numeric_categorical_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    """
    Split feature columns into numeric and categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    feature_columns : list[str]
        Candidate explanatory variables.

    Returns
    -------
    tuple[list[str], list[str]]
        Numeric feature names and categorical feature names.
    """
    numeric_features = []
    categorical_features = []

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features


def save_processed_data(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save the processed dataset to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset.
    path : str or pathlib.Path
        Output CSV path.

    Notes
    -----
    The parent directory is created if it does not already exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_processed_data(path: str | Path) -> pd.DataFrame:
    """
    Load the processed dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the processed CSV file.

    Returns
    -------
    pd.DataFrame
        Processed dataset.

    Raises
    ------
    FileNotFoundError
        If the processed file does not exist.
    ValueError
        If the processed file is empty.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"The processed data file is empty: {path}")

    return df