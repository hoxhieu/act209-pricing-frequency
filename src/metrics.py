"""
Evaluation metrics for insurance claim frequency models.

This module contains reusable functions to evaluate frequency models on the
claim count scale.

Actuarial notation
------------------
For each policy i:

    N_i = observed claim count
    E_i = exposure
    lambda_hat_i = predicted annual claim frequency
    mu_hat_i = E_i * lambda_hat_i = predicted expected claim count

The main evaluation quantities are:

    - mean Poisson deviance,
    - train-test gap,
    - calibration by predicted-risk deciles,
    - lift by predicted-risk deciles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-15


def compute_expected_claims(predicted_frequency, exposure) -> np.ndarray:
    """
    Convert predicted annual claim frequency into expected claim count.

    Parameters
    ----------
    predicted_frequency : array-like
        Predicted annual claim frequency lambda_hat_i.
    exposure : array-like
        Exposure E_i.

    Returns
    -------
    np.ndarray
        Predicted expected claim count:

            mu_hat_i = E_i * lambda_hat_i

    Raises
    ------
    ValueError
        If inputs have different shapes, if exposure is not strictly positive,
        or if predicted frequencies are negative.
    """
    predicted_frequency = np.asarray(predicted_frequency, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    if predicted_frequency.shape != exposure.shape:
        raise ValueError("predicted_frequency and exposure must have the same shape.")

    if np.any(exposure <= 0):
        raise ValueError("Exposure must be strictly positive.")

    if np.any(predicted_frequency < 0):
        raise ValueError("Predicted frequencies must be non-negative.")

    return exposure * predicted_frequency


def mean_poisson_deviance_count(y_count, mu_pred) -> float:
    """
    Compute the mean Poisson deviance on the claim count scale.

    Parameters
    ----------
    y_count : array-like
        Observed claim counts N_i. Must be non-negative.
    mu_pred : array-like
        Predicted expected claim counts mu_hat_i. Must be strictly positive.

    Returns
    -------
    float
        Mean Poisson deviance.

    Notes
    -----
    The observation-level Poisson deviance is:

        D_i = 2 * [N_i log(N_i / mu_hat_i) - (N_i - mu_hat_i)]

    with the convention:

        0 * log(0 / mu_hat_i) = 0.
    """
    y_count = np.asarray(y_count, dtype=float)
    mu_pred = np.asarray(mu_pred, dtype=float)

    if y_count.shape != mu_pred.shape:
        raise ValueError("y_count and mu_pred must have the same shape.")

    if np.any(y_count < 0):
        raise ValueError("Observed claim counts must be non-negative.")

    if np.any(mu_pred <= 0):
        raise ValueError("Predicted expected claim counts must be strictly positive.")

    term = np.zeros_like(y_count, dtype=float)

    positive_mask = y_count > 0

    term[positive_mask] = (
        y_count[positive_mask]
        * np.log(y_count[positive_mask] / mu_pred[positive_mask])
    )

    deviance = 2.0 * (term - (y_count - mu_pred))

    return float(np.mean(deviance))


def mean_poisson_deviance_from_frequency(y_count, exposure, predicted_frequency) -> float:
    """
    Compute mean Poisson deviance from predicted annual frequency.

    Parameters
    ----------
    y_count : array-like
        Observed claim counts N_i.
    exposure : array-like
        Exposure E_i.
    predicted_frequency : array-like
        Predicted annual frequency lambda_hat_i.

    Returns
    -------
    float
        Mean Poisson deviance computed on:

            mu_hat_i = E_i * lambda_hat_i.
    """
    mu_pred = compute_expected_claims(predicted_frequency, exposure)

    # Numerical safety: Poisson deviance requires strictly positive predictions.
    mu_pred = np.maximum(mu_pred, EPSILON)

    return mean_poisson_deviance_count(y_count, mu_pred)


def train_test_gap(train_loss: float, test_loss: float) -> float:
    """
    Compute the train-test gap for a loss.

    Parameters
    ----------
    train_loss : float
        Loss on the training set.
    test_loss : float
        Loss on the test set.

    Returns
    -------
    float
        test_loss - train_loss.

    Interpretation
    --------------
    Since Poisson deviance is a loss, a large positive gap means that the
    model performs much better on training data than on test data, which is
    a warning sign of overfitting.
    """
    return float(test_loss - train_loss)


def global_observed_frequency(y_count, exposure) -> float:
    """
    Compute global observed annual claim frequency.

    Parameters
    ----------
    y_count : array-like
        Observed claim counts N_i.
    exposure : array-like
        Exposure E_i.

    Returns
    -------
    float
        Global observed annual claim frequency:

            sum_i N_i / sum_i E_i.
    """
    y_count = np.asarray(y_count, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    if y_count.shape != exposure.shape:
        raise ValueError("y_count and exposure must have the same shape.")

    if np.any(exposure <= 0):
        raise ValueError("Exposure must be strictly positive.")

    total_exposure = exposure.sum()

    if total_exposure <= 0:
        raise ValueError("Total exposure must be strictly positive.")

    return float(y_count.sum() / total_exposure)


def calibration_table_by_decile(
    y_count,
    exposure,
    predicted_frequency,
    n_groups: int = 10,
) -> pd.DataFrame:
    """
    Build a calibration and lift table by predicted-risk groups.

    Groups are defined by sorting predicted annual frequencies and splitting
    them into quantile-based groups.

    Parameters
    ----------
    y_count : array-like
        Observed claim counts N_i.
    exposure : array-like
        Exposure E_i.
    predicted_frequency : array-like
        Predicted annual frequency lambda_hat_i.
    n_groups : int, default=10
        Number of predicted-risk groups. Use 10 for deciles.

    Returns
    -------
    pd.DataFrame
        A table with, for each risk group:

        - total exposure,
        - observed claim count,
        - predicted claim count,
        - observed frequency,
        - predicted frequency,
        - calibration ratio O/P,
        - observed lift.

    Notes
    -----
    The lift uses observed frequencies, but the groups are defined by the
    model predictions. Therefore, lift evaluates the ranking induced by the
    model.
    """
    y_count = np.asarray(y_count, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    predicted_frequency = np.asarray(predicted_frequency, dtype=float)

    if not (y_count.shape == exposure.shape == predicted_frequency.shape):
        raise ValueError("y_count, exposure and predicted_frequency must have the same shape.")

    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")

    if np.any(y_count < 0):
        raise ValueError("Observed claim counts must be non-negative.")

    if np.any(exposure <= 0):
        raise ValueError("Exposure must be strictly positive.")

    if np.any(predicted_frequency < 0):
        raise ValueError("Predicted frequencies must be non-negative.")

    mu_pred = compute_expected_claims(predicted_frequency, exposure)
    mu_pred = np.maximum(mu_pred, EPSILON)

    df_eval = pd.DataFrame(
        {
            "claim_count": y_count,
            "exposure": exposure,
            "predicted_frequency": predicted_frequency,
            "predicted_claim_count": mu_pred,
        }
    )

    df_eval = df_eval.sort_values("predicted_frequency").reset_index(drop=True)

    # qcut creates quantile-based groups. The rank(method="first") avoids
    # problems when many observations have identical predicted frequencies.
    df_eval["risk_group"] = (
        pd.qcut(
            df_eval["predicted_frequency"].rank(method="first"),
            q=n_groups,
            labels=False,
        )
        + 1
    )

    table = (
        df_eval.groupby("risk_group", observed=True)
        .agg(
            exposure=("exposure", "sum"),
            observed_claims=("claim_count", "sum"),
            predicted_claims=("predicted_claim_count", "sum"),
            min_predicted_frequency=("predicted_frequency", "min"),
            max_predicted_frequency=("predicted_frequency", "max"),
            mean_predicted_frequency=("predicted_frequency", "mean"),
            n_policies=("claim_count", "size"),
        )
        .reset_index()
    )

    table["observed_frequency"] = table["observed_claims"] / table["exposure"]
    table["predicted_frequency"] = table["predicted_claims"] / table["exposure"]

    table["calibration_ratio_observed_over_predicted"] = (
        table["observed_claims"] / table["predicted_claims"]
    )

    freq_global = global_observed_frequency(y_count, exposure)
    table["lift_observed"] = table["observed_frequency"] / freq_global

    return table


def model_performance_row(
    model_name: str,
    train_deviance: float,
    test_deviance: float,
) -> dict:
    """
    Create one row for a model performance table.

    Parameters
    ----------
    model_name : str
        Name of the model.
    train_deviance : float
        Poisson deviance on the training set.
    test_deviance : float
        Poisson deviance on the test set.

    Returns
    -------
    dict
        Dictionary containing model name, train deviance, test deviance,
        and train-test gap.
    """
    return {
        "model": model_name,
        "train_poisson_deviance": float(train_deviance),
        "test_poisson_deviance": float(test_deviance),
        "test_train_gap": train_test_gap(train_deviance, test_deviance),
    }