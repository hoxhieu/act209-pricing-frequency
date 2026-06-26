"""
Modeling utilities for the ACT209 pricing frequency project.

This module defines reusable functions to build and save models.

Actuarial setup
---------------
For each policy i:

    N_i = observed claim count
    E_i = exposure
    Y_i^{freq} = N_i / E_i

The models are trained on empirical frequency:

    y_i = N_i / E_i

with exposure weights:

    sample_weight_i = E_i

The predicted expected claim count is reconstructed later as:

    mu_hat_i = E_i * lambda_hat_i
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline


def build_poisson_glm_model(
    preprocessor,
    alpha: float = 0.0,
    max_iter: int = 1000,
) -> Pipeline:
    """
    Build a Poisson GLM pipeline.

    Parameters
    ----------
    preprocessor : sklearn transformer
        Preprocessing transformer.
    alpha : float, default=0.0
        L2 regularization strength. The default value alpha=0.0 corresponds
        to an unpenalized Poisson GLM.
    max_iter : int, default=1000
        Maximum number of iterations for numerical optimization.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with preprocessing and PoissonRegressor.

    Notes
    -----
    The model is trained on empirical frequency:

        y_i = ClaimNb_i / Exposure_i

    with exposure weights:

        sample_weight_i = Exposure_i

    This is consistent with evaluating predictions on the count scale:

        mu_hat_i = Exposure_i * lambda_hat_i
    """
    model = PoissonRegressor(
        alpha=alpha,
        max_iter=max_iter,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def build_gradient_boosting_model(
    preprocessor,
    learning_rate: float = 0.05,
    max_iter: int = 100,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 100,
    l2_regularization: float = 0.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a Gradient Boosting pipeline with Poisson loss.

    Parameters
    ----------
    preprocessor : sklearn transformer
        Preprocessing transformer.
    learning_rate : float, default=0.05
        Shrinkage parameter. Smaller values make boosting updates more
        conservative.
    max_iter : int, default=100
        Number of boosting iterations.
    max_leaf_nodes : int, default=31
        Maximum number of leaves for each tree.
    min_samples_leaf : int, default=100
        Minimum number of samples per leaf.
    l2_regularization : float, default=0.0
        L2 regularization parameter for the boosting model.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with preprocessing and HistGradientBoostingRegressor.

    Notes
    -----
    The model is trained on empirical frequency:

        y_i = ClaimNb_i / Exposure_i

    with exposure weights:

        sample_weight_i = Exposure_i

    The model predicts annual claim frequency lambda_hat_i.
    The expected claim count is reconstructed later as:

        mu_hat_i = Exposure_i * lambda_hat_i
    """
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def save_model(model, path: str | Path) -> None:
    """
    Save a trained model to disk using joblib.

    Parameters
    ----------
    model : object
        Trained model or pipeline.
    path : str or pathlib.Path
        Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    """
    Load a trained model from disk.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the saved model.

    Returns
    -------
    object
        Loaded model or pipeline.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)