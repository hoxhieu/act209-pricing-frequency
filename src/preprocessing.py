"""
Preprocessing utilities for the ACT209 pricing frequency project.

This module defines preprocessing pipelines for:

1. Poisson GLM baseline;
2. Gradient Boosting model.

Actuarial context
-----------------
The target is claim frequency:

    Y_i^{freq} = ClaimNb_i / Exposure_i

The exposure is used as sample weight:

    w_i = Exposure_i

Therefore, exposure is not used as an ordinary explanatory feature in the
recommended modeling approach.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_glm_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline for the Poisson GLM.

    Parameters
    ----------
    numeric_features : list[str]
        Names of numeric explanatory variables.
    categorical_features : list[str]
        Names of categorical explanatory variables.

    Returns
    -------
    ColumnTransformer
        Preprocessing object for the GLM.

    Notes
    -----
    For the GLM:

    - numeric variables are imputed with the median and standardized;
    - categorical variables are imputed with the most frequent category
      and one-hot encoded with one reference modality dropped.

    Dropping one categorical modality avoids exact linear dependence with the
    intercept in the unpenalized Poisson GLM. Standardization is useful because
    the GLM is linear in the transformed variables and relies on numerical
    optimization.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_tree_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline for tree-based Gradient Boosting.

    Parameters
    ----------
    numeric_features : list[str]
        Names of numeric explanatory variables.
    categorical_features : list[str]
        Names of categorical explanatory variables.

    Returns
    -------
    ColumnTransformer
        Preprocessing object for tree-based models.

    Notes
    -----
    For tree-based Gradient Boosting:

    - numeric variables are imputed with the median;
    - categorical variables are imputed with the most frequent category
      and one-hot encoded;
    - numeric variables are not standardized.

    Standardization is generally not necessary for tree-based models, because
    split decisions depend on orderings and thresholds, not on Euclidean
    distances.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor