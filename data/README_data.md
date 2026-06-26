# Data README — ACT209 Motor Insurance Frequency Project

## Purpose

This file documents the data used in the ACT209 motor insurance claim frequency pricing project.

The project uses the `freMTPL2freq` dataset to model annual claim frequency in motor insurance.

The main objective is to predict claim frequency while correctly accounting for policy exposure.

---

## Data files

The project uses two main data files:

```text
data/raw/freMTPL2freq.csv
data/processed/frequency_clean.csv
```

### Raw data

```text
data/raw/freMTPL2freq.csv
```

This is the original dataset used in the project. It should not be modified manually.

### Processed data

```text
data/processed/frequency_clean.csv
```

This file is created by `01_data_exploration.ipynb` after applying the basic actuarial data checks and creating the empirical frequency target.

The processed file is used as input for:

```text
02_modeling_glm_gb.ipynb
03_final_results.ipynb
```

---

## Data source and download

The dataset used in this project is `freMTPL2freq`, a motor third-party liability claim frequency dataset.

The original academic source is the R package `CASdatasets`:

```text
https://dutangc.github.io/CASdatasets/reference/freMTPL.html
```

The CSV file used in this project contains 678,013 observations and 12 variables. It was downloaded from the following Hugging Face dataset repository:

```text
https://huggingface.co/datasets/mabilton/fremtpl2/tree/main
```

The identity of the file used in this project was verified by comparing the SHA256 hash of the local project file with a fresh download from the Hugging Face repository.

```text
SHA256: 66477B983CC81B23F75590AD72E00AD4CC4B92A956B22B93732ADB6070F006FB
```

Therefore, the raw CSV file used in this project is exactly the file `freMTPL2freq.csv` downloaded from the Hugging Face repository above.

For reproducibility, the raw CSV file is not redistributed in the GitHub repository. To reproduce the project, download `freMTPL2freq.csv` from the Hugging Face link above, place it in:

```text
data/raw/
```

and then run the notebooks in order.

---

## Dataset dimensions

The raw dataset contains:

```text
Number of rows:     678,013
Number of columns:  12
```

After processing, the dataset contains one additional column:

```text
claim_frequency
```

Therefore, the processed dataset contains:

```text
Number of rows:     678,013
Number of columns:  13
```

No row is removed by the basic cleaning step because the core actuarial validity checks are satisfied in the raw data.

---

## Original columns

The raw dataset contains the following columns:

| Column       | Description               | Role in project                                               |
| ------------ | ------------------------- | ------------------------------------------------------------- |
| `IDpol`      | Policy identifier         | Excluded from modeling                                        |
| `ClaimNb`    | Observed number of claims | Target count                                                  |
| `Exposure`   | Policy exposure           | Used as sample weight and to reconstruct expected claim count |
| `VehPower`   | Vehicle power             | Numerical explanatory variable                                |
| `VehAge`     | Vehicle age               | Numerical explanatory variable                                |
| `DrivAge`    | Driver age                | Numerical explanatory variable                                |
| `BonusMalus` | Bonus-malus coefficient   | Numerical explanatory variable                                |
| `VehBrand`   | Vehicle brand             | Categorical explanatory variable                              |
| `VehGas`     | Vehicle gas type          | Categorical explanatory variable                              |
| `Area`       | Geographic area category  | Categorical explanatory variable                              |
| `Density`    | Population density        | Numerical explanatory variable                                |
| `Region`     | Region category           | Categorical explanatory variable                              |

---

## Target variable

The observed claim count for policy (i) is:

[
N_i = \text{ClaimNb}_i.
]

The exposure for policy (i) is:

[
E_i = \text{Exposure}_i.
]

The empirical annual claim frequency is defined as:

[
Y_i^{freq} = \frac{N_i}{E_i}.
]

In the processed dataset, this target is stored as:

```text
claim_frequency = ClaimNb / Exposure
```

The models are trained to predict annual claim frequency:

[
\widehat{\lambda}_i.
]

The expected number of claims over the observed exposure period is then reconstructed as:

[
\widehat{\mu}_i = E_i \widehat{\lambda}_i.
]

---

## Data quality checks

The following checks are performed in `01_data_exploration.ipynb`:

```text
ClaimNb is not missing.
Exposure is not missing.
Exposure is strictly positive.
ClaimNb is non-negative.
ClaimNb is integer-valued.
```

The observed quality diagnostics are:

| Metric                           |          Value |
| -------------------------------- | -------------: |
| Number of rows                   |        678,013 |
| Number of original columns       |             12 |
| Missing `ClaimNb` values         |              0 |
| Missing `Exposure` values        |              0 |
| Rows with `Exposure <= 0`        |              0 |
| Rows with negative `ClaimNb`     |              0 |
| Total claim count                |         36,102 |
| Total exposure                   | 358,499.445462 |
| Global observed annual frequency |       0.100703 |

The global observed annual frequency is computed as:

[
\frac{\sum_i N_i}{\sum_i E_i}
=============================

\frac{36102}{358499.445462}
\approx 0.1007.
]

This corresponds to approximately:

```text
10.07% annual claim frequency
```

---

## Important warning about frequency averages

The individual empirical frequency is:

[
\frac{N_i}{E_i}.
]

However, the simple average of individual frequencies:

[
\frac{1}{n}\sum_i \frac{N_i}{E_i}
]

is not the same as the portfolio-level annual frequency:

[
\frac{\sum_i N_i}{\sum_i E_i}.
]

For actuarial interpretation, the exposure-weighted portfolio frequency:

[
\frac{\sum_i N_i}{\sum_i E_i}
]

is the more relevant quantity.

This is especially important when some policies have small exposures. Small exposures can create very large individual values of:

```text
claim_frequency
```

even when the observed number of claims is small.

---

## Claim count distribution

The observed claim count distribution is highly concentrated at zero.

The main observed counts are:

| `ClaimNb` | Number of policies |
| --------: | -----------------: |
|         0 |            643,953 |
|         1 |             32,178 |
|         2 |              1,784 |
| 3 or more |               Rare |

Approximately 95% of policies have no observed claim.

This is typical in motor insurance frequency modeling, where claim counts are sparse and highly asymmetric.

---

## Feature columns used for modeling

The explanatory variables used in the modeling notebooks are:

```text
VehPower
VehAge
DrivAge
BonusMalus
VehBrand
VehGas
Area
Density
Region
```

The numerical variables are:

```text
VehPower
VehAge
DrivAge
BonusMalus
Density
```

The categorical variables are:

```text
VehBrand
VehGas
Area
Region
```

The following variables are excluded from the explanatory feature set:

```text
IDpol
ClaimNb
Exposure
claim_frequency
```

`Exposure` is not used as a standard explanatory variable. It is used as:

```text
1. sample weight during model fitting;
2. exposure multiplier to reconstruct expected claim counts.
```

---

## Cleaning procedure

The processed dataset is created from the raw dataset using the following basic actuarial cleaning rules:

```text
1. Remove rows with missing Exposure.
2. Remove rows with missing ClaimNb.
3. Keep only rows with Exposure > 0.
4. Keep only rows with ClaimNb >= 0.
5. Check that ClaimNb is integer-valued.
6. Create claim_frequency = ClaimNb / Exposure.
```

In the current dataset, these checks do not remove observations because the raw data already satisfies the core validity constraints.

The cleaning step does not perform:

```text
- imputation of explanatory variables;
- scaling of numerical variables;
- encoding of categorical variables;
- train / validation / test split.
```

These operations are performed later inside scikit-learn preprocessing pipelines in the modeling notebook.

This avoids data leakage from validation or test data into the training preprocessing steps.

---

## Processed data schema

The processed dataset contains the original 12 columns plus the target frequency column.

| Column            | Type of role               |
| ----------------- | -------------------------- |
| `IDpol`           | Identifier                 |
| `ClaimNb`         | Observed claim count       |
| `Exposure`        | Exposure                   |
| `VehPower`        | Numerical feature          |
| `VehAge`          | Numerical feature          |
| `DrivAge`         | Numerical feature          |
| `BonusMalus`      | Numerical feature          |
| `VehBrand`        | Categorical feature        |
| `VehGas`          | Categorical feature        |
| `Area`            | Categorical feature        |
| `Density`         | Numerical feature          |
| `Region`          | Categorical feature        |
| `claim_frequency` | Empirical frequency target |

---

## Files generated from the data

The data exploration notebook generates the processed dataset:

```text
data/processed/frequency_clean.csv
```

It also generates exploratory figures in:

```text
outputs/figures/
```

The main exploratory figures are:

```text
eda_claim_count_distribution.png
eda_exposure_distribution.png
eda_frequency_by_bonusmalus_decile.png
eda_frequency_by_area.png
eda_frequency_by_vehgas.png
```

These figures are used to understand the structure of the insurance portfolio before modeling.

---

## Reproducibility

To reproduce the processed dataset, run:

```text
01_data_exploration.ipynb
```

from a clean Jupyter kernel using:

```text
Kernel -> Restart Kernel and Run All Cells
```

The output file should be:

```text
data/processed/frequency_clean.csv
```

The processed file should contain:

```text
678,013 rows
13 columns
```

and should include the column:

```text
claim_frequency
```

---

## Data limitations

The following limitations should be kept in mind:

1. The project models claim frequency only, not claim severity.
2. The project does not model pure premium directly.
3. The individual frequency `ClaimNb / Exposure` can be very large for small exposures.
4. The dataset is treated as a tabular supervised learning dataset.
5. The analysis assumes that the available explanatory variables are sufficient for the comparison of the selected models.
6. No causal interpretation should be made from the fitted models.
7. The processed dataset is intended for educational and modeling purposes within the ACT209 project.

---

## Final data statement

The data preparation step produces a clean actuarial frequency dataset with valid claim counts, valid exposures, and an explicit empirical frequency target.

The processed dataset is suitable for comparing a Poisson GLM and a Gradient Boosting model for motor insurance claim frequency prediction.
