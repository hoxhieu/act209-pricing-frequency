# ACT209 — Motor Insurance Claim Frequency Pricing

## Project overview

This project studies motor insurance claim frequency pricing using the `freMTPL2freq` dataset.

The objective is to compare a classical actuarial model and a machine learning model for predicting annual claim frequency:

1. a constant frequency benchmark;
2. an unpenalized Poisson GLM;
3. a Gradient Boosting model with Poisson loss.

The models are trained to predict annual claim frequency. For each policy (i), the observed claim count is denoted by:

[
N_i,
]

the exposure by:

[
E_i,
]

and the empirical claim frequency by:

[
Y_i^{freq} = \frac{N_i}{E_i}.
]

The models predict an annual frequency:

[
\widehat{\lambda}_i.
]

The expected claim count over the exposure period is then reconstructed as:

[
\widehat{\mu}_i = E_i \widehat{\lambda}_i.
]

Model performance is evaluated on the claim count scale using Poisson deviance.

---

## Repository structure

```text
act209_pricing_frequency/
│
├── data/
│   ├── raw/
│   │   └── freMTPL2freq.csv
│   ├── processed/
│   │   └── frequency_clean.csv
│   └── README_data.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_modeling_glm_gb.ipynb
│   └── 03_final_results.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── modeling.py
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
│
├── report/
├── slides/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The raw dataset contains:

```text
678,013 observations
12 original columns
```

The original columns are:

```text
IDpol
ClaimNb
Exposure
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

The main actuarial target variables are:

```text
ClaimNb   : observed claim count
Exposure  : exposure duration
```

The empirical frequency target is created as:

[
\text{claim_frequency}_i =
\frac{\text{ClaimNb}_i}{\text{Exposure}_i}.
]

After cleaning, the processed dataset contains one additional column:

```text
claim_frequency
```

---

## Data quality checks

The following core actuarial checks were performed:

```text
Number of rows:                         678,013
Number of original columns:             12
Missing ClaimNb values:                 0
Missing Exposure values:                0
Exposure <= 0:                          0
Negative claim counts:                  0
Total claim count:                      36,102
Total exposure:                         358,499.445462
Global observed annual frequency:       0.100703
```

The global observed annual claim frequency is:

[
\frac{\sum_i N_i}{\sum_i E_i}
=============================

\frac{36102}{358499.445462}
\approx 0.1007.
]

This corresponds to an annual claim frequency of approximately:

```text
10.07%
```

Most policies have no claim. The observed claim count distribution is highly asymmetric, which is typical in motor insurance frequency modeling.

---

## Explanatory variables

The selected explanatory variables are:

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

`Exposure` is not used as an ordinary explanatory variable. It is used as sample weight during model fitting and to reconstruct expected claim counts.

---

## Methodology

### Train / validation / test split

The dataset is split into:

```text
60% training
20% validation
20% test
```

The split summary is:

| Set        | Number of policies | Total exposure | Observed claims | Observed frequency |
| ---------- | -----------------: | -------------: | --------------: | -----------------: |
| Train      |            406,807 |     215,069.76 |          21,571 |           0.100298 |
| Validation |            135,603 |      71,805.26 |           7,289 |           0.101525 |
| Test       |            135,603 |      71,624.42 |           7,242 |           0.101097 |

The observed frequencies are close across the three samples, which indicates that the split is reasonably balanced.

---

## Models

### 1. Constant frequency benchmark

The constant benchmark predicts the same annual frequency for all policies:

[
\widehat{\lambda}*0 =
\frac{\sum*{i\in train} N_i}
{\sum_{i\in train} E_i}.
]

The estimated training frequency is:

```text
lambda_constant_train = 0.10029768996830889
```

This benchmark provides a minimum reference level for model comparison.

---

### 2. Poisson GLM

The Poisson GLM is the classical actuarial benchmark.

It is fitted as an unpenalized Poisson regression model:

```text
alpha = 0.0
```

The model uses:

```text
- median imputation and standardization for numerical variables;
- most-frequent imputation and one-hot encoding for categorical variables;
- one dropped reference modality for categorical variables;
- exposure as sample weight.
```

The model has the form:

[
\log(\lambda_i) = \beta_0 + X_i^\top \beta.
]

For a numerical variable (X_j), the multiplicative effect is:

[
\exp(\beta_j).
]

Because numerical variables are standardized, numerical coefficients are interpreted for a one-standard-deviation increase in the corresponding variable, all else being equal.

---

### 3. Gradient Boosting with Poisson loss

The Gradient Boosting model is used as the flexible machine learning model.

It is fitted with Poisson loss and the following parameters:

```text
learning_rate = 0.05
max_iter = 100
max_leaf_nodes = 31
min_samples_leaf = 100
l2_regularization = 0.0
random_state = 42
```

The model is also trained on empirical claim frequency using exposure as sample weight.

Compared with the Poisson GLM, Gradient Boosting can capture nonlinear effects and interactions. However, it is less directly interpretable and may produce more extreme predictions.

---

## Model performance

The models are compared using mean Poisson deviance. Lower values indicate better predictive performance.

| Model              | Train deviance | Validation deviance | Test deviance | Test-train gap |
| ------------------ | -------------: | ------------------: | ------------: | -------------: |
| Constant benchmark |       0.330765 |            0.331719 |      0.331425 |       0.000660 |
| Poisson GLM        |       0.320103 |            0.320796 |      0.321590 |       0.001487 |
| Gradient Boosting  |       0.298048 |            0.303698 |      0.305728 |       0.007679 |

The ranking by test Poisson deviance is:

```text
Gradient Boosting < Poisson GLM < Constant benchmark
```

Since Poisson deviance is a loss, this means that Gradient Boosting has the best global predictive performance among the tested models.

The Gradient Boosting model improves the test deviance relative to the Poisson GLM by approximately:

[
\frac{0.321590 - 0.305728}{0.321590}
\approx 4.93%.
]

It improves the test deviance relative to the constant benchmark by approximately:

[
\frac{0.331425 - 0.305728}{0.331425}
\approx 7.75%.
]

However, the Gradient Boosting model has a larger train-test gap than the GLM, indicating a higher degree of model flexibility and a greater need for monitoring.

---

## GLM coefficient interpretation

The numerical GLM coefficients are:

| Feature    | Coefficient | Multiplicative effect | Percentage effect |
| ---------- | ----------: | --------------------: | ----------------: |
| BonusMalus |    0.348296 |              1.416652 |           +41.67% |
| VehAge     |   -0.222863 |              0.800224 |           -19.98% |
| DrivAge    |    0.092979 |              1.097439 |            +9.74% |
| VehPower   |    0.027806 |              1.028197 |            +2.82% |
| Density    |    0.016487 |              1.016623 |            +1.66% |

Interpretation:

* A one-standard-deviation increase in `BonusMalus` is associated with an approximately 41.7% higher predicted annual claim frequency, all else being equal.
* A one-standard-deviation increase in `VehAge` is associated with an approximately 20.0% lower predicted annual claim frequency, all else being equal.
* `DrivAge`, `VehPower`, and `Density` have smaller positive associations in the GLM.

These are model-based associations, not causal effects.

---

## Gradient Boosting feature importance

Permutation feature importance was computed on the validation set.

The most important variables for the Gradient Boosting model are:

| Feature    | Importance mean |
| ---------- | --------------: |
| VehAge     |        0.022781 |
| BonusMalus |        0.022542 |
| VehBrand   |        0.010009 |
| DrivAge    |        0.004970 |
| VehGas     |        0.004031 |
| VehPower   |        0.003590 |
| Density    |        0.001305 |
| Region     |        0.000674 |
| Area       |        0.000047 |

Interpretation:

```text
VehAge and BonusMalus are the two most important variables for the Gradient Boosting model.
```

This importance is predictive, not causal. It measures how much the validation Poisson deviance deteriorates when a variable is randomly permuted.

---

## Calibration analysis

Calibration is assessed by predicted-risk decile.

For a group (g), observed and predicted claim counts are compared through:

[
\sum_{i\in g} N_i
]

and

[
\sum_{i\in g} \widehat{\mu}_i.
]

The calibration ratio is:

[
\frac{\sum_{i\in g} N_i}
{\sum_{i\in g} \widehat{\mu}_i}.
]

A ratio close to 1 indicates good calibration.

### Calibration summary

| Model             | Mean absolute calibration error | Maximum absolute calibration error | Top decile observed frequency | Top decile predicted frequency | Top decile lift |
| ----------------- | ------------------------------: | ---------------------------------: | ----------------------------: | -----------------------------: | --------------: |
| Poisson GLM       |                        0.005344 |                           0.012756 |                      0.222566 |                       0.233303 |        2.201511 |
| Gradient Boosting |                        0.004250 |                           0.017106 |                      0.355134 |                       0.348393 |        3.512816 |

Interpretation:

* The Gradient Boosting model has a slightly lower average calibration error.
* The Gradient Boosting model has a larger maximum local calibration error.
* The Poisson GLM is slightly more regular in local calibration.
* The Gradient Boosting model produces a much stronger top-decile lift.

Therefore, Gradient Boosting is better on average and much stronger for risk segmentation, but its local calibration should be monitored.

---

## Lift analysis

Lift measures the ranking ability of the model.

For a predicted-risk group (g), the observed lift is:

[
\text{Lift}*g =
\frac{
\frac{\sum*{i\in g} N_i}{\sum_{i\in g} E_i}
}{
\frac{\sum_i N_i}{\sum_i E_i}
}.
]

The top-decile lift is:

```text
Poisson GLM:        2.201511
Gradient Boosting:  3.512816
```

This means that the top predicted-risk decile selected by Gradient Boosting has an observed claim frequency approximately 3.51 times the test portfolio average.

The top Gradient Boosting decile contains:

```text
n_policies = 13,561
exposure = 4,913.631512
observed_claims = 1,745
predicted_claims = 1,711.872873
observed_frequency = 0.355134
predicted_frequency = 0.348393
calibration_ratio = 1.019351
lift = 3.512816
```

This is the strongest evidence that Gradient Boosting provides better risk segmentation.

---

## Prediction stability and extreme predictions

The distribution of predicted annual frequencies is:

| Model             | Mean predicted frequency | Standard deviation | 99% quantile | 99.9% quantile |  Maximum |
| ----------------- | -----------------------: | -----------------: | -----------: | -------------: | -------: |
| Poisson GLM       |                 0.107803 |           0.056744 |     0.323613 |       0.529880 | 3.885052 |
| Gradient Boosting |                 0.114047 |           0.122715 |     0.636624 |       1.198751 | 9.760177 |

Gradient Boosting produces a more dispersed distribution of predicted frequencies.

The standard deviation of Gradient Boosting predictions is approximately:

[
\frac{0.122715}{0.056744}
\approx 2.16
]

times the standard deviation of GLM predictions.

The 99.9% quantile of Gradient Boosting predictions is approximately:

[
\frac{1.198751}{0.529880}
\approx 2.26
]

times the 99.9% quantile of GLM predictions.

The maximum Gradient Boosting predicted frequency is approximately:

[
\frac{9.760177}{3.885052}
\approx 2.51
]

times the maximum GLM predicted frequency.

This confirms that Gradient Boosting is more predictive but also less stable in the upper tail.

Extreme predictions are not necessarily wrong. However, they should be monitored before operational use, especially when they occur on policies with small exposures or rare combinations of risk factors.

---

## Main conclusion

The final evaluation leads to the following conclusion:

```text
Gradient Boosting is the best predictive model in this study.
Poisson GLM remains the main actuarial benchmark.
```

Gradient Boosting has:

```text
- the lowest test Poisson deviance;
- the strongest top-decile lift;
- better risk segmentation;
- stronger ability to identify high-risk policies.
```

Poisson GLM remains useful because it is:

```text
- simpler;
- more stable;
- more directly interpretable;
- easier to justify as an actuarial benchmark.
```

The final actuarial conclusion is therefore:

```text
Gradient Boosting should be presented as the best predictive model, while the Poisson GLM should be retained as a transparent and stable actuarial benchmark.
```

Before any operational use, the Gradient Boosting model would require additional monitoring of calibration, prediction stability and extreme predicted frequencies.

---

## Generated outputs

The project generates the following main output tables:

```text
outputs/tables/split_summary.csv
outputs/tables/model_performance.csv
outputs/tables/test_predictions.csv
outputs/tables/glm_numeric_coefficients.csv
outputs/tables/gb_feature_importance.csv
outputs/tables/calibration_table_glm.csv
outputs/tables/calibration_table_gb.csv
outputs/tables/calibration_summary.csv
outputs/tables/prediction_distribution_summary.csv
outputs/tables/prediction_stability.csv
```

The project generates the following main figures:

```text
outputs/figures/eda_claim_count_distribution.png
outputs/figures/eda_exposure_distribution.png
outputs/figures/eda_frequency_by_bonusmalus_decile.png
outputs/figures/eda_frequency_by_area.png
outputs/figures/eda_frequency_by_vehgas.png
outputs/figures/glm_numeric_coefficients.png
outputs/figures/gb_feature_importance.png
outputs/figures/calibration_deciles.png
outputs/figures/lift_chart.png
outputs/figures/glm_observed_vs_predicted_frequency.png
outputs/figures/gb_observed_vs_predicted_frequency.png
outputs/figures/prediction_stability.png
```

The fitted models are saved as:

```text
outputs/models/glm_poisson.pkl
outputs/models/gb_poisson.pkl
```

---

## How to run the project

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv-act209-frequency
.\.venv-act209-frequency\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the notebooks in order

Run the notebooks in the following order:

```text
01_data_exploration.ipynb
02_modeling_glm_gb.ipynb
03_final_results.ipynb
```

Each notebook should be run with:

```text
Kernel -> Restart Kernel and Run All Cells
```

The correct execution order is important because:

```text
01_data_exploration.ipynb creates data/processed/frequency_clean.csv
02_modeling_glm_gb.ipynb creates predictions, model outputs and fitted models
03_final_results.ipynb creates calibration, lift and stability analyses
```

---

## Reproducibility

The project uses:

```text
random_state = 42
```

for train / validation / test splitting and model reproducibility where applicable.

The final notebooks should be executed sequentially from a clean kernel. The output verification cells in the notebooks should return `True` for all expected output files.

---

## Limitations

The project has the following limitations:

1. The Gradient Boosting hyperparameters were not selected through an exhaustive grid search.
2. The analysis focuses on claim frequency only, not claim severity or pure premium.
3. Extreme Gradient Boosting predictions require additional actuarial monitoring before operational use.
4. The interpretation of Gradient Boosting relies on permutation feature importance, which measures predictive contribution, not causal effect.
5. The GLM is more interpretable but less flexible than Gradient Boosting.
6. The results are based on the available dataset and the selected train / validation / test split.

---

## Final project statement

This project shows that a flexible machine learning model can improve predictive performance and risk segmentation compared with a classical Poisson GLM in motor insurance frequency modeling.

However, the Poisson GLM remains an essential actuarial benchmark because of its transparency, stability and interpretability.

The final recommendation is to present Gradient Boosting as the best predictive model in this study, while using the Poisson GLM as the reference actuarial benchmark.
