# Routex: Machine-Learning-Based Vehicle Fuel Consumption Prediction

## Progress Evaluation Report

---

## 1. Project Title and Group Information

| Item | Details |
|---|---|
| **Project title** | Routex: Machine-Learning-Based Vehicle Fuel Consumption Prediction |
| **Course / module** | *[to be completed by the group]* |
| **University / department** | *[to be completed by the group]* |
| **Supervisor** | *[to be completed by the group]* |
| **Report date** | 27 November 2026 |
| **Code repository** | https://github.com/harshaprasad599651-prog/ROUTEX |

**Group members and responsibilities**

| Member | Name | Responsibilities | Status |
|---|---|---|---|
| Member 1 | Harsha Prasad *(student ID: [to be completed])* | Repository and Python environment setup; dataset inspection (Notebook 01) | Completed |
| Member 2 | Rajitha *(full name and student ID: [to be completed])* | Data preprocessing; exploratory data analysis; training of the first three regression models; model evaluation; progress evaluation documentation | Completed (this report) |
| *[Other members, if any]* | *[to be completed]* | *[to be completed]* | |

---

## 2. Project Overview

Routex is a supervised machine-learning project that predicts the **combined fuel economy
of a vehicle** (miles per gallon, MPG) from its technical specifications, such as engine
size, number of cylinders, transmission, drive type, vehicle class and powertrain type.

The work is organised as a reproducible Python pipeline. There are five Jupyter notebooks
(inspection, preprocessing, EDA, training, evaluation) and reusable code in `src/`. All
work is version-controlled with Git. Each stage builds on the previous one and saves its
outputs (cleaned data splits, trained models, metrics and figures), so every result in
this report can be reproduced by re-running the notebooks.

**Technology:** Python 3.14.5, pandas 3.1.1, NumPy 2.5.5, scikit-learn 1.9.1, matplotlib,
seaborn and Jupyter. The dependencies are listed in `requirements.txt`.

---

## 3. Problem Statement

A vehicle's fuel economy is normally known only after standardised laboratory testing.
For planning, comparison or early design decisions, it is useful to **estimate fuel
economy from specifications alone**, before or without a test result.

The relationship between specifications and fuel economy is not simple. It is
non-linear (e.g. the effect of engine size levels off for large engines), it involves
interactions (e.g. a hybrid powertrain changes the effect of engine size), and it has
changed over time with new technology. The project asks:

> *How accurately can machine-learning regression models predict a vehicle's combined
> fuel economy (`comb08`) from its specifications, and which model type is most suitable?*

A key requirement is that predictions must **not** use information that is itself a
result of the fuel-economy test (e.g. city/highway MPG, fuel cost or CO₂ emissions).
Using such information would make the problem trivial and the model useless in practice.

---

## 4. Project Objective

**Main objective:** build and compare regression models that predict `comb08` from
leakage-free vehicle specifications.

**Specific objectives:**

1. Inspect and understand the dataset, including data quality issues. *(completed)*
2. Design a leakage-free, reproducible preprocessing pipeline. *(completed)*
3. Explore the data to understand the drivers of fuel economy. *(completed)*
4. Train and tune Linear Regression, Decision Tree and Random Forest models. *(completed)*
5. Evaluate and compare them with MAE, MSE, RMSE and R². *(completed)*
6. Implement Gradient Boosting and XGBoost and compare them with the current best
   model. *(planned: later project stage)*

---

## 5. Dataset Description and Source

| Item | Details |
|---|---|
| **Original data source** | U.S. fuel-economy data published at fueleconomy.gov (U.S. Department of Energy / EPA) |
| **Copy used** | `vehicles.csv` from the public GitHub repository *ajhenry/searchable-car-database* (raw file, `master` branch) |
| **Size** | 42,774 vehicle records × 83 columns (about 17.1 MB) |
| **Model years** | 1984 to 2021 |
| **Unit of observation** | one vehicle configuration (make, model, year, engine, transmission) |
| **Target variable** | `comb08`, combined city/highway fuel economy, in MPG (MPGe for electric vehicles) |
| **Column types** | 32 float, 27 integer, 23 text, 1 boolean |

The dataset is not stored in the Git repository (it is listed in `.gitignore`). Each
member downloads it to `data/raw/vehicles.csv`. The source repository describes its
coverage as starting in 1987, but the file itself contains records from 1984.

**Main feature groups used for modelling:** model year; engine displacement and
cylinders; transmission (type, number of gears, CVT, descriptors); turbo/supercharger;
drive type; vehicle class; manufacturer; fuel type(s); powertrain type (conventional,
hybrid, plug-in hybrid, diesel, flex-fuel, CNG, etc.); start-stop system; interior
passenger and luggage volume.

---

## 6. Data Inspection Findings *(Notebook 01, completed by Member 1)*

The inspection only examined the data; nothing was changed.

| Finding | Result |
|---|---|
| Dimensions | 42,774 rows, 83 columns |
| Missing values | 17 of 83 columns contain missing values (473,613 missing cells in total). Several are almost entirely empty: `c240bDscr` 99.8%, `sCharger` 97.9%, `evMotor` 97.5%, `guzzler` 94.1%, `atvType` 91.0%. `drive` is missing in 2.8% of rows, `displ`/`cylinders` in 0.6%. |
| Duplicates | 0 exact duplicate rows; `id` is unique |
| Constant columns | 1 (`charge120`) |
| Target `comb08` | no missing values; range 7 to 141; mean 20.80, median 20.0; no zero or negative values |
| Target outliers (IQR rule) | 5 low, 1,309 high |
| Electric vehicles | 437 records (1.02%) have a fuel type that mentions electricity. They account for 308 of the 1,309 high outliers. |
| Target leakage | columns derived from the same test correlate very strongly with `comb08`: `city08` 0.991, `UCity` 0.986, `highway08` 0.977, `UHighway` 0.972; `co2TailpipeGpm` −0.785, `barrels08` −0.773, `fuelCost08` −0.667, `youSaveSpend` 0.667 |
| Placeholders | several columns use `-1` or `0` to mean "not available" |

These findings defined the main questions for preprocessing: how to treat electric
vehicles, which columns leak the target, and how to handle missing and placeholder values.

---

## 7. Data Preprocessing Methods *(Notebook 02, completed)*

The reusable code is in `src/preprocessing.py`. All rule-based cleaning is deterministic
(it learns nothing from the data). All steps that learn from data are placed in a
scikit-learn `ColumnTransformer` and are **fitted on the training set only**.

### 7.1 Row-level decisions

| Step | Rows affected | Reasoning |
|---|---|---|
| **Excluded pure electric vehicles** (`fuelType1 == "Electricity"`) | 241 removed | For EVs, `comb08` is MPGe (median 103), a different unit from MPG (median 20). EVs also have no engine data, and they would dominate the error metrics. Hybrids (729) and plug-in hybrids (196) were **kept**, because their `comb08` is petrol MPG (plug-in hybrids: petrol-only rating, 17 to 54 MPG). |
| **Removed duplicate records** (identical except `id`) | 40 removed | Duplicates could end up in both the training and the test set. |
| **Outliers in the target** | kept | They are genuine vehicles (hybrids, diesels, microcars, supercars), not errors. |

Result: **42,493 vehicles**.

### 7.2 Column-level decisions (all 83 columns accounted for)

| Decision | Columns | Reason |
|---|---|---|
| Dropped: required leakage list | 8 (`city08`, `highway08`, `UCity`, `UHighway`, `fuelCost08`, `barrels08`, `co2TailpipeGpm`, `youSaveSpend`) | another form of the target measurement |
| Dropped: additional leakage found | 38 | unrounded MPG (`comb08U` is the target before rounding), CO₂ and EPA scores, the gas-guzzler flag (assigned *because* MPG is low), second-fuel and electric test results, driving range |
| Dropped: identifiers / metadata | 6 (`id`, `engId`, `model`, `createdOn`, `modifiedOn`, `mpgData`) | no predictive meaning, or a risk of memorising records |
| Dropped: constant / sparse / redundant | 7 | e.g. `charge120` (constant), EV charging fields (almost empty after removing EVs), `mfrCode` (duplicates `make`) |
| Replaced by engineered features | 8 | see 7.3 |
| **Kept as model inputs** | **15 raw columns → 22 features** | |

The `GUZZLER` tag was also removed from the engine-description text (`eng_dscr`), so the
leaking information could not re-enter through the text.

### 7.3 Feature engineering and placeholder handling

- **Transmission** (`trany`) → `trans_type` (automatic/manual), `trans_speeds` (number of
  gears), `is_cvt`. `trany` itself is also kept.
- **Interior volume:** three column pairs (2-door, 4-door, hatchback) were combined into
  `passenger_volume` and `luggage_volume`; `0` ("not reported") was converted to missing.
- **Turbo / supercharger:** text flags converted to 0/1.
- **Meaningful blanks labelled explicitly:** e.g. empty `atvType` → `"Conventional"`,
  empty `fuelType2` → `"Single fuel"`, empty `startStop` → `"Not reported"`.

### 7.4 Final feature set and pipeline

| Type | Features | Transformation (fitted on training data) |
|---|---|---|
| Numerical (6) | `year`, `displ`, `cylinders`, `trans_speeds`, `passenger_volume`, `luggage_volume` | median imputation + missing-value indicator; standard scaling |
| Binary (4) | `turbo`, `supercharger`, `is_cvt`, `phevBlended` | passed through as 0/1 |
| Categorical (12) | `make`, `VClass`, `drive`, `trany`, `trans_type`, `trans_dscr`, `fuelType`, `fuelType1`, `fuelType2`, `atvType`, `startStop`, `eng_dscr` | `"Missing"` imputation; one-hot encoding; categories seen fewer than 20 times are grouped; unseen categories are handled safely |

- **Train/test split:** 80% / 20%, `random_state = 42`, giving **33,994 training** and
  **8,499 test** vehicles. The same split is used in all later notebooks.
- After encoding: **267 model input columns**, with no missing values.
- **Leakage checks** are built into the notebook: assertions confirm that no leakage
  column or `id` is a feature, that the scaler saw only the training rows, and that the
  imputation medians equal the training medians.

---

## 8. Exploratory Data Analysis Findings *(Notebook 03, completed)*

To avoid leakage, the EDA was performed on the **training split only**. 18 figures
are saved in `reports/figures/03_*.png`.

1. **Engine size is the dominant driver.** Rank (Spearman) correlation with `comb08`:
   `displ` −0.82, `cylinders` −0.78. Linear (Pearson) correlations are weaker
   (−0.73, −0.69), which shows the relationship is **curved**. MPG falls steeply from about
   30 MPG (≤1.5 L) to about 20 MPG (3 L), then levels off.
2. **Powertrain matters strongly for a small group.** Median MPG: hybrids 29, plug-in
   hybrids 28, diesel 23, conventional 20, flex-fuel 18. Hybrids and plug-in hybrids
   account for almost all vehicles that are unusually efficient for their engine size.
3. **Vehicle class and drive type:** vans, pickups and 4WD SUVs have medians of about
   15 to 17 MPG; small cars about 22 to 24. Front-wheel drive has the highest median
   (about 23).
4. **Technology trend:** median MPG was flat at about 18 to 19 until about 2008, then
   rose to about 23. Over the same period, average gears rose from about 4 to about 7.7
   and turbocharging from under 10% to over 50%.
5. **Multicollinearity:** `displ`–`cylinders` r = 0.91; `year`–`trans_speeds` r = 0.75.
6. **Target distribution:** right-skewed (skew 1.15); `log(comb08)` is almost symmetric
   (skew 0.09).
7. **Missing values are informative:** passenger volume is missing mainly for pickups,
   SUVs, vans and two-seaters, which have lower MPG (mean 17.9 vs 22.5).
8. **Data-quality issues:** drive-type labels changed around 2009 ("4-Wheel or
   All-Wheel Drive" until 2009, "All-Wheel Drive" from 2009). Six records from 1984 and
   1991 have impossible passenger volumes (1 to 6 cubic feet).

**How the EDA shaped modelling:** tree-based models were expected to beat the linear
model (curved relationships, interactions); a log target was tested for Linear
Regression; scaling and rare-category grouping were kept; tree complexity was tuned;
and errors were planned to be reported per powertrain group.

---

## 9. The Three Regression Models Implemented *(Notebook 04, completed)*

The reusable code is in `src/modeling.py`. Every model is a single scikit-learn
`Pipeline` (preprocessing → model), so the preprocessing is re-fitted inside every
cross-validation fold and never sees validation or test data. Hyperparameters were
selected by **5-fold cross-validation on the training set** (`GridSearchCV`, shuffled
folds, `random_state = 42`, lowest RMSE). **The test set was used only once**, for the
final evaluation.

| Model | Purpose | Settings searched | Selected settings |
|---|---|---|---|
| **Linear Regression** | interpretable baseline | 2 options: raw target vs `log(comb08)` (via `TransformedTargetRegressor`) | `log(comb08)` target (CV RMSE 1.83 vs 1.98) |
| **Decision Tree Regressor** | captures non-linear effects and interactions; rule-based | 30 combinations: `max_depth` ∈ {6, 10, 14, 18, 24, None} × `min_samples_leaf` ∈ {1, 3, 5, 10, 20} | `max_depth = 24`, `min_samples_leaf = 1` |
| **Random Forest Regressor** | ensemble of 300 trees on bootstrap samples; reduces the variance of a single tree | 9 combinations: `max_features` ∈ {0.33, 0.6, 1.0} × `min_samples_leaf` ∈ {1, 2, 4} | `max_features = 0.6`, `min_samples_leaf = 1`, `n_estimators = 300` |

As a reference, a **baseline** that always predicts the training mean (20.38 MPG) was
also scored. The trained pipelines are saved in `models/` (excluded from Git because of
size; the Random Forest is about 76 MB). The metrics, CV results and test predictions
are saved in `results/`.

---

## 10. Evaluation Metrics and Actual Results *(Notebook 05, completed)*

The saved models were loaded and evaluated on the **same 8,499 test vehicles**. No model
was refitted on test data. The metrics were also recalculated independently from the
saved predictions as a check.

### 10.1 Test-set results

| Model | MAE (MPG) | MSE (MPG²) | RMSE (MPG) | R² | Worst single error (MPG) |
|---|---|---|---|---|---|
| Linear Regression | 1.283 | 3.325 | 1.823 | 0.885 | 18.9 |
| Decision Tree | **0.473** | 0.882 | 0.939 | 0.969 | 15.0 |
| **Random Forest** | 0.476 | **0.566** | **0.752** | **0.980** | **7.5** |
| Baseline (training mean) | 4.030 | 28.903 | 5.376 | −0.001 | 37.6 |

**Training vs test RMSE (overfitting check):** Linear Regression 1.807 → 1.823; Decision
Tree 0.370 → 0.939 (test 2.5 times train); Random Forest 0.361 → 0.752 (2.1 times).

### 10.2 Reliability of the comparison

- **Bootstrap (1,000 resamples of the test set):** Random Forest RMSE 95% CI
  [0.729, 0.776]; Decision Tree [0.896, 0.986]; Linear Regression [1.759, 1.884]. Random
  Forest had a lower RMSE than the Decision Tree in **100%** of resamples (mean
  difference 0.187 MPG, 95% CI [0.155, 0.224]).
- **MAE:** the Decision Tree and Random Forest are effectively tied (0.473 vs 0.476).
  The forest's advantage is that it makes far fewer large errors.

### 10.3 Detailed findings

- **Residuals:** all models are unbiased overall (mean residual within ±0.03 MPG). Linear
  Regression systematically **under-predicts efficient vehicles** (mean residual +2.6 MPG
  for 30 to 40 MPG and +6.7 MPG above 40 MPG). Random Forest's bias is below 0.6 MPG in
  every band; 88% of its predictions are within 1 MPG.
- **By powertrain (test RMSE):** hybrids are the hardest group for every model
  (Linear Regression 4.46, Decision Tree 2.03, Random Forest 1.34). Random Forest has the
  lowest RMSE for conventional, flex-fuel, diesel and hybrid vehicles, and in all 15
  largest vehicle classes. For plug-in hybrids (43 test vehicles), the Decision Tree
  scored better (0.96 vs 1.37), but this group is small.
- **Feature importance (permutation importance on the test set):** `displ` is the most
  important feature for all models (shuffling it increases the Random Forest's RMSE by
  3.57 MPG), followed by `atvType`, `year`, `cylinders` and `drive`. The single tree
  concentrates on `displ` (64% of its impurity importance), while the forest spreads
  importance across correlated features.
- **Robustness: unseen vehicle models.** A grouped 5-fold cross-validation on the
  training data (all records of a make + model in the same fold) gave RMSE values of
  **1.30 (Random Forest)**, 1.72 (Decision Tree) and 1.90 (Linear Regression), compared
  with 0.81, 1.05 and 1.83 for random folds. Random Forest remains the best model; the
  Decision Tree is the least robust.

### 10.4 Current best model

**Random Forest Regressor:** best RMSE, MSE and R²; fewest large errors; best on the
difficult groups; and best under the grouped robustness check.
**Ranking:** 1. Random Forest, 2. Decision Tree, 3. Linear Regression.

---

## 11. Current Progress

### Completed

| Stage | Output | Status |
|---|---|---|
| Repository, environment and folder structure | Git repository, `requirements.txt`, `.gitignore` | ✅ Completed (Member 1) |
| Dataset inspection | `notebooks/01_dataset_inspection.ipynb` | ✅ Completed (Member 1) |
| Data preprocessing | `notebooks/02_preprocessing.ipynb`, `src/preprocessing.py` | ✅ Completed (Member 2), merged into `main` (pull request #1) |
| Exploratory data analysis | `notebooks/03_exploratory_data_analysis.ipynb`, 18 figures | ✅ Completed (Member 2) |
| Training of three regression models | `notebooks/04_model_training.ipynb`, `src/modeling.py`, `results/` | ✅ Completed (Member 2) |
| Model evaluation | `notebooks/05_model_evaluation.ipynb`, 10 result files, 6 figures | ✅ Completed (Member 2) |
| Progress evaluation report | `reports/progress_evaluation.md` | ✅ This document |

The EDA, training and evaluation work is committed on the branch `member-2-ml-pipeline`
and pushed to GitHub. It has **not yet been merged into `main`**.

### Planned (not yet started)

- Gradient Boosting and XGBoost models (Section 15)
- The remaining items listed in Section 14

---

## 12. Challenges Encountered

| Challenge | How it was handled |
|---|---|
| **Python environment.** The first virtual environment was built on an MSYS2 Python, for which numpy/pandas/scikit-learn have no installable packages. | Recreated the environment with the standard Windows Python 3.14 and verified all libraries and the Jupyter kernel. |
| **Mixed units in the target.** EV `comb08` values are MPGe, not MPG. | Investigated EVs, plug-in hybrids and hybrids separately; excluded only pure EVs, with documented reasoning. |
| **Hidden target leakage** beyond the 8 known columns (e.g. `comb08U`, CO₂, EPA scores, gas-guzzler flag and its tag inside `eng_dscr`). | Every one of the 83 columns was assigned an explicit, documented decision, enforced by automated checks. |
| **Placeholder values** (`0`, `-1`) and meaningful blanks. | Converted to missing values or explicit labels, with missing-value indicators. |
| **High-cardinality categories** (`make` 125, `eng_dscr` about 500 values in training). | One-hot encoding with rare-category grouping, learned from training data only. |
| **A CSV reload bug.** The label `"None"` was silently read back as missing by pandas. | Detected by an automated reload check; labels renamed and a safe loader added (`load_processed_split`). |
| **Optimistic scores** from near-identical vehicles across model years. | Measured with a grouped cross-validation check (Section 10.3) and reported as a limitation. |
| **Training time** (Random Forest grid search about 7 minutes: 45 fits of 300 trees). | Kept the grids focused, ran training in the background, and saved models for reuse. |

---

## 13. Limitations

1. **Scope:** the models apply to vehicles with a combustion engine, including hybrids
   and plug-in hybrids. **Pure battery-electric vehicles are not covered.**
2. **Optimistic random-split scores:** for genuinely new vehicle models, the expected
   Random Forest error is about 1.30 MPG RMSE (grouped CV) rather than 0.75.
3. **Hybrids are the least accurate group** (Random Forest RMSE 1.34 vs 0.73 for
   conventional vehicles), and some groups are very small (e.g. 43 plug-in hybrids,
   9 CNG vehicles in the test set).
4. **Overfitting of the tree models:** training RMSE is about 0.36 to 0.37 versus 0.75
   to 0.94 on test data.
5. **Data limitations:** no vehicle weight or aerodynamic data; drive-type labels
   changed around 2009; a few impossible passenger-volume values remain in the data.
   The dataset ends at model year 2021.
6. **Interpretability:** Linear Regression coefficients cannot be interpreted
   individually (multicollinearity, redundant one-hot columns, log target).
   Explanations rely on permutation importance.
7. **Model size:** the Random Forest file is about 76 MB, so trained models are not
   stored in Git and must be recreated by running Notebook 04 (about 10 to 13 minutes on
   the development machine).

---

## 14. Remaining Work

| Item | Description |
|---|---|
| Gradient Boosting and XGBoost | Implement, tune and evaluate (Section 15) |
| Final model comparison | Compare all five models with the same split, pipeline and evaluation code; select the final model |
| Small preprocessing fix | Treat passenger volumes below 40 cubic feet as missing (recommended by the EDA; affects 6 training records), then re-run Notebooks 02 to 05 |
| Merge branch | Open a pull request to merge `member-2-ml-pipeline` (EDA, training, evaluation) into `main` |
| Project documentation | `README.md` is currently empty: add the project description, setup steps and how to run the notebooks |
| Final report and presentation | Extend this progress report with the final results |
| Task allocation | *[to be completed by the group]* |

---

## 15. Future Implementation of Gradient Boosting and XGBoost *(planned)*

**Why these models:** gradient boosting builds trees **sequentially**, each new tree
correcting the errors of the previous ones. It often outperforms Random Forest on
tabular data, and is promising for the weaknesses found here: the remaining error on
hybrids, and the overfitting of deep, fully grown trees.

**Planned approach:**

1. **Same experimental setup:** same cleaned data, the same 80/20 split
   (`random_state = 42`), the same `Pipeline` with `src/preprocessing.py`, and the same
   5-fold cross-validation (`src/modeling.py`), so results are directly comparable.
2. **Gradient Boosting:** scikit-learn `GradientBoostingRegressor` or
   `HistGradientBoostingRegressor`. Hyperparameters to tune include the number of
   trees, the learning rate, tree depth / leaf count and subsampling.
3. **XGBoost:** `xgboost.XGBRegressor` (the `xgboost>=2.0` entry is already prepared in
   `requirements.txt` and needs to be enabled). Hyperparameters to tune include
   `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree` and
   regularisation (`reg_lambda`, `reg_alpha`), with early stopping on a validation fold.
4. **Evaluation:** reuse Notebook 05 unchanged: MAE, MSE, RMSE, R², bootstrap
   confidence intervals, residuals, errors by powertrain, permutation importance and the
   grouped robustness check.
5. **Benchmark to beat:** Random Forest, test RMSE **0.752 MPG** (R² 0.980) and grouped
   CV RMSE **1.30 MPG**. The test set will again be used only once, for the final
   comparison.

No results are available for these models yet.

---

## 16. Conclusion

The project has completed its first major phase. The dataset was inspected; a
leakage-free and reproducible preprocessing pipeline was built (42,493 vehicles,
22 features); an exploratory analysis identified engine size, powertrain, model year
and vehicle class as the main drivers of fuel economy; and three regression models were
trained, tuned with cross-validation and evaluated on a held-out test set.

All three models clearly beat the baseline of predicting the average (RMSE 5.38 MPG).
**The Random Forest Regressor is currently the best model**, with a test RMSE of
**0.752 MPG**, MAE of 0.476 MPG and **R² of 0.980**. Its advantage over the Decision Tree
(RMSE 0.939) and Linear Regression (RMSE 1.823) is consistent across bootstrap resamples,
vehicle groups and the stricter grouped validation. The main open challenges are the
accuracy for hybrid vehicles and generalisation to entirely new vehicle models.

The next stage will implement Gradient Boosting and XGBoost within the same validated
framework, using the Random Forest as the benchmark to beat.

---

*All figures in this report were taken from the executed notebooks and the files in
`results/`. Fields marked "to be completed" require information that is not recorded in
the project repository.*
