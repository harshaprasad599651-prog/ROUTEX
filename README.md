# Routex – Machine Learning-Based Vehicle Fuel Consumption Prediction

Routex is a university Machine Learning mini project. It predicts a vehicle's
**combined fuel economy** (`comb08`, in miles per gallon) from the vehicle's
technical specifications.

## 1. Problem statement

Fuel economy matters to buyers, fleet operators and anyone studying the
environment. Given a vehicle's specifications (for example engine size, number
of cylinders, transmission, drive type, fuel type, vehicle class and model
year), can we predict its combined MPG accurately?

This is a **supervised regression** problem:

- **Input (features):** vehicle technical specifications
- **Output (target):** `comb08`, the combined city/highway fuel economy in MPG

## 2. Dataset

- **Source:** `vehicles.csv` from
  [ajhenry/searchable-car-database](https://github.com/ajhenry/searchable-car-database)
  (based on US EPA / fueleconomy.gov data)
- **Size:** more than 40,000 vehicle records
- **Usage:** we use **only the dataset** from that repository. All
  preprocessing, analysis, modelling, evaluation, visualisation and prototype
  code in this project is our own work.

The dataset is **not stored in this repository** (see `.gitignore`). Each team
member downloads `vehicles.csv` and places it in `data/raw/`.

> **Note on data leakage:** the dataset also contains other fuel-economy
> columns (such as city and highway MPG, fuel cost and CO2 figures). These are
> computed from the same measurements as `comb08`, so using them as features
> would let the model "cheat". We will select features carefully during
> preprocessing.

## 3. Models

| # | Model | Phase |
|---|-------|-------|
| 1 | Linear Regression | Progress evaluation |
| 2 | Decision Tree Regression | Progress evaluation |
| 3 | Random Forest Regression | Progress evaluation |
| 4 | Gradient Boosting Regression | Final submission |
| 5 | XGBoost Regression | Final submission |

All models will be trained and tested on the same train/test split, so their
results can be compared fairly. Planned evaluation metrics:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

## 4. Project structure

```
Routex_ML_Project/
├── data/
│   ├── raw/            # Original vehicles.csv, never edited by hand
│   └── processed/      # Cleaned data produced by our preprocessing code
├── notebooks/          # Jupyter notebooks for exploration (EDA) and experiments
├── src/                # Reusable Python code (preprocessing, training, evaluation)
├── models/             # Saved trained models (.joblib files)
├── results/            # Model evaluation results (e.g. metrics tables as CSV)
├── reports/
│   └── figures/        # Charts and plots used in the report and presentation
├── requirements.txt    # Python libraries needed to run the project
├── .gitignore          # Files and folders Git should not track
└── README.md           # This file
```

### Purpose of each folder

| Folder | Purpose |
|--------|---------|
| `data/raw/` | Holds the original dataset exactly as downloaded. Keeping it untouched means we can always re-run the pipeline from the start. |
| `data/processed/` | Holds cleaned and prepared data created by our code (missing values handled, features selected, categories encoded). |
| `notebooks/` | Jupyter notebooks for exploring the data, trying ideas and showing analysis step by step. Good for explaining our reasoning during evaluation. |
| `src/` | Python scripts with the final, reusable pipeline code. Each model is meant to be added in one place, so Gradient Boosting and XGBoost can be added later without restructuring. |
| `models/` | Trained models saved to disk, so they can be reloaded for evaluation or the prototype without retraining. |
| `results/` | Numerical outputs such as model comparison tables (MAE, RMSE, R²). |
| `reports/figures/` | Saved visualisations (distributions, correlation heatmaps, predicted-vs-actual plots, feature importance). |

Empty folders contain a `.gitkeep` file so that Git keeps them in the
repository.

## 5. Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download vehicles.csv from the dataset repository and place it in:
#    data/raw/vehicles.csv
```

## 6. Planned workflow

1. **Data understanding:** load the data and inspect columns, types and missing values
2. **Exploratory data analysis:** visualise the target and its relationship with features
3. **Preprocessing:** select features, handle missing values, encode categories, scale where needed
4. **Train/test split:** a fixed `random_state` for reproducibility
5. **Model training:** Linear Regression, Decision Tree, Random Forest (later Gradient Boosting, XGBoost)
6. **Evaluation and comparison:** MAE, RMSE, R² and visual comparisons
7. **Prototype:** a simple interface that predicts MPG from user-entered specifications

## 7. Reproducibility

- All library versions are listed in `requirements.txt`.
- A fixed random seed will be used for data splitting and model training.
- Raw data is never modified; every processed file is generated by code.

## 8. Project status

- [x] Project structure and documentation
- [ ] Data loading and exploration
- [ ] Preprocessing
- [ ] Linear Regression, Decision Tree, Random Forest (progress evaluation)
- [ ] Gradient Boosting, XGBoost (final)
- [ ] Prototype

## 9. Team

| Name | Role |
|------|------|
| _Member 1_ | _To be filled in_ |
| _Member 2_ | _To be filled in_ |
| _Member 3_ | _To be filled in_ |
| _Member 4_ | _To be filled in_ |
