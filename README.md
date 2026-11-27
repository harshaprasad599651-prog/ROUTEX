# ROUTEX — Machine-Learning-Based Vehicle Fuel Consumption Prediction

## Project overview
ROUTEX is a supervised regression project that predicts a vehicle's combined fuel economy (`comb08`) in miles per gallon (MPG) from vehicle specifications. The project compares five regression approaches: Linear Regression, Decision Tree, Random Forest, Gradient Boosting and XGBoost.

The modelling workflow uses a leakage-aware preprocessing pipeline, an 80/20 train-test split (`random_state=42`) and five-fold cross-validation. Pure battery-electric vehicles are excluded because their combined rating is MPGe rather than MPG. The trained model is intended for exploratory estimates, not official fuel-economy certification.

## Group information
- Group number: **48**
- Student IDs:
  - EG/2023/5695
  - EG/2023/5515
  - EG/2023/5904
  - EG/2023/5551
- Student names: to be added by the group.
- Course / module: to be confirmed.
- Supervisor: to be confirmed.

## Repository structure
```text
ROUTEX/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/                 # place vehicles.csv here
│   └── processed/
├── models/                  # generated model artifacts
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── reports/
│   ├── figures/
│   ├── member3_data_quality.md
│   └── progress_evaluation.md
├── results/
└── src/
    ├── preprocessing.py
    ├── modeling.py
    ├── member3_data_quality.py
    └── member3_advanced_models.py
```

## Dataset
The project uses the U.S. vehicle fuel-economy dataset (`vehicles.csv`) originally published through fueleconomy.gov / U.S. Department of Energy and EPA. The project copy is referenced in the existing project report as coming from the public `ajhenry/searchable-car-database` GitHub repository.

Place the CSV at:
```text
data/raw/vehicles.csv
```
The dataset is not guaranteed to be included in a fresh Git clone. Download the matching `vehicles.csv` source file and place it at the path above. Do not rename the target column `comb08`.

## Environment setup
Recommended Python version: **3.10 or newer**. Use a virtual environment.

### Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the project in order
Run commands from the repository root.

1. **Dataset inspection:** open and run `notebooks/01_dataset_inspection.ipynb`.
2. **Preprocessing:** open and run `notebooks/02_preprocessing.ipynb`.
3. **EDA:** open and run `notebooks/03_exploratory_data_analysis.ipynb`.
4. **Initial model training:** open and run `notebooks/04_model_training.ipynb`.
5. **Initial model evaluation:** open and run `notebooks/05_model_evaluation.ipynb`.
6. **Member 3 passenger-volume data-quality check:**
   ```bash
   python -m src.member3_data_quality
   ```
7. **Train and compare all five models (full hyperparameter search):**
   ```bash
   python -m src.member3_advanced_models
   ```
   A shorter pipeline check is available with:
   ```bash
   python -m src.member3_advanced_models --quick
   ```
   The full run can take a significant amount of time. It saves models, metrics, predictions and figures under `models/`, `results/` and `reports/figures/`.

## Run the prediction prototype
After the XGBoost model artifact has been generated:
```bash
streamlit run app/streamlit_app.py
```
The app accepts vehicle specifications and displays a predicted combined fuel economy in MPG. The demo requires `models/xgboost.joblib` and the same compatible Python / scikit-learn / XGBoost versions used to train that artifact.

## Main outputs
- `results/member3_model_metrics.csv` — common-split comparison of all five models.
- `results/member3_training_summary.json` — split details, search setup, selected parameters and metrics.
- `results/member3_test_predictions.csv` — held-out test predictions.
- `results/member3_feature_importance.csv` — model feature importance output.
- `reports/member3_data_quality.md` — passenger-volume data-quality decision.
- `reports/figures/06_*.png` — Member 3 model and data-quality figures.
- `reports/progress_evaluation.md` — consolidated project report.

## Final model result in the supplied project snapshot
On the saved 8,499-row test set, XGBoost achieved RMSE **0.71261 MPG**, MAE **0.48276 MPG**, and R² **0.98242**. Its five-fold CV RMSE was **0.77558 MPG**, the lowest among the five models. Gradient Boosting achieved a very similar test RMSE of **0.71342 MPG**. See the consolidated report and CSV results for full metrics.

## Important limitations
- Pure battery-electric vehicles are outside the model scope because the target is MPG, not MPGe.
- The dataset covers model years 1984–2021 in the supplied snapshot.
- Random train-test split results may be more optimistic than performance on entirely unseen makes / models.
- Predictions are estimates and should not be represented as official EPA ratings.
- Serialized model files can be sensitive to library-version changes. Recreate the model in the documented environment if loading produces compatibility warnings or errors.

## Reproducibility
Use the same source data, Python package versions, fixed random seed and commands documented above. Keep the test set held out from model fitting and tuning.
