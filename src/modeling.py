"""Reusable model-building and scoring helpers for ROUTEX.

Every model is wrapped in one sklearn Pipeline::

    Pipeline([("preprocess", build_preprocessor()), ("model", <regressor>)])

so the preprocessing is re-fitted on the training data (or on the training part
of each cross-validation fold) every time the pipeline is fitted. Test data is
never used to fit anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.preprocessing import PROJECT_ROOT, RANDOM_STATE, build_preprocessor

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

N_CV_FOLDS = 5

# Scoring names understood by sklearn's cross-validation tools. The first one
# is used to choose the best hyperparameters.
CV_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}
REFIT_METRIC = "rmse"


def make_cv(n_splits: int = N_CV_FOLDS, random_state: int = RANDOM_STATE) -> KFold:
    """Shuffled K-fold splitter with a fixed seed, so folds are reproducible."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_pipeline(model, scale_numeric: bool = True) -> Pipeline:
    """Wrap a regressor together with the ROUTEX preprocessing step."""
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
        ("model", model),
    ])


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, MAPE (%) and R² for one set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape_%": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "r2": r2_score(y_true, y_pred),
    }


def cv_summary(cv_results: dict, index) -> dict:
    """Mean and standard deviation of the cross-validation scores at `index`.

    Works with the `cv_results_` of GridSearchCV (index = best_index_) and with
    the output of `cross_validate` (index = None -> all folds of one model).
    """
    out = {}
    for name in CV_SCORING:
        if index is None:
            scores = np.asarray(cv_results[f"test_{name}"])
            mean, std = scores.mean(), scores.std()
        else:
            mean = cv_results[f"mean_test_{name}"][index]
            std = cv_results[f"std_test_{name}"][index]
        sign = -1 if CV_SCORING[name].startswith("neg_") else 1
        out[f"cv_{name}"] = sign * mean
        out[f"cv_{name}_std"] = std
    return out


def search_results_table(search) -> pd.DataFrame:
    """Readable table of a GridSearchCV run (one row per hyperparameter combination)."""
    res = pd.DataFrame(search.cv_results_)
    params = res["params"].apply(pd.Series)
    params.columns = [c.replace("model__", "").replace("regressor__", "") for c in params.columns]
    table = pd.concat([params, pd.DataFrame({
        "cv_rmse": -res["mean_test_rmse"],
        "cv_rmse_std": res["std_test_rmse"],
        "cv_mae": -res["mean_test_mae"],
        "cv_r2": res["mean_test_r2"],
        "train_rmse": -res["mean_train_rmse"] if "mean_train_rmse" in res else np.nan,
        "fit_time_s": res["mean_fit_time"],
        "rank": res[f"rank_test_{REFIT_METRIC}"],
    })], axis=1)
    return table.sort_values("rank").reset_index(drop=True)
