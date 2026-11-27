"""Member 3 - Gradient Boosting and XGBoost regression for ROUTEX.

Same experimental setup as notebook 04 (Member 2), so results are directly
comparable:
  * same cleaned data and 80/20 split (random_state = 42),
  * same Pipeline: build_preprocessor() + model (fitted on training data only),
  * same 5-fold cross-validation and scoring (src/modeling.py),
  * the test set is used once, at the end, for the final comparison.

The passenger-volume fix (fix_passenger_volume in src/member3_data_quality.py;
src/preprocessing.py is not modified) changes the input data slightly, so Linear Regression, Decision Tree and Random Forest
are retrained with Member 2's tuned hyperparameters on the corrected data. All
five models are therefore compared on identical data.

Run from the project root:
    python -m src.member3_advanced_models              # full run (~15-30 min)
    python -m src.member3_advanced_models --quick      # fast check (~3-5 min)
    python -m src.member3_advanced_models --gb-variant classic   # sklearn GradientBoostingRegressor
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.tree import DecisionTreeRegressor

from src import modeling as mdl
from src import preprocessing as prep
from src.member3_data_quality import MIN_PASSENGER_VOLUME, fix_passenger_volume

RANDOM_STATE = prep.RANDOM_STATE
FIGURES_DIR = prep.PROJECT_ROOT / "reports" / "figures"
MEMBER2_RF_TEST_RMSE = 0.752  # benchmark from the progress report (before the fix)

BLUE, ORANGE, GREEN, GREY, TEXT = "#257de0", "#f47230", "#00b894", "#999999", "#191919"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e6e6", "axes.axisbelow": True, "legend.frameon": False,
    "axes.titleweight": "bold",
})

FILE_NAMES = {
    "Linear Regression": "linear_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "XGBoost": "xgboost",
}


# --------------------------------------------------------------------------
# Models and search spaces
# --------------------------------------------------------------------------
def reference_models() -> dict:
    """Member 2's three models with the hyperparameters chosen in notebook 04."""
    return {
        "Linear Regression": TransformedTargetRegressor(
            regressor=LinearRegression(), func=np.log, inverse_func=np.exp),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=24, min_samples_leaf=1, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_features=0.6, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1),
    }


def gradient_boosting_setup(variant: str):
    """Model + hyperparameter space for Gradient Boosting.

    "hist" (default): HistGradientBoostingRegressor. Same gradient-boosting
    algorithm, but it bins each feature into at most 255 values first, which
    makes it about 10-15x faster on our 34,000 x 267 training matrix.
    "classic": GradientBoostingRegressor (slower; smaller search space).
    """
    if variant == "classic":
        model = GradientBoostingRegressor(random_state=RANDOM_STATE)
        space = {
            "model__n_estimators": [300, 500],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [4, 6],
            "model__subsample": [0.8, 1.0],
            "model__max_features": [0.5],
        }
        return model, space, -1  # single-threaded model -> parallel search

    model = HistGradientBoostingRegressor(early_stopping=False, random_state=RANDOM_STATE)
    space = {
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__max_iter": [300, 600, 1000],        # number of boosting rounds (trees)
        "model__max_leaf_nodes": [15, 31, 63, 127],  # tree size
        "model__min_samples_leaf": [5, 10, 20, 40],
        "model__l2_regularization": [0.0, 0.1, 1.0],
        "model__max_features": [0.5, 0.8, 1.0],      # share of columns tried per split
    }
    return model, space, 1  # model already uses all CPU cores


def xgboost_setup():
    try:
        from xgboost import XGBRegressor
    except ImportError as err:
        raise SystemExit(
            "xgboost is not installed. Run:  python -m pip install xgboost\n"
            "(or skip it with --skip-xgb)"
        ) from err

    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    space = {
        "model__n_estimators": [400, 800, 1200],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [4, 6, 8, 10],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.7, 0.85, 1.0],          # share of rows per tree
        "model__colsample_bytree": [0.5, 0.7, 1.0],    # share of columns per tree
        "model__reg_lambda": [0.5, 1.0, 5.0],          # L2 regularisation
        "model__reg_alpha": [0.0, 0.1, 1.0],           # L1 regularisation
    }
    return model, space, 1


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def with_mse(metrics: dict) -> dict:
    """Add MSE (= RMSE squared) next to the metrics from src/modeling.py."""
    out = dict(metrics)
    out["mse"] = metrics["rmse"] ** 2
    return out


def original_feature(encoded_name: str) -> str:
    """Map an encoded column (e.g. 'cat__make_Ford') back to its feature ('make')."""
    block, _, rest = encoded_name.partition("__")
    if block == "num":
        return rest.replace("missingindicator_", "")
    if block == "bin":
        return rest
    # Longest match first, so 'fuelType1_...' is not mistaken for 'fuelType'.
    for feat in sorted(prep.CATEGORICAL_FEATURES, key=len, reverse=True):
        if rest.startswith(feat + "_"):
            return feat
    return rest


def grouped_importance(pipe) -> pd.Series | None:
    """Model importance summed over all encoded columns of each original feature."""
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None  # HistGradientBoosting has no built-in importance
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    imp = pd.Series(model.feature_importances_, index=names)
    return imp.groupby(imp.index.map(original_feature)).sum().sort_values(ascending=False)


def evaluate(name, pipe, X_train, y_train, X_test, y_test, cv_scores, extra=None) -> tuple:
    train_pred = pipe.predict(X_train)
    start = time.perf_counter()
    test_pred = pipe.predict(X_test)
    predict_time = time.perf_counter() - start
    row = {
        "model": name,
        **{f"train_{k}": v for k, v in with_mse(mdl.regression_metrics(y_train, train_pred)).items()},
        **cv_scores,
        **{f"test_{k}": v for k, v in with_mse(mdl.regression_metrics(y_test, test_pred)).items()},
        "test_predict_time_s": predict_time,
        **(extra or {}),
    }
    return row, test_pred


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="small search (4 candidates, 3 folds) to check everything runs")
    parser.add_argument("--n-iter", type=int, default=12,
                        help="random hyperparameter combinations per model (default 12)")
    parser.add_argument("--gb-variant", choices=["hist", "classic"], default="hist")
    parser.add_argument("--skip-xgb", action="store_true")
    args = parser.parse_args()

    n_iter = 4 if args.quick else args.n_iter
    cv = mdl.make_cv(n_splits=3 if args.quick else mdl.N_CV_FOLDS)
    for folder in (mdl.MODELS_DIR, mdl.RESULTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    print(f"Python {platform.python_version()} | scikit-learn {sklearn.__version__} | "
          f"random_state {RANDOM_STATE} | CV folds {cv.get_n_splits()} | "
          f"search candidates per model {n_iter}")

    # ------------------------------------------------------------ data
    raw = prep.load_raw_data()
    clean, clean_report = prep.clean_dataset(raw)
    X_train, X_test, y_train, y_test = prep.split_features_target(clean)
    # Member 3 data-quality fix: impossible passenger volumes -> missing.
    # Fixed rule (nothing learned), applied the same way to both sets.
    n_fixed = (int((X_train["passenger_volume"] < MIN_PASSENGER_VOLUME).sum()),
               int((X_test["passenger_volume"] < MIN_PASSENGER_VOLUME).sum()))
    X_train, X_test = fix_passenger_volume(X_train), fix_passenger_volume(X_test)
    print(f"Passenger-volume fix: {n_fixed[0]} training and {n_fixed[1]} test values set to missing")
    print(f"Training set: {len(X_train):,} rows | Test set: {len(X_test):,} rows")

    # Leakage guard (same checks as notebook 04)
    assert list(X_train.columns) == prep.FEATURES
    assert not set(prep.LEAKAGE_COLUMNS) & set(X_train.columns)
    assert prep.TARGET not in X_train.columns and prep.ID_COLUMN not in X_train.columns
    assert X_train["passenger_volume"].dropna().min() >= MIN_PASSENGER_VOLUME
    print("Leakage and data-quality checks passed.\n")

    rows, test_predictions, fitted, search_tables, best_params = [], {}, {}, {}, {}

    # --------------------------------- 1. reference models (Member 2's settings)
    for name, model in reference_models().items():
        pipe = mdl.make_pipeline(model)
        start = time.perf_counter()
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=mdl.CV_SCORING, n_jobs=1)
        pipe.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        row, pred = evaluate(name, pipe, X_train, y_train, X_test, y_test,
                             mdl.cv_summary(cv_res, None), {"search_time_s": elapsed})
        rows.append(row)
        test_predictions[name], fitted[name] = pred, pipe
        print(f"{name:<18} CV RMSE {row['cv_rmse']:.3f} | test RMSE {row['test_rmse']:.3f} "
              f"({elapsed:.0f} s)")

    # --------------------------------- 2. boosting models (tuned here)
    boosting = {"Gradient Boosting": gradient_boosting_setup(args.gb_variant)}
    if not args.skip_xgb:
        boosting["XGBoost"] = xgboost_setup()

    for name, (model, space, search_jobs) in boosting.items():
        print(f"\nTuning {name}: {n_iter} combinations x {cv.get_n_splits()} folds "
              f"= {n_iter * cv.get_n_splits()} fits ...")
        search = RandomizedSearchCV(
            mdl.make_pipeline(model), space, n_iter=n_iter, cv=cv,
            scoring=mdl.CV_SCORING, refit=mdl.REFIT_METRIC, return_train_score=True,
            random_state=RANDOM_STATE, n_jobs=search_jobs, verbose=1,
        )
        start = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        print(f"Best parameters: {params}  ({elapsed:.0f} s)")

        pipe = search.best_estimator_
        row, pred = evaluate(name, pipe, X_train, y_train, X_test, y_test,
                             mdl.cv_summary(search.cv_results_, search.best_index_),
                             {"search_time_s": elapsed, "refit_time_s": search.refit_time_})
        rows.append(row)
        test_predictions[name], fitted[name] = pred, pipe
        search_tables[name] = mdl.search_results_table(search)
        best_params[name] = {"estimator": type(model).__name__,
                             **{k: (v.item() if hasattr(v, "item") else v)
                                for k, v in params.items()}}
        print(f"{name:<18} CV RMSE {row['cv_rmse']:.3f} | test RMSE {row['test_rmse']:.3f}")

    # --------------------------------- 3. comparison table
    base = y_train.mean()
    rows.append({"model": "Baseline (training mean)",
                 **{f"train_{k}": v for k, v in with_mse(mdl.regression_metrics(
                     y_train, np.full(len(y_train), base))).items()},
                 **{f"test_{k}": v for k, v in with_mse(mdl.regression_metrics(
                     y_test, np.full(len(y_test), base))).items()}})
    metrics = pd.DataFrame(rows).set_index("model")

    models_only = metrics.drop(index="Baseline (training mean)")
    best_name = models_only["cv_rmse"].idxmin()  # choose by CV, not by test score
    rf_rmse = metrics.loc["Random Forest", "test_rmse"]

    show = ["cv_rmse", "test_mae", "test_mse", "test_rmse", "test_r2", "train_rmse"]
    print("\n" + "=" * 78)
    print("FINAL COMPARISON (same data, same split, same pipeline)")
    print("=" * 78)
    print(metrics[show].round(4).to_string())
    print("-" * 78)
    for name in boosting:
        diff = metrics.loc[name, "test_rmse"] - rf_rmse
        print(f"{name} vs Random Forest: test RMSE {diff:+.3f} MPG "
              f"({diff / rf_rmse * 100:+.1f}%) -> "
              f"{'BETTER' if diff < 0 else 'not better'} than Random Forest")
    print(f"(Member 2's reported Random Forest benchmark before the data fix: "
          f"{MEMBER2_RF_TEST_RMSE} MPG; retrained here: {rf_rmse:.3f} MPG)")
    print(f"Best model by cross-validation RMSE: {best_name}")

    # --------------------------------- 4. save results
    metrics.to_csv(mdl.RESULTS_DIR / "member3_model_metrics.csv", float_format="%.5f")
    for name, table in search_tables.items():
        table.to_csv(mdl.RESULTS_DIR / f"cv_results_{FILE_NAMES[name]}.csv",
                     index=False, float_format="%.5f")

    preds = X_test[["year", "make", "VClass", "atvType", "displ", "cylinders"]].copy()
    preds.insert(0, "test_row", np.arange(len(X_test)))
    preds["actual_comb08"] = y_test.values
    for name, pred in test_predictions.items():
        preds[f"pred_{FILE_NAMES[name]}"] = np.round(pred, 4)
    preds.to_csv(mdl.RESULTS_DIR / "member3_test_predictions.csv", index=False)

    for name in boosting:
        joblib.dump(fitted[name], mdl.MODELS_DIR / f"{FILE_NAMES[name]}.joblib", compress=3)

    summary = {
        "target": prep.TARGET,
        "random_state": RANDOM_STATE,
        "split": {"train_rows": len(X_train), "test_rows": len(X_test)},
        "cross_validation": {"folds": cv.get_n_splits(), "selection_metric": "RMSE",
                             "search": f"RandomizedSearchCV, {n_iter} candidates per model"},
        "data_quality_fix": f"passenger_volume < {MIN_PASSENGER_VOLUME} cu ft -> missing",
        "best_params": best_params,
        "best_model_by_cv": best_name,
        "test_metrics": metrics[show].round(4).to_dict(orient="index"),
        "member2_random_forest_benchmark_rmse": MEMBER2_RF_TEST_RMSE,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__},
    }
    if "XGBoost" in boosting:
        import xgboost
        summary["versions"]["xgboost"] = xgboost.__version__
    with open(mdl.RESULTS_DIR / "member3_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # --------------------------------- 5. figures
    names = list(models_only.index)
    colours = [GREY, GREY, BLUE, ORANGE, GREEN][:len(names)]

    # 5a. RMSE of all models
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(names))
    for i, (col, label, alpha) in enumerate([("train_rmse", "Training", 0.35),
                                             ("cv_rmse", "Cross-validation", 0.65),
                                             ("test_rmse", "Test", 1.0)]):
        vals = metrics.loc[names, col]
        bars = ax.bar(x + (i - 1) * 0.26, vals, width=0.24, color=colours, alpha=alpha,
                      edgecolor="white", label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylabel("RMSE (MPG), lower is better")
    ax.set_title(f"All {len(names)} models: training, cross-validation and test RMSE")
    ax.legend(title="bars, left to right")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIGURES_DIR / "06_all_models_rmse_comparison.png")
    plt.close(fig)

    # 5b. Predicted vs actual (Random Forest benchmark + boosting models)
    compare = ["Random Forest", *boosting]
    lims = (5, 62)
    fig, axes = plt.subplots(1, len(compare), figsize=(5 * len(compare), 4.8),
                             sharex=True, sharey=True)
    for ax, name in zip(np.atleast_1d(axes), compare):
        ax.scatter(y_test, test_predictions[name], s=7, alpha=0.3, color=BLUE, edgecolors="none")
        ax.plot(lims, lims, "--", color=TEXT, linewidth=1, label="Perfect prediction")
        m = metrics.loc[name]
        ax.set_title(f"{name}\nTest RMSE {m.test_rmse:.3f} MPG, R² {m.test_r2:.3f}", fontsize=11)
        ax.set_xlabel("Actual comb08 (MPG)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    np.atleast_1d(axes)[0].set_ylabel("Predicted comb08 (MPG)")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_boosting_predicted_vs_actual.png")
    plt.close(fig)

    # 5c. Residuals
    fig, axes = plt.subplots(1, len(compare), figsize=(5 * len(compare), 4.2),
                             sharex=True, sharey=True)
    for ax, name in zip(np.atleast_1d(axes), compare):
        pred = test_predictions[name]
        ax.scatter(pred, y_test.values - pred, s=6, alpha=0.3, color=BLUE, edgecolors="none")
        ax.axhline(0, color=TEXT, linewidth=1)
        ax.set_title(f"{name} residuals")
        ax.set_xlabel("Predicted comb08 (MPG)")
    np.atleast_1d(axes)[0].set_ylabel("Actual - predicted (MPG)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_boosting_residuals.png")
    plt.close(fig)

    # 5d. Feature importance (built-in importance, grouped by original feature)
    importances = {n: grouped_importance(fitted[n]) for n in compare}
    importances = {n: s for n, s in importances.items() if s is not None}
    if importances:
        fig, axes = plt.subplots(1, len(importances), figsize=(5.5 * len(importances), 5))
        for ax, (name, imp) in zip(np.atleast_1d(axes), importances.items()):
            top = imp.head(12)[::-1]
            ax.barh(top.index, top.values, color=BLUE)
            ax.set_title(f"{name}: top 12 features")
            ax.set_xlabel("share of importance")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "06_boosting_feature_importance.png")
        plt.close(fig)
        pd.DataFrame(importances).to_csv(mdl.RESULTS_DIR / "member3_feature_importance.csv",
                                         float_format="%.5f")

    print("\nSaved: results/member3_model_metrics.csv, results/member3_training_summary.json,")
    print("       results/member3_test_predictions.csv, results/cv_results_*.csv,")
    print("       reports/figures/06_*.png, models/gradient_boosting.joblib, models/xgboost.joblib")


if __name__ == "__main__":
    main()
