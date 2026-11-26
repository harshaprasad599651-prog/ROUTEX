"""Reusable preprocessing for the ROUTEX fuel-economy regression project.

Typical use (from a notebook or training script)::

    from src.preprocessing import (
        load_raw_data, clean_dataset, split_features_target, build_preprocessor,
    )

    raw = load_raw_data()
    clean, report = clean_dataset(raw)
    X_train, X_test, y_train, y_test = split_features_target(clean)

    # Fit the preprocessor on the TRAINING data only, usually inside a Pipeline:
    pipe = Pipeline([("preprocess", build_preprocessor()), ("model", LinearRegression())])
    pipe.fit(X_train, y_train)

Design rules
------------
* ``clean_dataset`` only applies fixed, row-by-row rules (filtering, renaming,
  feature extraction). It learns nothing from the data, so it is safe to run
  before the train/test split.
* Everything that *learns* from the data (median imputation, scaling,
  category lists) lives in the ``ColumnTransformer`` returned by
  ``build_preprocessor`` and must be fitted on the training set only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Reproducibility and paths
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "vehicles.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "comb08"
ID_COLUMN = "id"

# --------------------------------------------------------------------------
# Columns removed from the model inputs, grouped by reason
# --------------------------------------------------------------------------
DROPPED_COLUMNS: dict[str, tuple[str, list[str]]] = {
    "leakage_required": (
        "Target leakage (named in the project brief): another form of the same "
        "fuel-economy measurement as comb08.",
        ["city08", "highway08", "UCity", "UHighway", "fuelCost08",
         "barrels08", "co2TailpipeGpm", "youSaveSpend"],
    ),
    "leakage_additional": (
        "Target leakage (found during inspection): unrounded MPG values, CO2 and "
        "EPA scores calculated from fuel economy, and the gas-guzzler tax flag, "
        "which is assigned from the MPG figure itself.",
        ["comb08U", "city08U", "highway08U", "co2", "feScore", "ghgScore", "guzzler"],
    ),
    "leakage_alternative_fuel": (
        "Target leakage: fuel-economy results for a vehicle's second fuel "
        "(E85, electricity, CNG...). They come from the same test as comb08.",
        ["cityA08", "cityA08U", "highwayA08", "highwayA08U", "combA08", "combA08U",
         "UCityA", "UHighwayA", "barrelsA08", "fuelCostA08", "co2A",
         "co2TailpipeAGpm", "ghgScoreA"],
    ),
    "leakage_electric": (
        "Target leakage: electricity consumption, charge-depleting results and "
        "driving range. These are outcomes of the efficiency test, not vehicle specs.",
        ["cityE", "highwayE", "combE", "cityCD", "highwayCD", "combinedCD",
         "cityUF", "highwayUF", "combinedUF", "phevCity", "phevHwy", "phevComb",
         "range", "rangeCity", "rangeHwy", "rangeCityA", "rangeHwyA", "rangeA"],
    ),
    "identifier": (
        "Identifier: unique or near-unique codes that would let a model memorise "
        "records instead of learning general patterns.",
        ["id", "engId", "model"],
    ),
    "metadata": (
        "Record metadata about the fueleconomy.gov website, not the vehicle.",
        ["createdOn", "modifiedOn", "mpgData"],
    ),
    "constant": (
        "Constant: the same value in every row, so it carries no information.",
        ["charge120"],
    ),
    "sparse_or_redundant": (
        "Almost always empty after excluding pure EVs (EV charging / motor details), "
        "or duplicates another kept column (mfrCode duplicates make).",
        ["charge240", "charge240b", "c240Dscr", "c240bDscr", "evMotor", "mfrCode"],
    ),
    "replaced_by_engineered": (
        "Replaced by an engineered feature that keeps the same information in a "
        "cleaner form (see engineer_features).",
        ["pv2", "pv4", "hpv", "lv2", "lv4", "hlv", "tCharger", "sCharger"],
    ),
}

LEAKAGE_COLUMNS: list[str] = [
    col
    for key, (_, cols) in DROPPED_COLUMNS.items()
    if key.startswith("leakage")
    for col in cols
]

# --------------------------------------------------------------------------
# Model input features
# --------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "year", "displ", "cylinders", "trans_speeds", "passenger_volume", "luggage_volume",
]
BINARY_FEATURES = ["turbo", "supercharger", "is_cvt", "phevBlended"]
CATEGORICAL_FEATURES = [
    "make", "VClass", "drive", "trany", "trans_type", "trans_dscr", "fuelType", "fuelType1",
    "fuelType2", "atvType", "startStop", "eng_dscr",
]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Categories seen fewer times than this in the training data are grouped into
# one "infrequent" category by the OneHotEncoder.
MIN_CATEGORY_FREQUENCY = 20


# --------------------------------------------------------------------------
# Loading and cleaning (fixed rules only - nothing is learned here)
# --------------------------------------------------------------------------
def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw vehicles.csv file without changing it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download vehicles.csv into data/raw/ first."
        )
    return pd.read_csv(path, low_memory=False)


def is_pure_electric(df: pd.DataFrame) -> pd.Series:
    """True for battery-electric vehicles, whose comb08 is MPGe, not MPG."""
    return df["fuelType1"].eq("Electricity")


def clean_engine_description(series: pd.Series) -> pd.Series:
    """Remove the GUZZLER tag (leakage) and tidy the whitespace in eng_dscr."""
    cleaned = (
        series.fillna("")
        .str.replace(r"\(?GUZZLER\)?", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return cleaned.mask(cleaned.eq(""), "Not specified")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cleaner versions of several raw columns. Returns a new DataFrame."""
    out = df.copy()

    # Transmission: "Automatic (S6)" -> type "Automatic", 6 speeds.
    # Continuously variable transmissions have no fixed number of gears.
    # (Missing trany values stay NaN and are imputed later inside the pipeline.)
    trany = out["trany"]
    out["trans_type"] = trany.str.split().str[0]
    out["is_cvt"] = (
        trany.str.contains(r"variable|\(AV", regex=True).eq(True).astype(int)
    )
    speeds = pd.to_numeric(trany.str.extract(r"(\d+)")[0], errors="coerce")
    out["trans_speeds"] = speeds.astype(float).where(out["is_cvt"].eq(0))

    # Interior volumes: the dataset stores them in three column pairs (2-door,
    # 4-door, hatchback) and uses 0 for "not reported". Combine each group into
    # one column and turn 0 into a real missing value.
    out["passenger_volume"] = out[["pv2", "pv4", "hpv"]].replace(0, np.nan).max(axis=1)
    out["luggage_volume"] = out[["lv2", "lv4", "hlv"]].replace(0, np.nan).max(axis=1)

    # Forced induction: the raw columns contain "T"/"S" or nothing.
    out["turbo"] = out["tCharger"].notna().astype(int)
    out["supercharger"] = out["sCharger"].notna().astype(int)
    out["phevBlended"] = out["phevBlended"].astype(int)

    # Missing values that have a known meaning get an explicit label.
    out["atvType"] = out["atvType"].fillna("Conventional")
    # (Labels are chosen so pandas.read_csv never mistakes them for missing values,
    # which would happen with e.g. "None" or "NA".)
    out["fuelType2"] = out["fuelType2"].fillna("Single fuel")
    out["startStop"] = out["startStop"].fillna("Not reported")
    out["trans_dscr"] = out["trans_dscr"].fillna("Not specified")
    out["eng_dscr"] = clean_engine_description(out["eng_dscr"])
    return out


def clean_dataset(
    raw: pd.DataFrame, exclude_electric: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Apply all rule-based cleaning steps.

    Returns the cleaned frame (model features + target) and a small report
    with the number of rows removed at each step.
    """
    report = {"rows_raw": len(raw)}
    df = raw.copy()

    if exclude_electric:
        ev_mask = is_pure_electric(df)
        report["rows_removed_pure_ev"] = int(ev_mask.sum())
        df = df.loc[~ev_mask]

    # Records that are identical in every column except their id.
    dup_mask = df.drop(columns=[ID_COLUMN]).duplicated(keep="first")
    report["rows_removed_duplicates"] = int(dup_mask.sum())
    df = df.loc[~dup_mask]

    df = engineer_features(df)
    df = df[FEATURES + [TARGET]].reset_index(drop=True)
    report["rows_clean"] = len(df)
    report["n_features"] = len(FEATURES)
    return df, report


def column_accounting(raw_columns) -> pd.DataFrame:
    """List every raw column and what happened to it (kept or dropped + why)."""
    rows = []
    dropped = {
        col: (group, reason)
        for group, (reason, cols) in DROPPED_COLUMNS.items()
        for col in cols
    }
    for col in raw_columns:
        if col == TARGET:
            rows.append((col, "target", "Prediction target (combined MPG)."))
        elif col in dropped:
            rows.append((col, "dropped: " + dropped[col][0], dropped[col][1]))
        elif col in FEATURES:
            rows.append((col, "kept", "Used as a model input."))
        else:
            rows.append((col, "UNACCOUNTED", ""))
    return pd.DataFrame(rows, columns=["column", "decision", "reason"])


# --------------------------------------------------------------------------
# Splitting and the learned preprocessing step
# --------------------------------------------------------------------------
def split_features_target(
    clean: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Split the cleaned data into train/test feature and target sets."""
    X = clean[FEATURES]
    y = clean[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_processed_split(processed_dir: Path | str = PROCESSED_DIR):
    """Load data/processed/train.csv and test.csv as X_train, X_test, y_train, y_test.

    Only empty cells are read as missing, so category labels are never turned
    into NaN by pandas' default missing-value strings.
    """
    processed_dir = Path(processed_dir)
    frames = []
    for name in ("train", "test"):
        path = processed_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run notebooks/02_preprocessing.ipynb first.")
        frames.append(pd.read_csv(path, keep_default_na=False, na_values=[""]))
    train, test = frames
    return train[FEATURES], test[FEATURES], train[TARGET], test[TARGET]


def build_preprocessor(
    min_frequency: int = MIN_CATEGORY_FREQUENCY, scale_numeric: bool = True
) -> ColumnTransformer:
    """Return an UNFITTED ColumnTransformer.

    * numeric: median imputation (+ missing-value indicator columns) and scaling
    * binary: passed through as 0/1
    * categorical: constant "Missing" imputation and one-hot encoding; rare
      categories are grouped and unseen categories are handled safely.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
