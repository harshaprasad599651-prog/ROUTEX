"""Member 3 - Data quality: impossible passenger-volume values.

The EDA (notebook 03) found a few vehicles whose passenger compartment is only
1 to 6 cubic feet. No real car is that small (a two-seat sports car has about
40-50 cu ft), so these are data-entry errors.

Decision: treat passenger volumes below MIN_PASSENGER_VOLUME as missing.
The fix lives here (src/preprocessing.py is not modified). It is applied with
fix_passenger_volume() to the cleaned feature table, so the pipeline then fills
those values with the training median, like any other unreported volume.

Run from the project root:
    python -m src.member3_data_quality
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import preprocessing as prep

# Smallest believable passenger compartment, in cubic feet.
MIN_PASSENGER_VOLUME = 40

RESULTS_DIR = prep.PROJECT_ROOT / "results"
FIGURES_DIR = prep.PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = prep.PROJECT_ROOT / "reports" / "member3_data_quality.md"


def fix_passenger_volume(X: pd.DataFrame, limit: float = MIN_PASSENGER_VOLUME) -> pd.DataFrame:
    """Return a copy of X where impossible passenger volumes (< limit) are missing.

    A fixed rule written by hand: nothing is learned from the data, so it is
    safe to apply to the training and the test set in the same way.
    """
    out = X.copy()
    pv = out["passenger_volume"]
    out["passenger_volume"] = pv.where(pv.isna() | (pv >= limit))
    return out


def impossible_mask(X: pd.DataFrame, limit: float = MIN_PASSENGER_VOLUME) -> pd.Series:
    """True for rows whose passenger volume is reported but impossibly small."""
    pv = X["passenger_volume"]
    return pv.notna() & (pv < limit)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    limit = MIN_PASSENGER_VOLUME

    # ------------------------------------------------------------ 1. find
    raw = prep.load_raw_data()
    clean, _ = prep.clean_dataset(raw)          # Member 2's cleaning, unchanged
    X_train, X_test, y_train, y_test = prep.split_features_target(clean)

    bad = impossible_mask(clean)
    # Look up model name and raw columns for the report (same row order as clean).
    kept = raw.loc[~prep.is_pure_electric(raw)]
    kept = kept.loc[~kept.drop(columns=[prep.ID_COLUMN]).duplicated(keep="first")]
    kept = kept.reset_index(drop=True)
    flagged = kept.loc[bad, ["id", "year", "make", "model", "VClass", "pv2", "pv4", "hpv"]].copy()
    flagged["passenger_volume"] = clean.loc[bad, "passenger_volume"]
    flagged["comb08"] = clean.loc[bad, prep.TARGET]
    flagged["split"] = np.where(flagged.index.isin(X_train.index), "train", "test")
    flagged = flagged.sort_values("year")

    pv = clean["passenger_volume"]
    print("=" * 70)
    print("PASSENGER VOLUME - DATA QUALITY CHECK")
    print("=" * 70)
    print(f"Vehicles in cleaned data:        {len(clean):,}")
    print(f"Passenger volume reported:       {pv.notna().sum():,}")
    print(f"Plausibility limit:              >= {limit} cubic feet")
    print(f"Impossible values found:         {len(flagged)} "
          f"({(flagged['split'] == 'train').sum()} train, {(flagged['split'] == 'test').sum()} test)\n")
    print(flagged.to_string(index=False))

    smallest_ok = (kept.assign(passenger_volume=pv).loc[pv >= limit]
                   .nsmallest(8, "passenger_volume"))
    print("\nSmallest believable passenger volumes (for comparison):")
    print(smallest_ok[["year", "make", "model", "VClass", "passenger_volume"]].to_string(index=False))

    # ------------------------------------------------- 2. apply + check
    fixed = fix_passenger_volume(clean)
    changed = pv.notna() & fixed["passenger_volume"].isna()
    assert len(fixed) == len(clean), "Row count changed!"
    assert changed.equals(bad), "Something other than the impossible values changed!"
    assert fixed.drop(columns="passenger_volume").equals(clean.drop(columns="passenger_volume"))
    assert fixed["passenger_volume"].dropna().min() >= limit
    print(f"\nChecks passed: same rows and split, only {int(changed.sum())} values changed, "
          f"smallest passenger volume is now {fixed['passenger_volume'].min():.0f} cu ft.")

    # ---------------------------------------------------------- 3. evidence
    flagged.to_csv(RESULTS_DIR / "member3_passenger_volume_issues.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(pv.dropna(), bins=80, color="#bbbbbb")
    axes[0].axvline(limit, color="crimson", linestyle="--", label=f"limit = {limit} cu ft")
    axes[0].plot(pv[bad], np.zeros(int(bad.sum())), "v", color="crimson", markersize=9,
                 clip_on=False, label="impossible values")
    axes[0].set_title(f"Before: {int(bad.sum())} impossible values (red markers)")
    axes[0].legend()
    axes[1].hist(fixed["passenger_volume"].dropna(), bins=80, color="#257de0")
    axes[1].set_title("After: impossible values treated as missing")
    for ax in axes:
        ax.set_xlabel("passenger_volume (cubic feet)")
        ax.set_ylabel("vehicles")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_passenger_volume_fix.png", dpi=130)
    plt.close(fig)

    lines = [
        "# Data Quality Decision: Impossible Passenger Volumes (Member 3)",
        "",
        "## Problem",
        f"{len(flagged)} vehicle records have a passenger volume between "
        f"{flagged['passenger_volume'].min():.0f} and {flagged['passenger_volume'].max():.0f} "
        f"cubic feet. The smallest believable cars in the data have about "
        f"{smallest_ok['passenger_volume'].min():.0f} cu ft or more, so these values are "
        "data-entry errors, not real vehicles.",
        "",
        "| year | make | model | VClass | passenger volume (cu ft) | comb08 | split |",
        "|---|---|---|---|---:|---:|---|",
        *[f"| {r.year} | {r.make} | {r.model} | {r.VClass} | {r.passenger_volume:.0f} "
          f"| {r.comb08} | {r.split} |" for r in flagged.itertuples()],
        "",
        "## Options considered",
        "1. **Keep them**: the model would learn from impossible values.",
        "2. **Delete the rows**: loses valid information (engine, year, fuel type, target) "
        "and changes the train/test split, so results would no longer be comparable.",
        "3. **Treat the value as missing (chosen)**: consistent with how the dataset's own "
        "'0 = not reported' code is already handled.",
        "",
        "## Decision",
        f"Passenger volumes below **{limit} cubic feet** are set to missing by "
        "`fix_passenger_volume()` in `src/member3_data_quality.py`. `src/preprocessing.py` "
        "is not modified. The model pipeline then fills these values with the "
        "**training-set median** and marks them with the missing-value indicator.",
        "",
        "## Why this is safe (no data leakage)",
        "- It is a fixed rule written by hand; nothing is learned from the data.",
        "- The row count is unchanged, so the 80/20 split (`random_state = 42`) is identical.",
        "- Only the flagged values change (checked automatically).",
        "",
        "## Evidence",
        "- `results/member3_passenger_volume_issues.csv`: the affected records",
        "- `reports/figures/06_passenger_volume_fix.png`: distribution before / after",
        "- `results/member3_model_metrics.csv`: all models trained on the corrected data",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for path in [RESULTS_DIR / "member3_passenger_volume_issues.csv",
                 FIGURES_DIR / "06_passenger_volume_fix.png", REPORT_PATH]:
        print(f"Saved {path.relative_to(prep.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
