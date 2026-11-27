# Data Quality Decision: Impossible Passenger Volumes (Member 3)

## Problem
9 vehicle records have a passenger volume between 1 and 28 cubic feet. The smallest believable cars in the data have about 40 cu ft or more, so these values are data-entry errors, not real vehicles.

| year | make | model | VClass | passenger volume (cu ft) | comb08 | split |
|---|---|---|---|---:|---:|---|
| 1984 | Bitter Gmbh and Co. Kg | SC | Midsize Cars | 1 | 13 | train |
| 1984 | Bitter Gmbh and Co. Kg | SC | Midsize Cars | 1 | 14 | train |
| 1984 | Import Foreign Auto Sales Inc | 1fas 410 | Small Pickup Trucks 4WD | 28 | 19 | train |
| 1991 | Mercedes-Benz | 190E | Compact Cars | 4 | 17 | test |
| 1991 | Mercedes-Benz | 260E | Midsize Cars | 5 | 16 | test |
| 1991 | Mercedes-Benz | 300E | Midsize Cars | 5 | 15 | train |
| 1991 | Porsche | 928 S4 | Midsize Cars | 2 | 15 | train |
| 1991 | Mercedes-Benz | 230E | Midsize Cars | 5 | 17 | train |
| 1991 | Mercedes-Benz | 300SEL | Large Cars | 6 | 15 | train |

## Options considered
1. **Keep them**: the model would learn from impossible values.
2. **Delete the rows**: loses valid information (engine, year, fuel type, target) and changes the train/test split, so results would no longer be comparable.
3. **Treat the value as missing (chosen)**: consistent with how the dataset's own '0 = not reported' code is already handled.

## Decision
Passenger volumes below **40 cubic feet** are set to missing by `fix_passenger_volume()` in `src/member3_data_quality.py`. `src/preprocessing.py` is not modified. The model pipeline then fills these values with the **training-set median** and marks them with the missing-value indicator.

## Why this is safe (no data leakage)
- It is a fixed rule written by hand; nothing is learned from the data.
- The row count is unchanged, so the 80/20 split (`random_state = 42`) is identical.
- Only the flagged values change (checked automatically).

## Evidence
- `results/member3_passenger_volume_issues.csv`: the affected records
- `reports/figures/06_passenger_volume_fix.png`: distribution before / after
- `results/member3_model_metrics.csv`: all models trained on the corrected data
