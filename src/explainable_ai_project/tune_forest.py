"""
Training pathway, block 1B, step 16: tune on validation.

Tries a small grid of Random Forest settings for the monotonic model,
scores each one on the VALIDATION part, and keeps the best.
The test part is still untouched.

Settings tried:
    n_estimators       number of trees                          200, 400
    max_depth          how deep each tree may grow              no limit, 12
    min_samples_leaf   fewest applicants allowed in a leaf      10, 20, 50
    max_features       share of features each split may look at "sqrt", 0.5
That is 2 x 2 x 3 x 2 = 24 runs per dataset.

The best run is the one with the highest validation AUC. If two runs are within
0.001 AUC, the one with the lower Brier score wins (better probabilities matter
for the amount sweep).

Every run is logged. The best settings are saved, and model A is retrained
with them and checked again (steps 15's checks: validation scores and the sweep).
Step 17 uses the same functions for model B.

Put this file in:  src/explainable_ai_project/tune_forest.py

Writes:
    results/tables/tuning_log_dataset_1.csv      every run with its scores
    models/best_settings_dataset_1.json          the chosen settings
    models/rf_monotonic_dataset_1_v1.joblib      (replaced by the tuned model)
"""

import json
from pathlib import Path

import pandas as pd

from explainable_ai_project.load_split import load_split
from explainable_ai_project.logistic_baseline import score_predictions
from explainable_ai_project.monotonic_forest_a import build_monotonic_forest
from explainable_ai_project.monotonic_forest_a import train_and_check


# ------------------------------------------------------------------
# Where the files go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

MODELS_FOLDER = PROJECT_ROOT / "models"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"


# ------------------------------------------------------------------
# The grid
# ------------------------------------------------------------------
TREE_COUNTS = [200, 400]
MAX_DEPTHS = [None, 12]
LEAF_SIZES = [10, 20, 50]
FEATURE_SHARES = ["sqrt", 0.5]

# Runs closer than this in AUC count as a tie
AUC_TIE = 0.001


def all_settings():
    """Every combination of the grid, as a list of dictionaries."""

    settings_list = []

    for tree_count in TREE_COUNTS:
        for max_depth in MAX_DEPTHS:
            for leaf_size in LEAF_SIZES:
                for feature_share in FEATURE_SHARES:
                    settings = {
                        "n_estimators": tree_count,
                        "max_depth": max_depth,
                        "min_samples_leaf": leaf_size,
                        "max_features": feature_share,
                    }
                    settings_list.append(settings)

    return settings_list


# ------------------------------------------------------------------
# Running the grid
# ------------------------------------------------------------------

def run_grid(dataset_name):
    """Train and score one monotonic forest per combination. Returns the log table."""

    train = load_split(dataset_name, "train")
    validation = load_split(dataset_name, "validation")
    feature_names = list(train["features"].columns)

    settings_list = all_settings()
    run_count = len(settings_list)
    log_rows = []

    for run_number in range(run_count):
        settings = settings_list[run_number]

        model = build_monotonic_forest(feature_names, settings)
        model.fit(train["features"], train["labels"])

        probabilities = model.predict_proba(validation["features"])
        approval_probabilities = probabilities[:, 1]
        scores = score_predictions(validation["labels"], approval_probabilities)

        log_row = {"run": run_number + 1}
        for setting_name, value in settings.items():
            log_row[setting_name] = value
        for metric_name, value in scores.items():
            log_row[metric_name] = value
        log_rows.append(log_row)

        progress = "  run " + str(run_number + 1) + " of " + str(run_count) + ":"
        print(progress, settings, "-> AUC", scores["auc"], "Brier", scores["brier"])

    log_table = pd.DataFrame(log_rows)
    return log_table


def pick_best_run(log_table):
    """Highest AUC; among runs within AUC_TIE of the best, the lowest Brier score."""

    best_auc = log_table["auc"].max()
    lowest_allowed_auc = best_auc - AUC_TIE

    is_near_best = log_table["auc"] >= lowest_allowed_auc
    near_best_runs = log_table[is_near_best]

    sorted_runs = near_best_runs.sort_values("brier", ascending=True)
    best_run = sorted_runs.iloc[0]
    return best_run


def settings_from_run(best_run):
    """Turn a row of the log back into a settings dictionary."""

    max_depth = best_run["max_depth"]
    # pandas stores "no limit" (None) as a missing value
    if pd.isna(max_depth):
        max_depth = None
    else:
        max_depth = int(max_depth)

    max_features = best_run["max_features"]
    if max_features != "sqrt":
        max_features = float(max_features)

    settings = {
        "n_estimators": int(best_run["n_estimators"]),
        "max_depth": max_depth,
        "min_samples_leaf": int(best_run["min_samples_leaf"]),
        "max_features": max_features,
    }
    return settings


def best_settings_file(dataset_name):
    return MODELS_FOLDER / ("best_settings_" + dataset_name + ".json")


def load_best_settings(dataset_name):
    """Read the settings chosen by tuning (used by later steps)."""

    settings_file = best_settings_file(dataset_name)
    with open(settings_file, "r", encoding="utf-8") as file:
        settings = json.load(file)
    return settings


# ------------------------------------------------------------------
# Tuning one dataset (used for A here and for B in step 17)
# ------------------------------------------------------------------

def tune_monotonic_forest(dataset_name):
    """Run the grid, log every run, save the best settings. Returns them."""

    print(dataset_name + ": tuning", len(all_settings()), "settings on the validation part")

    log_table = run_grid(dataset_name)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    log_file = TABLES_FOLDER / ("tuning_log_" + dataset_name + ".csv")
    log_table.to_csv(log_file, index=False)
    print("  saved", log_file.relative_to(PROJECT_ROOT))

    best_run = pick_best_run(log_table)
    best_settings = settings_from_run(best_run)

    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    settings_file = best_settings_file(dataset_name)
    with open(settings_file, "w", encoding="utf-8") as file:
        json.dump(best_settings, file, indent=2)

    print("  best settings:", best_settings)
    print("  best validation AUC:", best_run["auc"], "Brier:", best_run["brier"])
    print("  saved", settings_file.relative_to(PROJECT_ROOT))
    print()

    return best_settings


def tune_and_retrain_a():
    """Step 16: tune model A, then retrain it with the best settings and check it again."""

    best_settings = tune_monotonic_forest("dataset_1")

    print("Retraining model A with the best settings")
    model_a = train_and_check("dataset_1", best_settings)
    return model_a