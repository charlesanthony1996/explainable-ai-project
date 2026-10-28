"""
Training pathway, block 1B, step 12: load a split ready for training.

Every model step needs the same three things for one dataset and one part
(train, validation or test):

    features   the model inputs as numbers, one column per feature,
               made by the saved preprocessor from step 10
    labels     approved (1) or not (0)
    rows       the original rows (ID, age, age group, loan amount, ...),
               kept for the fairness checks and the amount sweep

This step reads them from disk, so 1B can run without repeating 1A:
    datasets/processed/dataset_X_clean.csv     from step 8
    datasets/processed/splits_dataset_X.csv    from step 9
    models/preprocessor_dataset_X_v1.joblib    from step 10

Put this file in:  src/explainable_ai_project/load_split.py
Run it with:       uv run python -m explainable_ai_project.load_split
"""

import json
from pathlib import Path

import joblib
import pandas as pd


# ------------------------------------------------------------------
# Where the files are
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

PROCESSED_FOLDER = PROJECT_ROOT / "datasets" / "processed"
MODELS_FOLDER = PROJECT_ROOT / "models"

VERSION = "v1"

ID_COLUMN = "applicant_id"
LABEL_COLUMN = "approved"

DATASET_NAMES = ["dataset_1", "dataset_2"]
SPLIT_NAMES = ["train", "validation", "test"]


# ------------------------------------------------------------------
# Reading the saved files
# ------------------------------------------------------------------

def check_file_exists(file_path, step_name):
    file_exists = file_path.exists()
    if not file_exists:
        relative_path = file_path.relative_to(PROJECT_ROOT)
        message = "Missing " + str(relative_path) + ". Run " + step_name + " first."
        raise FileNotFoundError(message)


def read_clean_data(dataset_name):
    file_path = PROCESSED_FOLDER / (dataset_name + "_clean.csv")
    check_file_exists(file_path, "step 8 (add loan-to-income and age groups)")
    clean_data = pd.read_csv(file_path)
    return clean_data


def read_split_ids(dataset_name):
    file_path = PROCESSED_FOLDER / ("splits_" + dataset_name + ".csv")
    check_file_exists(file_path, "step 9 (split 70 / 15 / 15)")
    split_ids = pd.read_csv(file_path)
    return split_ids


def read_preprocessor(dataset_name):
    file_path = MODELS_FOLDER / ("preprocessor_" + dataset_name + "_" + VERSION + ".joblib")
    check_file_exists(file_path, "step 10 (fit the preprocessor)")
    preprocessor = joblib.load(file_path)
    return preprocessor


def read_feature_names(dataset_name):
    file_path = MODELS_FOLDER / ("feature_names_" + dataset_name + "_" + VERSION + ".json")
    check_file_exists(file_path, "step 10 (fit the preprocessor)")
    with open(file_path, "r", encoding="utf-8") as file:
        feature_names = json.load(file)
    return feature_names


# ------------------------------------------------------------------
# One split
# ------------------------------------------------------------------

def select_split_rows(clean_data, split_ids, split_name, dataset_name):
    """Keep the rows of one split, in the order of the split file."""

    is_this_split = split_ids["split"] == split_name
    ids_of_this_split = split_ids[is_this_split]

    if len(ids_of_this_split) == 0:
        message = dataset_name + " has no rows in a split called " + split_name
        raise ValueError(message)

    # Merging on the ID keeps exactly the rows listed in the split file
    rows = ids_of_this_split.merge(clean_data, on=ID_COLUMN, how="inner")
    rows = rows.drop(columns=["split"])

    expected_count = len(ids_of_this_split)
    actual_count = len(rows)
    if actual_count != expected_count:
        message = f"{dataset_name} {split_name}: the split file lists {expected_count} IDs, "
        message = message + f"but only {actual_count} were found in the clean data. Run steps 8 to 10 again."
        raise ValueError(message)

    return rows


def make_features(rows, preprocessor, feature_names):
    """Turn rows into model inputs, with one named column per feature."""

    feature_values = preprocessor.transform(rows)

    # The preprocessor must give the same columns, in the same order, as step 10 saved
    preprocessor_names = list(preprocessor.get_feature_names_out())
    if preprocessor_names != feature_names:
        raise ValueError("The saved preprocessor and feature names do not match. Run step 10 again.")

    features = pd.DataFrame(feature_values, columns=feature_names, index=rows.index)
    return features


def load_split(dataset_name, split_name):
    """Load one part of one dataset, ready for a model.

    Example:
        train = load_split("dataset_1", "train")
        model.fit(train["features"], train["labels"])

    Returns a dictionary with:
        "features"   model inputs (a table with named columns)
        "labels"     approved, 1 or 0
        "rows"       the original rows, for fairness checks and the amount sweep
    """

    clean_data = read_clean_data(dataset_name)
    split_ids = read_split_ids(dataset_name)
    preprocessor = read_preprocessor(dataset_name)
    feature_names = read_feature_names(dataset_name)

    rows = select_split_rows(clean_data, split_ids, split_name, dataset_name)
    features = make_features(rows, preprocessor, feature_names)
    labels = rows[LABEL_COLUMN]

    split = {
        "features": features,
        "labels": labels,
        "rows": rows,
    }
    return split


def load_all_splits(dataset_name):
    """Load train, validation and test of one dataset.

    Returns a dictionary: {"train": split, "validation": split, "test": split}
    """

    all_splits = {}
    for split_name in SPLIT_NAMES:
        split = load_split(dataset_name, split_name)
        all_splits[split_name] = split
    return all_splits


# ------------------------------------------------------------------
# Check that everything loads
# ------------------------------------------------------------------

def check_all_splits_load():
    """Load every split of both datasets and print their sizes."""

    for dataset_name in DATASET_NAMES:
        print(dataset_name)
        all_splits = load_all_splits(dataset_name)

        for split_name in SPLIT_NAMES:
            split = all_splits[split_name]
            row_count = split["features"].shape[0]
            feature_count = split["features"].shape[1]

            approval_rate = split["labels"].mean()
            approval_rate = round(approval_rate, 3)

            print("  " + split_name + ":", row_count, "rows x", feature_count, "features,",
                  "approval rate", approval_rate)
        print()


# if __name__ == "__main__":
#     check_all_splits_load()