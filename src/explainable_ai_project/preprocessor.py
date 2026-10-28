"""
Training pathway, block 1A, step 10: fit the preprocessor.

Models need numbers only. The preprocessor turns each applicant into a row of numbers:
    - the 7 number columns pass through unchanged
      (Random Forests do not need scaling)
    - the 4 category columns are one-hot encoded: "education = Postgraduate"
      becomes a column education_Postgraduate with 1 or 0

It is fitted on the TRAINING part only, so nothing from validation or test
leaks into it. At inference time the same saved preprocessor is loaded,
so a new applicant is turned into numbers in exactly the same way.

age_group is NOT a model input: it is only used for the fairness checks.

Put this file in:  src/explainable_ai_project/preprocessor.py
Run it with:       uv run python -m explainable_ai_project.preprocessor

Writes:
    models/preprocessor_dataset_1_v1.joblib
    models/preprocessor_dataset_2_v1.joblib
    models/feature_names_dataset_1_v1.json    the model input columns, in order
    models/feature_names_dataset_2_v1.json    (needed for the monotonic constraint)
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from explainable_ai_project.add_features import LOAN_TO_INCOME_COLUMN
from explainable_ai_project.add_features import add_loan_to_income_and_age_groups
from explainable_ai_project.cleaning import check_ranges_and_clean
from explainable_ai_project.common_columns import keep_common_features
from explainable_ai_project.harmonize import harmonise_labels_and_units
from explainable_ai_project.load import load_both_raw_datasets
from explainable_ai_project.split import split_70_15_15


# ------------------------------------------------------------------
# Where the files go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

MODELS_FOLDER = PROJECT_ROOT / "models"

VERSION = "v1"


# ------------------------------------------------------------------
# The model inputs
# ------------------------------------------------------------------
NUMBER_INPUTS = [
    "age",
    "dependents",
    "annual_income",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    LOAN_TO_INCOME_COLUMN,
]

CATEGORY_INPUTS = [
    "marital_status",
    "education",
    "employment_status",
    "loan_purpose",
]


def preprocessor_file(dataset_name):
    file_name = "preprocessor_" + dataset_name + "_" + VERSION + ".joblib"
    return MODELS_FOLDER / file_name


def feature_names_file(dataset_name):
    file_name = "feature_names_" + dataset_name + "_" + VERSION + ".json"
    return MODELS_FOLDER / file_name


def build_preprocessor():
    """An unfitted preprocessor: numbers pass through, categories are one-hot encoded."""

    # handle_unknown="ignore": a category never seen in training becomes all zeros
    # instead of crashing (for example at inference time)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numbers", "passthrough", NUMBER_INPUTS),
            ("categories", encoder, CATEGORY_INPUTS),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor


def check_transformed_split(preprocessor, split_data, split_name, feature_count):
    """Transform one split and check the result has the right width and no missing values."""

    transformed = preprocessor.transform(split_data)

    row_count = transformed.shape[0]
    column_count = transformed.shape[1]

    if column_count != feature_count:
        message = split_name + ": expected " + str(feature_count) + " columns, got " + str(column_count)
        raise ValueError(message)

    missing_flags = np.isnan(transformed)
    missing_count = int(missing_flags.sum())
    if missing_count > 0:
        message = split_name + ": " + str(missing_count) + " missing values after transforming"
        raise ValueError(message)

    print("  " + split_name + ":", row_count, "rows x", column_count, "features, no missing values")


def fit_one_preprocessor(splits, dataset_name):
    """Fit on train, check on all three parts, save the preprocessor and its feature names."""

    print(dataset_name)

    train_data = splits["train"]

    preprocessor = build_preprocessor()
    preprocessor.fit(train_data)

    feature_names = list(preprocessor.get_feature_names_out())
    feature_count = len(feature_names)
    print("  fitted on", len(train_data), "training rows ->", feature_count, "features")

    for split_name in ["train", "validation", "test"]:
        check_transformed_split(preprocessor, splits[split_name], split_name, feature_count)

    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

    model_file = preprocessor_file(dataset_name)
    joblib.dump(preprocessor, model_file)

    names_file = feature_names_file(dataset_name)
    with open(names_file, "w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=2)

    print("  saved", model_file.relative_to(PROJECT_ROOT))
    print("  saved", names_file.relative_to(PROJECT_ROOT))
    print("  features:", feature_names)
    print()

    return preprocessor, feature_names


def fit_preprocessors(all_splits):
    """Run the step on both datasets.

    all_splits is the dictionary returned by split_70_15_15().
    Returns a dictionary: {"dataset_1": preprocessor, "dataset_2": preprocessor}
    """

    preprocessors = {}

    for dataset_name, splits in all_splits.items():
        preprocessor, feature_names = fit_one_preprocessor(splits, dataset_name)
        preprocessors[dataset_name] = preprocessor

    return preprocessors


def load_preprocessor(dataset_name):
    """Load a saved preprocessor (used by the model steps and at inference)."""

    model_file = preprocessor_file(dataset_name)
    preprocessor = joblib.load(model_file)
    return preprocessor


# if __name__ == "__main__":
#     loaded_datasets = load_both_raw_datasets()
#     print()
#     common = keep_common_features(loaded_datasets)
#     print()
#     harmonised = harmonise_labels_and_units(common)
#     print()
#     cleaned = check_ranges_and_clean(harmonised)
#     print()
#     final = add_loan_to_income_and_age_groups(cleaned)
#     print()
#     splits_per_dataset = split_70_15_15(final)
#     print()
#     fit_preprocessors(splits_per_dataset)