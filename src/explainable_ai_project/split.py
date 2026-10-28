"""
Training pathway, block 1A, step 9: split 70 / 15 / 15.

Splits each dataset into three parts that never share an applicant:
    train        70 %   the models learn from this
    validation   15 %   used to tune settings and to calibrate probabilities
    test         15 %   used ONCE at the end, to report final results

The split is stratified by approval AND age group together. That keeps:
    - the same approval rate in all three parts
    - the same share of each age group in all three parts, so the small
      61+ group still has enough test rows for the fairness checks

The seed is fixed, so everyone on the team gets exactly the same split.

Put this file in:  src/explainable_ai_project/split.py
Run it with:       uv run python -m explainable_ai_project.split

Writes:
    datasets/processed/splits_dataset_1.csv   applicant ID -> train / validation / test
    datasets/processed/splits_dataset_2.csv
    results/tables/split_summary.csv          rows and approval rate per part
    results/tables/split_age_groups.csv       rows per age group per part
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from explainable_ai_project.add_features import AGE_GROUP_COLUMN
from explainable_ai_project.add_features import AGE_GROUP_LABELS
from explainable_ai_project.add_features import add_loan_to_income_and_age_groups
from explainable_ai_project.cleaning import check_ranges_and_clean
from explainable_ai_project.common_columns import ID_COLUMN
from explainable_ai_project.common_columns import LABEL_COLUMN
from explainable_ai_project.common_columns import keep_common_features
from explainable_ai_project.harmonize import harmonise_labels_and_units
from explainable_ai_project.load import load_both_raw_datasets


# ------------------------------------------------------------------
# Where the files go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

PROCESSED_FOLDER = PROJECT_ROOT / "datasets" / "processed"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
SEED = 42

TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15
TEST_SHARE = 0.15

SPLIT_NAMES = ["train", "validation", "test"]

# Test groups smaller than this are flagged: fairness numbers on them are less reliable
SMALL_GROUP_LIMIT = 100


# ------------------------------------------------------------------
# Splitting one dataset
# ------------------------------------------------------------------

def make_strata(data):
    """One label per row combining approval and age group, e.g. '1_31-45'.

    Splitting on this keeps both the approval rate and the age mix equal in every part.
    """

    approval_text = data[LABEL_COLUMN].astype(str)
    age_text = data[AGE_GROUP_COLUMN].astype(str)
    strata = approval_text + "_" + age_text
    return strata


def split_one_dataset(data):
    """Split in two steps:
        1. 70 % train, 30 % rest
        2. the rest in half: 15 % validation, 15 % test
    """

    strata = make_strata(data)
    rest_share = VALIDATION_SHARE + TEST_SHARE

    train_data, rest_data = train_test_split(
        data,
        test_size=rest_share,
        stratify=strata,
        random_state=SEED,
    )

    rest_strata = make_strata(rest_data)
    test_share_of_rest = TEST_SHARE / rest_share

    validation_data, test_data = train_test_split(
        rest_data,
        test_size=test_share_of_rest,
        stratify=rest_strata,
        random_state=SEED,
    )

    splits = {
        "train": train_data.reset_index(drop=True),
        "validation": validation_data.reset_index(drop=True),
        "test": test_data.reset_index(drop=True),
    }
    return splits


def check_no_overlap(splits, dataset_name):
    """Stop with an error if any applicant ends up in two parts."""

    train_ids = set(splits["train"][ID_COLUMN])
    validation_ids = set(splits["validation"][ID_COLUMN])
    test_ids = set(splits["test"][ID_COLUMN])

    in_train_and_validation = train_ids & validation_ids
    in_train_and_test = train_ids & test_ids
    in_validation_and_test = validation_ids & test_ids

    overlap_count = len(in_train_and_validation) + len(in_train_and_test) + len(in_validation_and_test)

    if overlap_count > 0:
        message = dataset_name + ": " + str(overlap_count) + " applicants are in more than one part"
        raise ValueError(message)

    print("  no applicant is in more than one part")


def save_split_ids(splits, dataset_name):
    """Save which applicant went to which part, so every later step uses the same rows."""

    parts = []
    for split_name in SPLIT_NAMES:
        split_data = splits[split_name]

        part = pd.DataFrame()
        part[ID_COLUMN] = split_data[ID_COLUMN]
        part["split"] = split_name
        parts.append(part)

    all_ids = pd.concat(parts, ignore_index=True)

    output_file = PROCESSED_FOLDER / ("splits_" + dataset_name + ".csv")
    all_ids.to_csv(output_file, index=False)
    print("  saved", output_file.relative_to(PROJECT_ROOT))


# ------------------------------------------------------------------
# Summaries for the report
# ------------------------------------------------------------------

def split_summary_rows(splits, dataset_name):
    rows = []
    for split_name in SPLIT_NAMES:
        split_data = splits[split_name]

        row_count = len(split_data)
        approval_rate = split_data[LABEL_COLUMN].mean()
        approval_rate = round(approval_rate, 3)

        rows.append({
            "dataset": dataset_name,
            "split": split_name,
            "rows": row_count,
            "approval_rate": approval_rate,
        })
    return rows


def age_group_rows(splits, dataset_name):
    rows = []
    for split_name in SPLIT_NAMES:
        split_data = splits[split_name]
        counts = split_data[AGE_GROUP_COLUMN].value_counts()

        for group in AGE_GROUP_LABELS:
            count = int(counts.get(group, 0))

            is_test = split_name == "test"
            is_small = count < SMALL_GROUP_LIMIT
            small_test_group = is_test and is_small

            rows.append({
                "dataset": dataset_name,
                "split": split_name,
                "age_group": group,
                "rows": count,
                "small_test_group": small_test_group,
            })
    return rows


# ------------------------------------------------------------------
# Both datasets
# ------------------------------------------------------------------

def split_70_15_15(final_datasets):
    """Run the step on both datasets.

    final_datasets is the dictionary returned by add_loan_to_income_and_age_groups().
    Returns a dictionary of dictionaries:
        {"dataset_1": {"train": table, "validation": table, "test": table}, "dataset_2": {...}}
    """

    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    all_splits = {}
    summary_rows = []
    group_rows = []

    for dataset_name, data in final_datasets.items():
        print(dataset_name)

        splits = split_one_dataset(data)
        check_no_overlap(splits, dataset_name)
        save_split_ids(splits, dataset_name)

        all_splits[dataset_name] = splits

        rows = split_summary_rows(splits, dataset_name)
        summary_rows.extend(rows)

        rows = age_group_rows(splits, dataset_name)
        group_rows.extend(rows)

    print()
    summary_table = pd.DataFrame(summary_rows)
    print(summary_table.to_string(index=False))

    group_table = pd.DataFrame(group_rows)
    test_groups = group_table[group_table["split"] == "test"]
    print()
    print("Age groups in the test part:")
    print(test_groups.to_string(index=False))

    small_groups = test_groups[test_groups["small_test_group"]]
    if len(small_groups) > 0:
        print("WARNING: some test age groups have fewer than", SMALL_GROUP_LIMIT, "rows")

    summary_file = TABLES_FOLDER / "split_summary.csv"
    summary_table.to_csv(summary_file, index=False)
    group_file = TABLES_FOLDER / "split_age_groups.csv"
    group_table.to_csv(group_file, index=False)

    print()
    print("Saved", summary_file.relative_to(PROJECT_ROOT))
    print("Saved", group_file.relative_to(PROJECT_ROOT))

    return all_splits


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
#     split_70_15_15(final)