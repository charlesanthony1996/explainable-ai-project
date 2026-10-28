"""
Training pathway, block 1C, step 24: approval rates by age group (Topic 3: fairness metrics).

Does the final model treat age groups (18-30, 31-45, 46-60, 61+) differently?
Uses the final calibrated models A and B on their untouched TEST parts.

For each age group:
    actual approval rate     how often the group was really approved (in the data)
    predicted approval rate  how often the model says "approve" (P >= cut-off)
    true positive rate       of the really approved, the share the model also approves
    false positive rate      of the really rejected, the share the model approves anyway
    precision                of the model's approvals, the share that were really approved
    confusion counts         correctly / wrongly approved and rejected

And three gaps between the groups (0 = equal treatment):
    demographic parity       do groups get "approve" equally often?
                             (difference = largest rate - smallest rate,
                              ratio = smallest rate / largest rate; below 0.8 is a common warning sign)
    equal opportunity        are really-approved applicants approved equally often in every group?
                             (difference in true positive rate)
    predictive parity        is an "approve" equally trustworthy in every group?
                             (difference in precision)

Two cut-offs are reported: 0.5 (the usual one) and 0.8 (the one the amount sweep uses).

This step only measures. Whether a gap is explained by income and credit score
is step 25; removing age from the model is step 28.

Put this file in:  src/explainable_ai_project/fairness_age_groups.py

Writes:
    results/tables/fairness_by_age_group.csv     one row per dataset, cut-off and age group
    results/tables/fairness_gaps.csv             one row per dataset and cut-off
    results/figures/fairness_approval_rates.png  actual vs predicted approval per age group
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import demographic_parity_ratio
from fairlearn.metrics import false_negative_rate
from fairlearn.metrics import false_positive_rate
from fairlearn.metrics import selection_rate
from fairlearn.metrics import true_positive_rate
from sklearn.metrics import precision_score

from explainable_ai_project.calibrate import load_calibrated_model
from explainable_ai_project.load_split import load_split


# ------------------------------------------------------------------
# Where the files go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"
FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"

DATASET_NAMES = ["dataset_1", "dataset_2"]
AGE_GROUP_ORDER = ["18-30", "31-45", "46-60", "61+"]
CUT_OFFS = [0.5, 0.8]
FIGURE_CUT_OFF = 0.8

MODEL_LABELS = {
    "dataset_1": "Model A (dataset 1)",
    "dataset_2": "Model B (dataset 2)",
}


# ------------------------------------------------------------------
# Metrics per group
# ------------------------------------------------------------------

def precision_without_warnings(labels, predictions):
    """Precision that returns 0 instead of a warning when a group has no approvals."""
    value = precision_score(labels, predictions, zero_division=0)
    return value


def actual_approval_rate(labels, predictions):
    """Share of the group that was REALLY approved (ignores the predictions)."""
    value = np.mean(labels)
    return value


GROUP_METRICS = {
    "applicants": count,
    "actual_approval_rate": actual_approval_rate,
    "predicted_approval_rate": selection_rate,
    "true_positive_rate": true_positive_rate,
    "false_positive_rate": false_positive_rate,
    "false_negative_rate": false_negative_rate,
    "precision": precision_without_warnings,
}


def confusion_counts_per_group(labels, predictions, groups):
    rows = []
    for group in AGE_GROUP_ORDER:
        in_group = groups == group
        group_labels = labels[in_group]
        group_predictions = predictions[in_group]

        really_approved = group_labels == 1
        really_rejected = group_labels == 0
        predicted_approved = group_predictions == 1
        predicted_rejected = group_predictions == 0

        correctly_approved = really_approved & predicted_approved
        wrongly_approved = really_rejected & predicted_approved
        correctly_rejected = really_rejected & predicted_rejected
        wrongly_rejected = really_approved & predicted_rejected

        rows.append({
            "age_group": group,
            "correctly_approved": int(correctly_approved.sum()),
            "wrongly_approved": int(wrongly_approved.sum()),
            "correctly_rejected": int(correctly_rejected.sum()),
            "wrongly_rejected": int(wrongly_rejected.sum()),
        })

    table = pd.DataFrame(rows)
    return table


def group_table_for(labels, predictions, groups, dataset_name, cut_off):
    """All per-group metrics in one table."""

    metric_frame = MetricFrame(
        metrics=GROUP_METRICS,
        y_true=labels,
        y_pred=predictions,
        sensitive_features=groups,
    )

    by_group = metric_frame.by_group
    by_group = by_group.reindex(AGE_GROUP_ORDER)
    by_group = by_group.reset_index()
    by_group = by_group.rename(columns={by_group.columns[0]: "age_group"})

    counts = confusion_counts_per_group(labels, predictions, groups)
    table = by_group.merge(counts, on="age_group")

    table.insert(0, "cut_off", cut_off)
    table.insert(0, "dataset", dataset_name)

    rate_columns = [
        "actual_approval_rate",
        "predicted_approval_rate",
        "true_positive_rate",
        "false_positive_rate",
        "false_negative_rate",
        "precision",
    ]
    for column in rate_columns:
        table[column] = table[column].round(3)

    table["applicants"] = table["applicants"].astype(int)

    return table, metric_frame


def gaps_for(labels, predictions, groups, metric_frame, dataset_name, cut_off):
    """The three fairness gaps between the age groups."""

    parity_difference = demographic_parity_difference(labels, predictions, sensitive_features=groups)
    parity_ratio = demographic_parity_ratio(labels, predictions, sensitive_features=groups)

    differences = metric_frame.difference(method="between_groups")
    opportunity_difference = differences["true_positive_rate"]
    predictive_parity_difference = differences["precision"]

    rates = metric_frame.by_group["predicted_approval_rate"]
    most_approved_group = rates.idxmax()
    least_approved_group = rates.idxmin()

    gap_row = {
        "dataset": dataset_name,
        "cut_off": cut_off,
        "demographic_parity_difference": round(parity_difference, 3),
        "demographic_parity_ratio": round(parity_ratio, 3),
        "equal_opportunity_difference": round(opportunity_difference, 3),
        "predictive_parity_difference": round(predictive_parity_difference, 3),
        "most_approved_group": most_approved_group,
        "least_approved_group": least_approved_group,
    }
    return gap_row


# ------------------------------------------------------------------
# One dataset
# ------------------------------------------------------------------

def check_one_dataset(dataset_name):
    print(dataset_name + ":", MODEL_LABELS[dataset_name], "on the test part")

    model = load_calibrated_model(dataset_name)
    test = load_split(dataset_name, "test")

    labels = test["labels"].to_numpy()
    groups = test["rows"]["age_group"].to_numpy()

    probability_table = model.predict_proba(test["features"])
    probabilities = probability_table[:, 1]

    group_tables = []
    gap_rows = []

    for cut_off in CUT_OFFS:
        predicted_approved = probabilities >= cut_off
        predictions = predicted_approved.astype(int)

        group_table, metric_frame = group_table_for(labels, predictions, groups, dataset_name, cut_off)
        gap_row = gaps_for(labels, predictions, groups, metric_frame, dataset_name, cut_off)

        group_tables.append(group_table)
        gap_rows.append(gap_row)

        columns_to_show = [
            "age_group",
            "applicants",
            "actual_approval_rate",
            "predicted_approval_rate",
            "true_positive_rate",
            "precision",
        ]
        print("  cut-off", cut_off)
        print(group_table[columns_to_show].to_string(index=False))
        print("  demographic parity ratio:", gap_row["demographic_parity_ratio"],
              "| equal opportunity difference:", gap_row["equal_opportunity_difference"],
              "| predictive parity difference:", gap_row["predictive_parity_difference"])
        print()

    return group_tables, gap_rows


# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

def draw_figure(group_table):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    bar_width = 0.38
    positions = np.arange(len(AGE_GROUP_ORDER))

    for position in range(len(DATASET_NAMES)):
        dataset_name = DATASET_NAMES[position]
        axis = axes[position]

        is_dataset = group_table["dataset"] == dataset_name
        is_cut_off = group_table["cut_off"] == FIGURE_CUT_OFF
        rows = group_table[is_dataset & is_cut_off]

        actual_rates = rows["actual_approval_rate"].to_numpy()
        predicted_rates = rows["predicted_approval_rate"].to_numpy()

        axis.bar(positions - bar_width / 2, actual_rates, bar_width, color="#9AA5B1", label="really approved")
        axis.bar(positions + bar_width / 2, predicted_rates, bar_width, color="#D9822B",
                 label="model approves (P >= " + str(FIGURE_CUT_OFF) + ")")

        axis.set_xticks(positions)
        axis.set_xticklabels(AGE_GROUP_ORDER)
        axis.set_xlabel("Age group")
        axis.set_title(MODEL_LABELS[dataset_name] + ", test part")
        axis.set_ylim(0, 1)
        axis.legend(loc="upper right")

    axes[0].set_ylabel("Approval rate")
    figure.suptitle("Approval rates by age group: in the data and from the final model")
    figure.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / "fairness_approval_rates.png"
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    print("Saved", figure_file.relative_to(PROJECT_ROOT))


# ------------------------------------------------------------------
# The step
# ------------------------------------------------------------------

def approval_rates_by_age_group():
    """Step 24: fairness metrics per age group for models A and B."""

    all_group_tables = []
    all_gap_rows = []

    for dataset_name in DATASET_NAMES:
        group_tables, gap_rows = check_one_dataset(dataset_name)
        all_group_tables.extend(group_tables)
        all_gap_rows.extend(gap_rows)

    group_table = pd.concat(all_group_tables, ignore_index=True)
    gap_table = pd.DataFrame(all_gap_rows)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    group_file = TABLES_FOLDER / "fairness_by_age_group.csv"
    gap_file = TABLES_FOLDER / "fairness_gaps.csv"
    group_table.to_csv(group_file, index=False)
    gap_table.to_csv(gap_file, index=False)

    print("Fairness gaps:")
    print(gap_table.to_string(index=False))
    print()
    print("Saved", group_file.relative_to(PROJECT_ROOT))
    print("Saved", gap_file.relative_to(PROJECT_ROOT))

    draw_figure(group_table)
    return group_table, gap_table