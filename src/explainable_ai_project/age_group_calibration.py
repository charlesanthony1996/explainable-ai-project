"""
Training pathway, block 1C, step 26: calibration by age group (Topic 3: calibration and fairness).

Step 19 made the probabilities trustworthy ON AVERAGE. But a model can be well
calibrated overall and still be off for one group, for example telling older
applicants "0.10" when 25 % of them are really approved. Then the amount sweep
would be unfair to that group even if the overall numbers look fine.

For each age group, on the TEST part, with the final calibrated models A and B:
    mean predicted        the average P(approve) the model gives the group
    really approved       the share of the group that was really approved
    calibration gap       mean predicted - really approved
                          (above 0: the model is too optimistic for this group,
                           below 0: too pessimistic)
    Brier score           how close the probabilities are to what happened (lower is better)
    calibration error     average distance between predicted and real approval
                          inside 5 probability bins (0 = perfectly calibrated)

Small groups give unstable numbers (dataset 2 has only 120 test applicants aged 61+),
so every gap and Brier score gets a 95 % interval from a bootstrap:
    draw the group's applicants again at random, with replacement, 1,000 times,
    compute the number each time, and keep the middle 95 % of the results.
A wide interval means: do not read much into this number.

Put this file in:  src/explainable_ai_project/fairness_calibration.py

Writes:
    results/tables/calibration_by_age_group.csv
    results/figures/calibration_by_age_group.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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

SEED = 42
DATASET_NAMES = ["dataset_1", "dataset_2"]
AGE_GROUP_ORDER = ["18-30", "31-45", "46-60", "61+"]

BIN_COUNT = 5            # probability bins (few, because some groups are small)
BOOTSTRAP_ROUNDS = 1000

MODEL_LABELS = {
    "dataset_1": "Model A (dataset 1)",
    "dataset_2": "Model B (dataset 2)",
}
GROUP_COLOURS = {
    "18-30": "#3B6FD8",
    "31-45": "#2F8F5B",
    "46-60": "#D9822B",
    "61+": "#C0392B",
}


# ------------------------------------------------------------------
# Measures for one group
# ------------------------------------------------------------------

def calibration_gap(labels, probabilities):
    mean_predicted = probabilities.mean()
    really_approved = labels.mean()
    gap = mean_predicted - really_approved
    return gap


def calibration_error(labels, probabilities):
    """Average distance between predicted and real approval inside equal-width bins,
    weighted by how many applicants fall into each bin."""

    bin_edges = np.linspace(0, 1, BIN_COUNT + 1)
    bin_numbers = np.digitize(probabilities, bin_edges[1:-1])

    total_error = 0.0
    applicant_count = len(labels)

    for bin_number in range(BIN_COUNT):
        in_bin = bin_numbers == bin_number
        bin_size = in_bin.sum()
        if bin_size == 0:
            continue

        predicted_in_bin = probabilities[in_bin].mean()
        real_in_bin = labels[in_bin].mean()
        distance = abs(predicted_in_bin - real_in_bin)

        bin_weight = bin_size / applicant_count
        total_error = total_error + bin_weight * distance

    return total_error


def bootstrap_interval(labels, probabilities, measure, random_generator):
    """95 % interval of a measure, from 1,000 random redraws of the group."""

    applicant_count = len(labels)
    results = []

    for round_number in range(BOOTSTRAP_ROUNDS):
        picked = random_generator.integers(0, applicant_count, size=applicant_count)
        picked_labels = labels[picked]
        picked_probabilities = probabilities[picked]

        value = measure(picked_labels, picked_probabilities)
        results.append(value)

    lower = np.percentile(results, 2.5)
    upper = np.percentile(results, 97.5)
    return lower, upper


def brier(labels, probabilities):
    value = brier_score_loss(labels, probabilities)
    return value


def measures_for_group(labels, probabilities, random_generator):
    gap = calibration_gap(labels, probabilities)
    gap_lower, gap_upper = bootstrap_interval(labels, probabilities, calibration_gap, random_generator)

    brier_value = brier(labels, probabilities)
    brier_lower, brier_upper = bootstrap_interval(labels, probabilities, brier, random_generator)

    error = calibration_error(labels, probabilities)

    measures = {
        "applicants": len(labels),
        "mean_predicted": round(probabilities.mean(), 3),
        "really_approved": round(labels.mean(), 3),
        "calibration_gap": round(gap, 3),
        "gap_95_lower": round(gap_lower, 3),
        "gap_95_upper": round(gap_upper, 3),
        "brier": round(brier_value, 4),
        "brier_95_lower": round(brier_lower, 4),
        "brier_95_upper": round(brier_upper, 4),
        "calibration_error": round(error, 3),
    }
    return measures


# ------------------------------------------------------------------
# One dataset
# ------------------------------------------------------------------

def check_one_dataset(dataset_name, random_generator):
    print(dataset_name + ":", MODEL_LABELS[dataset_name], "on the test part")

    model = load_calibrated_model(dataset_name)
    test = load_split(dataset_name, "test")

    labels = test["labels"].to_numpy()
    groups = test["rows"]["age_group"].to_numpy()

    probability_table = model.predict_proba(test["features"])
    probabilities = probability_table[:, 1]

    rows = []
    curves = {}

    group_names = list(AGE_GROUP_ORDER)
    group_names.append("all")

    for group in group_names:
        if group == "all":
            in_group = np.ones(len(labels), dtype=bool)
        else:
            in_group = groups == group

        group_labels = labels[in_group]
        group_probabilities = probabilities[in_group]

        measures = measures_for_group(group_labels, group_probabilities, random_generator)

        row = {"dataset": dataset_name, "age_group": group}
        for name, value in measures.items():
            row[name] = value
        rows.append(row)

        if group != "all":
            curves[group] = (group_labels, group_probabilities)

    table = pd.DataFrame(rows)

    columns_to_show = [
        "age_group",
        "applicants",
        "mean_predicted",
        "really_approved",
        "calibration_gap",
        "gap_95_lower",
        "gap_95_upper",
        "brier",
    ]
    print(table[columns_to_show].to_string(index=False))
    print()

    return table, curves


# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

def draw_figure(curves_per_dataset):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for position in range(len(DATASET_NAMES)):
        dataset_name = DATASET_NAMES[position]
        axis = axes[position]
        curves = curves_per_dataset[dataset_name]

        axis.plot([0, 1], [0, 1], linestyle="--", color="#9AA5B1", label="perfectly calibrated")

        for group in AGE_GROUP_ORDER:
            group_labels, group_probabilities = curves[group]

            # Skip groups whose probabilities are all the same (no curve to draw)
            distinct_values = len(np.unique(group_probabilities))
            if distinct_values < 2:
                continue

            observed, predicted = calibration_curve(
                group_labels,
                group_probabilities,
                n_bins=BIN_COUNT,
                strategy="quantile",
            )
            line_label = group + " (" + str(len(group_labels)) + " applicants)"
            axis.plot(predicted, observed, marker="o", color=GROUP_COLOURS[group], label=line_label)

        axis.set_xlabel("Predicted approval probability")
        axis.set_ylabel("Share really approved")
        axis.set_title(MODEL_LABELS[dataset_name] + ", test part")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend(loc="upper left", fontsize=8)

    figure.suptitle("Calibration by age group: are the probabilities equally trustworthy for everyone?")
    figure.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / "calibration_by_age_group.png"
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    print("Saved", figure_file.relative_to(PROJECT_ROOT))


# ------------------------------------------------------------------
# The step
# ------------------------------------------------------------------

def calibration_by_age_group():
    """Step 26: calibration per age group, with bootstrap 95 % intervals."""

    random_generator = np.random.default_rng(SEED)

    tables = []
    curves_per_dataset = {}

    for dataset_name in DATASET_NAMES:
        table, curves = check_one_dataset(dataset_name, random_generator)
        tables.append(table)
        curves_per_dataset[dataset_name] = curves

    result_table = pd.concat(tables, ignore_index=True)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    result_file = TABLES_FOLDER / "calibration_by_age_group.csv"
    result_table.to_csv(result_file, index=False)
    print("Saved", result_file.relative_to(PROJECT_ROOT))

    draw_figure(curves_per_dataset)
    return result_table