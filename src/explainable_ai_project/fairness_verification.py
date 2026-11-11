"""Fairness verification with the definitions and intervals written down.

WHY THIS EXISTS ALONGSIDE STEP 24
---------------------------------
Step 24 (`age_approval.py`) computes the parity metrics and is untouched.
What it does not do is state what those numbers are being compared against,
or how much of each gap is sampling noise. A demographic parity ratio of
0.429 on 120 applicants and the same ratio on 12,000 are very different
claims, and a table of point estimates cannot tell them apart.

This module adds the four things a fairness claim needs before it can be
defended, and nothing else:

  1. The definitions, stated as constants rather than left implicit:
     protected attribute, favourable outcome, reference population,
     reference group, and the tolerance each metric is held to.
  2. Sample counts per group, including the ones that are uncomfortably
     small.
  3. A bootstrap confidence interval for every metric.
  4. A verdict per metric against the declared tolerance, with the interval
     deciding - not the point estimate.

WHERE THE TOLERANCES COME FROM
------------------------------
The 0.8 on the parity ratio is the four-fifths rule, a convention from US
employment law (EEOC, 29 CFR 1607.4D). It has no standing in EU credit law
and is used here as a widely recognised warning line, not as a legal test.

The 0.10 on the other three is a PROJECT DECISION with no external
authority. It is written down so that it can be argued with. A tolerance
nobody can see is not a tolerance, it is a result chosen afterwards.

WHAT THE INTERVAL IS AND IS NOT
-------------------------------
The test rows are resampled with replacement 2000 times and the metric is
recomputed each time; the interval is the 2.5th to 97.5th percentile of
those values. It covers sampling variation in the test split and nothing
else. It does not cover the choice of model, the split seed, or the fact
that both datasets are synthetic. Those are limitations, not intervals.

Resampling is not stratified by group, so group sizes vary between
replicates exactly as they would in a fresh sample. For the 61+ group in
dataset 2, n = 120, that variation is a large part of the width - which is
the point of showing it.

Usage
-----
    uv run main.py verify_fairness
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from explainable_ai_project.calibrate import load_calibrated_model
from explainable_ai_project.load_split import load_split


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"

PARITY_FILE = TABLES_FOLDER / "parity_table_with_intervals.csv"
GROUP_FILE = TABLES_FOLDER / "parity_by_group_with_intervals.csv"
DEFINITIONS_FILE = TABLES_FOLDER / "fairness_definitions.json"


# ------------------------------------------------------------------
# The definitions. Everything a reader needs in order to disagree.
# ------------------------------------------------------------------

PROTECTED_ATTRIBUTE = "age_group"
GROUPS = ["18-30", "31-45", "46-60", "61+"]

FAVOURABLE_OUTCOME = (
    "the model predicts approval, that is the calibrated probability of "
    "approval at the requested amount reaches the cut-off"
)

REFERENCE_POPULATION = (
    "the held-out test split of the dataset in question, 15 percent of the "
    "rows, never used for fitting, tuning, calibration or rule selection"
)

# Every ratio is taken against the group the model treats best, because that
# is the comparison the four-fifths rule is defined on. Naming the group in
# the output matters: it is not fixed in advance and it can differ between
# datasets and cut-offs.
REFERENCE_GROUP_RULE = "the age group with the highest predicted approval rate"

DATASET_NAMES = ["dataset_1", "dataset_2"]
CUT_OFFS = [0.5, 0.8]

BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 0.95
SEED = 42

TOLERANCES = {
    "demographic_parity_ratio": {
        "limit": 0.8, "direction": "at least",
        "source": "four-fifths rule, EEOC 29 CFR 1607.4D - a convention from "
                  "US employment law, used here as a warning line, not as a "
                  "legal test under EU credit law",
    },
    "demographic_parity_difference": {
        "limit": 0.10, "direction": "at most",
        "source": "project decision, no external authority",
    },
    "equal_opportunity_difference": {
        "limit": 0.10, "direction": "at most",
        "source": "project decision, no external authority",
    },
    "predictive_parity_difference": {
        "limit": 0.10, "direction": "at most",
        "source": "project decision, no external authority",
    },
}


# ------------------------------------------------------------------
# The metrics, computed from counts so a bootstrap replicate is cheap
# ------------------------------------------------------------------

def group_rates(codes, labels, predictions, how_many_groups):
    """Per-group rates from one sample, as plain arrays.

    Returns nan where a rate is undefined - no applicant of that group in
    the replicate, nobody actually approved, nobody predicted approved -
    rather than a zero, because a missing rate and a rate of zero are
    different facts and averaging them together would hide that.
    """

    counts = np.bincount(codes, minlength=how_many_groups).astype(float)
    predicted_positive = np.bincount(codes, weights=predictions,
                                     minlength=how_many_groups)
    actual_positive = np.bincount(codes, weights=labels,
                                  minlength=how_many_groups)
    true_positive = np.bincount(codes, weights=predictions * labels,
                                minlength=how_many_groups)

    with np.errstate(divide="ignore", invalid="ignore"):
        predicted_rate = np.where(counts > 0, predicted_positive / counts, np.nan)
        actual_rate = np.where(counts > 0, actual_positive / counts, np.nan)
        true_positive_rate = np.where(actual_positive > 0,
                                      true_positive / actual_positive, np.nan)
        precision = np.where(predicted_positive > 0,
                             true_positive / predicted_positive, np.nan)

    return {
        "applicants": counts,
        "actual_approval_rate": actual_rate,
        "predicted_approval_rate": predicted_rate,
        "true_positive_rate": true_positive_rate,
        "precision": precision,
    }


def spread(values):
    """max - min and min / max over the groups where the value is DEFINED.

    Undefined is not zero, and the difference matters here. Model A predicts
    approval for nobody aged 61 or over, so its precision in that group does
    not exist - there are no approvals to be precise about. Step 24 records
    it as 0.000 (scikit-learn's zero_division=0) and the predictive parity
    difference then reads 0.847, which says the model is wildly less precise
    for older applicants. It is not: a precision of 0 would mean every
    approval it granted them was wrong, and it granted none.

    Dropping the undefined group gives 0.005 across the three groups where
    precision exists, and that is the number this project reports. It also
    sharpens the point rather than softening it: precision is near-identical
    wherever it can be measured, while the approval rates are not, which is
    the impossibility result in one line.
    """

    usable = values[~np.isnan(values)]
    if len(usable) < 2:
        return np.nan, np.nan, int(len(usable))
    highest = usable.max()
    lowest = usable.min()
    ratio = lowest / highest if highest > 0 else 0.0
    return highest - lowest, ratio, int(len(usable))


def metrics_from(rates, with_counts=False):
    difference, ratio, used_rate = spread(rates["predicted_approval_rate"])
    opportunity, _, used_tpr = spread(rates["true_positive_rate"])
    predictive, _, used_precision = spread(rates["precision"])

    values = {
        "demographic_parity_difference": difference,
        "demographic_parity_ratio": ratio,
        "equal_opportunity_difference": opportunity,
        "predictive_parity_difference": predictive,
    }
    if not with_counts:
        return values

    return values, {
        "demographic_parity_difference": used_rate,
        "demographic_parity_ratio": used_rate,
        "equal_opportunity_difference": used_tpr,
        "predictive_parity_difference": used_precision,
    }


# ------------------------------------------------------------------
# One dataset, one cut-off
# ------------------------------------------------------------------

def verify_one(dataset_name, cut_off, probabilities, labels, codes,
               how_many_groups, rng):
    predictions = (probabilities >= cut_off).astype(float)

    observed_rates = group_rates(codes, labels, predictions, how_many_groups)
    observed, groups_used = metrics_from(observed_rates, with_counts=True)

    # Which group the ratios are taken against, named rather than assumed.
    predicted_rate = observed_rates["predicted_approval_rate"]
    if np.all(np.isnan(predicted_rate)):
        reference_group = None
    else:
        reference_group = GROUPS[int(np.nanargmax(predicted_rate))]

    total = len(labels)
    replicate_metrics = {name: np.empty(BOOTSTRAP_SAMPLES) for name in observed}
    replicate_rates = {
        name: np.empty((BOOTSTRAP_SAMPLES, how_many_groups))
        for name in observed_rates if name != "applicants"
    }
    degenerate = 0

    for step in range(BOOTSTRAP_SAMPLES):
        picked = rng.integers(0, total, total)
        rates = group_rates(codes[picked], labels[picked],
                            predictions[picked], how_many_groups)
        values = metrics_from(rates)
        for name in observed:
            replicate_metrics[name][step] = values[name]
        for name in replicate_rates:
            replicate_rates[name][step] = rates[name]
        if np.isnan(list(values.values())).any():
            degenerate += 1

    low = 100 * (1 - CONFIDENCE) / 2
    high = 100 - low

    rows = []
    for name, value in observed.items():
        samples = replicate_metrics[name]
        usable = samples[~np.isnan(samples)]
        tolerance = TOLERANCES[name]

        if len(usable) == 0:
            lower = upper = np.nan
            verdict = "not computable"
        else:
            lower = float(np.percentile(usable, low))
            upper = float(np.percentile(usable, high))
            # The interval decides, not the point estimate: a gap is only
            # called acceptable when the whole interval sits inside the
            # tolerance, and only called a breach when all of it sits
            # outside. Anything else is undecided at this sample size.
            if tolerance["direction"] == "at least":
                inside, outside = lower >= tolerance["limit"], upper < tolerance["limit"]
            else:
                inside, outside = upper <= tolerance["limit"], lower > tolerance["limit"]
            verdict = ("within tolerance" if inside
                       else "breaches tolerance" if outside
                       else "undecided at this sample size")

        rows.append({
            "dataset": dataset_name,
            "cut_off": cut_off,
            "metric": name,
            "value": round(float(value), 4) if not np.isnan(value) else None,
            "ci_low": round(lower, 4) if not np.isnan(lower) else None,
            "ci_high": round(upper, 4) if not np.isnan(upper) else None,
            "ci_width": round(upper - lower, 4) if not np.isnan(lower) else None,
            "tolerance": str(tolerance["direction"]) + " " + str(tolerance["limit"]),
            "verdict": verdict,
            "reference_group": reference_group,
            # How many of the four groups the metric could be computed on.
            # Fewer than four means a group had no defined value and was left
            # out rather than counted as zero.
            "groups_used": groups_used[name],
            "replicates_usable": int(len(usable)),
        })

    group_rows = []
    for place, group in enumerate(GROUPS[:how_many_groups]):
        row = {
            "dataset": dataset_name,
            "cut_off": cut_off,
            "age_group": group,
            "applicants": int(observed_rates["applicants"][place]),
            "is_reference_group": group == reference_group,
        }
        for name in replicate_rates:
            samples = replicate_rates[name][:, place]
            usable = samples[~np.isnan(samples)]
            observed_value = observed_rates[name][place]
            row[name] = (round(float(observed_value), 4)
                         if not np.isnan(observed_value) else None)
            row[name + "_ci_low"] = (round(float(np.percentile(usable, low)), 4)
                                     if len(usable) else None)
            row[name + "_ci_high"] = (round(float(np.percentile(usable, high)), 4)
                                      if len(usable) else None)
        group_rows.append(row)

    return rows, group_rows, degenerate


# ------------------------------------------------------------------

def run_fairness_verification():
    print("Fairness verification with bootstrap intervals")
    print("  protected attribute :", PROTECTED_ATTRIBUTE)
    print("  favourable outcome  :", FAVOURABLE_OUTCOME)
    print("  reference population:", REFERENCE_POPULATION)
    print("  reference group     :", REFERENCE_GROUP_RULE)
    print("  replicates          :", BOOTSTRAP_SAMPLES,
          "| confidence:", CONFIDENCE, "| seed:", SEED)
    print()

    rng = np.random.default_rng(SEED)
    all_rows = []
    all_group_rows = []

    for dataset_name in DATASET_NAMES:
        model = load_calibrated_model(dataset_name)
        test = load_split(dataset_name, "test")

        labels = test["labels"].to_numpy().astype(float)
        groups = test["rows"]["age_group"].to_numpy()
        codes = np.array([GROUPS.index(one) for one in groups])
        probabilities = model.predict_proba(test["features"])[:, 1]

        print(dataset_name, "- test rows:", len(labels))
        for place, group in enumerate(GROUPS):
            print("   ", group.ljust(6), int((codes == place).sum()))

        for cut_off in CUT_OFFS:
            rows, group_rows, degenerate = verify_one(
                dataset_name, cut_off, probabilities, labels, codes,
                len(GROUPS), rng)
            all_rows.extend(rows)
            all_group_rows.extend(group_rows)
            if degenerate:
                print("    cut-off", cut_off, "-", degenerate, "of",
                      BOOTSTRAP_SAMPLES,
                      "replicates had an undefined rate in some group")
        print()

    parity = pd.DataFrame(all_rows)
    by_group = pd.DataFrame(all_group_rows)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    parity.to_csv(PARITY_FILE, index=False)
    by_group.to_csv(GROUP_FILE, index=False)

    definitions = {
        "protected_attribute": PROTECTED_ATTRIBUTE,
        "groups": GROUPS,
        "favourable_outcome": FAVOURABLE_OUTCOME,
        "reference_population": REFERENCE_POPULATION,
        "reference_group_rule": REFERENCE_GROUP_RULE,
        "cut_offs": CUT_OFFS,
        "tolerances": TOLERANCES,
        "interval_method": {
            "kind": "non-parametric bootstrap, percentile interval",
            "replicates": BOOTSTRAP_SAMPLES,
            "confidence": CONFIDENCE,
            "seed": SEED,
            "resampling": "test rows with replacement, not stratified by "
                          "group, so group sizes vary as they would in a "
                          "fresh sample",
            "covers": "sampling variation in the test split only",
            "does_not_cover": ["the choice of model", "the split seed",
                               "that both datasets are synthetic"],
        },
        "verdict_rule": "a metric is called within tolerance only when the "
                        "whole interval sits inside it, and a breach only "
                        "when all of it sits outside; otherwise undecided",
    }
    DEFINITIONS_FILE.write_text(
        json.dumps(definitions, indent=2, ensure_ascii=False), encoding="utf-8")

    show = ["dataset", "cut_off", "metric", "value", "ci_low", "ci_high",
            "tolerance", "verdict"]
    print("-" * 94)
    print("PARITY WITH INTERVALS")
    print("-" * 94)
    print(parity[show].to_string(index=False))

    print()
    for path in [PARITY_FILE, GROUP_FILE, DEFINITIONS_FILE]:
        print("  written:", path.relative_to(PROJECT_ROOT))

    return 0
