"""
Training pathway, block 1C, step 27: choose deferral rules (Topic 3: deferral).

Deferral = the system does not give a definite answer and sends the case to a
human instead. A good deferral rule catches the cases where the model is most
likely to be wrong, without sending everyone to a human.

Five rules. An applicant is deferred if ANY of them applies:

    1. borderline     P(approve) at the requested amount is within 0.05 of the
                      0.8 threshold: a small change could flip the answer
    2. unusual input  a number (age, income, credit score, ...) is outside the
                      1st to 99th percentile of the training part: the model has
                      seen almost no one like this applicant
    3. missing data   a required field is empty
    4. disagreement   models A and B, trained on different data, differ by more
                      than 0.5 in P(approve) for this applicant
    5. weak group     the applicant's age group is one where the model cannot rank
                      applicants reliably (within-group AUC below 0.6 on the
                      VALIDATION part). Step 26 found this for model A and 61+.

Rules 2, 3 and 5 are SAFETY rules and are always on: the model should not decide
alone when the input is unusual, incomplete, or from a group it cannot rank.

Rules 1 and 4 are EVIDENCE rules: they are only switched on for a model if, on
the VALIDATION part, the model really is less accurate on the cases they catch
(at least 3 percentage points below its overall accuracy). Otherwise they would
send people to a human for no benefit.

The settings are chosen from the training and validation parts and saved, so
inference uses exactly the same rules. The rules are then EVALUATED on the test part:
    - how many applicants each rule catches, overall and per age group
    - is the model really less accurate on the deferred cases? (it should be)

Put this file in:  src/explainable_ai_project/deferral.py

Writes:
    models/deferral_settings_dataset_X.json     the rules (used at inference)
    results/tables/deferral_rule_choice.csv     validation evidence for rules 1 and 4
    results/tables/deferral_rules.csv           share caught by each rule
    results/tables/deferral_by_age_group.csv    deferral rate per age group
    results/tables/deferral_evaluation.csv      accuracy on deferred vs kept cases
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from explainable_ai_project.calibrate import load_calibrated_model
from explainable_ai_project.load_split import load_split
from explainable_ai_project.load_split import read_feature_names
from explainable_ai_project.load_split import read_preprocessor


# ------------------------------------------------------------------
# Where the files go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

MODELS_FOLDER = PROJECT_ROOT / "models"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"

DATASET_NAMES = ["dataset_1", "dataset_2"]
OTHER_DATASET = {"dataset_1": "dataset_2", "dataset_2": "dataset_1"}
AGE_GROUP_ORDER = ["18-30", "31-45", "46-60", "61+"]

# ------------------------------------------------------------------
# Rule settings
# ------------------------------------------------------------------
THRESHOLD = 0.8
BORDERLINE_MARGIN = 0.05
LOW_PERCENTILE = 0.01
HIGH_PERCENTILE = 0.99
DISAGREEMENT_LIMIT = 0.5
WEAK_GROUP_AUC = 0.6
CUT_OFF = 0.5          # for "is the model's answer right?" in the evaluation

NUMBER_INPUTS = ["age", "dependents", "annual_income", "credit_score", "loan_amount", "loan_term_months"]
REQUIRED_INPUTS = NUMBER_INPUTS + ["marital_status", "education", "employment_status", "loan_purpose"]

RULE_NAMES = ["borderline", "unusual_input", "missing_data", "disagreement", "weak_group"]
SAFETY_RULES = ["unusual_input", "missing_data", "weak_group"]
EVIDENCE_RULES = ["borderline", "disagreement"]
MINIMUM_ACCURACY_DROP = 0.03


# ------------------------------------------------------------------
# Probabilities from a dataset's model, for any applicants
# ------------------------------------------------------------------

def probabilities_from(model_dataset, rows):
    """P(approve) from the calibrated model of model_dataset, for these rows.

    The rows may come from either dataset: both use the same column names.
    """

    model = load_calibrated_model(model_dataset)
    preprocessor = read_preprocessor(model_dataset)
    feature_names = read_feature_names(model_dataset)

    filled_rows = rows.copy()
    for column in NUMBER_INPUTS:
        median_value = filled_rows[column].median()
        filled_rows[column] = filled_rows[column].fillna(median_value)
    incomes = filled_rows["annual_income"]
    filled_rows["loan_to_income"] = filled_rows["loan_amount"] / incomes

    feature_values = preprocessor.transform(filled_rows)
    features = pd.DataFrame(feature_values, columns=feature_names, index=rows.index)

    probability_table = model.predict_proba(features)
    probabilities = probability_table[:, 1]
    return probabilities


# ------------------------------------------------------------------
# Choosing the settings (training and validation parts only)
# ------------------------------------------------------------------

def settings_file(dataset_name):
    return MODELS_FOLDER / ("deferral_settings_" + dataset_name + ".json")


def find_weak_groups(dataset_name):
    """Age groups where the model's ranking on VALIDATION is close to guessing."""

    validation = load_split(dataset_name, "validation")
    rows = validation["rows"]
    labels = validation["labels"].to_numpy()
    groups = rows["age_group"].to_numpy()
    probabilities = probabilities_from(dataset_name, rows)

    group_aucs = {}
    weak_groups = []

    for group in AGE_GROUP_ORDER:
        in_group = groups == group
        group_labels = labels[in_group]
        group_probabilities = probabilities[in_group]

        both_outcomes_present = len(np.unique(group_labels)) == 2
        if not both_outcomes_present:
            continue

        auc = roc_auc_score(group_labels, group_probabilities)
        group_aucs[group] = round(auc, 3)

        if auc < WEAK_GROUP_AUC:
            weak_groups.append(group)

    return weak_groups, group_aucs


def choose_settings(dataset_name):
    train = load_split(dataset_name, "train")
    train_rows = train["rows"]

    ranges = {}
    for column in NUMBER_INPUTS:
        lowest = train_rows[column].quantile(LOW_PERCENTILE)
        highest = train_rows[column].quantile(HIGH_PERCENTILE)
        ranges[column] = [float(lowest), float(highest)]

    weak_groups, group_aucs = find_weak_groups(dataset_name)

    settings = {
        "threshold": THRESHOLD,
        "borderline_margin": BORDERLINE_MARGIN,
        "usual_ranges": ranges,
        "required_inputs": REQUIRED_INPUTS,
        "other_model": OTHER_DATASET[dataset_name],
        "disagreement_limit": DISAGREEMENT_LIMIT,
        "weak_groups": weak_groups,
        "validation_auc_per_age_group": group_aucs,
        "enabled_rules": list(RULE_NAMES),
    }

    print("  validation AUC per age group:", group_aucs)
    print("  weak groups (AUC below " + str(WEAK_GROUP_AUC) + "):", weak_groups)

    # Decide which evidence rules to switch on, using the validation part
    enabled_rules, choice_rows = choose_evidence_rules(dataset_name, settings)
    settings["enabled_rules"] = enabled_rules
    print("  rules switched on:", enabled_rules)

    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = settings_file(dataset_name)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)

    return settings, choice_rows


def choose_evidence_rules(dataset_name, settings):
    """Switch an evidence rule on only if the model is clearly less accurate
    on the validation cases it catches."""

    validation = load_split(dataset_name, "validation")
    labels = validation["labels"].to_numpy()
    flags = apply_rules(validation["rows"], dataset_name, settings)
    probabilities = flags["p_approve"].to_numpy()

    predicted_approved = probabilities >= CUT_OFF
    predictions = predicted_approved.astype(int)
    is_correct = predictions == labels
    overall_accuracy = is_correct.mean()

    enabled_rules = list(SAFETY_RULES)
    choice_rows = []

    for rule in EVIDENCE_RULES:
        caught = flags[rule].to_numpy()
        caught_count = int(caught.sum())

        if caught_count == 0:
            caught_accuracy = np.nan
            switched_on = False
        else:
            caught_accuracy = is_correct[caught].mean()
            accuracy_drop = overall_accuracy - caught_accuracy
            switched_on = accuracy_drop >= MINIMUM_ACCURACY_DROP

        if switched_on:
            enabled_rules.append(rule)

        choice_rows.append({
            "dataset": dataset_name,
            "rule": rule,
            "validation_cases_caught": caught_count,
            "validation_accuracy_overall": round(overall_accuracy, 4),
            "validation_accuracy_on_caught": round(caught_accuracy, 4),
            "switched_on": switched_on,
        })

    return enabled_rules, choice_rows


def load_deferral_settings(dataset_name):
    with open(settings_file(dataset_name), "r", encoding="utf-8") as file:
        settings = json.load(file)
    return settings


# ------------------------------------------------------------------
# Applying the rules (also used at inference)
# ------------------------------------------------------------------

def apply_rules(rows, dataset_name, settings):
    """One True/False column per rule, plus 'deferred' and a text with the reasons."""

    result = pd.DataFrame(index=rows.index)

    # 3. Missing data (checked first: the other rules need the values)
    required_values = rows[settings["required_inputs"]]
    missing_values = required_values.isna()
    missing_flags = missing_values.any(axis=1)
    result["missing_data"] = missing_flags

    # 1. Borderline
    probabilities = probabilities_from(dataset_name, rows)
    distance_to_threshold = np.abs(probabilities - settings["threshold"])
    result["borderline"] = distance_to_threshold <= settings["borderline_margin"]

    # 2. Unusual input
    unusual = pd.Series(False, index=rows.index)
    for column, limits in settings["usual_ranges"].items():
        values = rows[column]
        too_low = values < limits[0]
        too_high = values > limits[1]
        unusual = unusual | too_low | too_high
    result["unusual_input"] = unusual

    # 4. Disagreement between the two models
    other_probabilities = probabilities_from(settings["other_model"], rows)
    difference = np.abs(probabilities - other_probabilities)
    result["disagreement"] = difference > settings["disagreement_limit"]

    # 5. Weak group
    result["weak_group"] = rows["age_group"].isin(settings["weak_groups"])

    # Combined decision and readable reasons: only the switched-on rules count
    enabled_rules = settings["enabled_rules"]
    enabled_flags = result[enabled_rules]
    any_rule = enabled_flags.any(axis=1)
    result["deferred"] = any_rule

    reasons = []
    for row_label in result.index:
        row_reasons = []
        for rule in enabled_rules:
            if result.at[row_label, rule]:
                row_reasons.append(rule)
        reasons_text = ", ".join(row_reasons)
        reasons.append(reasons_text)
    result["reasons"] = reasons

    result["p_approve"] = np.round(probabilities, 4)
    result["p_approve_other_model"] = np.round(other_probabilities, 4)
    return result


def deferral(applicant_rows, dataset_name):
    """Inference helper: apply the saved rules to one or more applicants."""
    settings = load_deferral_settings(dataset_name)
    result = apply_rules(applicant_rows, dataset_name, settings)
    return result


# ------------------------------------------------------------------
# Evaluating the rules on the TEST part
# ------------------------------------------------------------------

def quality_of(labels, probabilities):
    if len(labels) == 0:
        return np.nan, np.nan

    predicted_approved = probabilities >= CUT_OFF
    predictions = predicted_approved.astype(int)
    accuracy = accuracy_score(labels, predictions)

    both_outcomes_present = len(np.unique(labels)) == 2
    if both_outcomes_present:
        auc = roc_auc_score(labels, probabilities)
    else:
        auc = np.nan

    return round(accuracy, 4), round(auc, 4)


def evaluate_one_dataset(dataset_name, settings):
    test = load_split(dataset_name, "test")
    rows = test["rows"]
    labels = test["labels"].to_numpy()

    result = apply_rules(rows, dataset_name, settings)
    deferred = result["deferred"].to_numpy()
    probabilities = result["p_approve"].to_numpy()

    # Share caught by each rule
    rule_rows = []
    for rule in RULE_NAMES:
        share = result[rule].mean()
        is_on = rule in settings["enabled_rules"]
        rule_rows.append({
            "dataset": dataset_name,
            "rule": rule,
            "switched_on": is_on,
            "share_of_applicants": round(share, 4),
        })

    deferred_share = deferred.mean()
    rule_rows.append({
        "dataset": dataset_name,
        "rule": "deferred (any switched-on rule)",
        "switched_on": True,
        "share_of_applicants": round(deferred_share, 4),
    })

    # Deferral rate per age group
    group_rows = []
    for group in AGE_GROUP_ORDER:
        in_group = rows["age_group"].to_numpy() == group
        group_rate = deferred[in_group].mean()
        group_rows.append({
            "dataset": dataset_name,
            "age_group": group,
            "applicants": int(in_group.sum()),
            "deferral_rate": round(group_rate, 4),
        })

    # Is the model worse on deferred cases?
    kept = ~deferred
    deferred_accuracy, deferred_auc = quality_of(labels[deferred], probabilities[deferred])
    kept_accuracy, kept_auc = quality_of(labels[kept], probabilities[kept])
    all_accuracy, all_auc = quality_of(labels, probabilities)

    evaluation_row = {
        "dataset": dataset_name,
        "applicants": len(labels),
        "deferred": int(deferred.sum()),
        "deferral_rate": round(deferred.mean(), 4),
        "accuracy_all": all_accuracy,
        "accuracy_kept": kept_accuracy,
        "accuracy_deferred": deferred_accuracy,
        "auc_all": all_auc,
        "auc_kept": kept_auc,
    }

    return rule_rows, group_rows, evaluation_row


# ------------------------------------------------------------------
# The step
# ------------------------------------------------------------------

def choose_deferral_rules():
    """Step 27: choose the deferral settings, then evaluate them on the test part."""

    all_choice_rows = []
    all_rule_rows = []
    all_group_rows = []
    evaluation_rows = []

    for dataset_name in DATASET_NAMES:
        print(dataset_name)
        settings, choice_rows = choose_settings(dataset_name)
        all_choice_rows.extend(choice_rows)
        print("  saved", settings_file(dataset_name).relative_to(PROJECT_ROOT))

        rule_rows, group_rows, evaluation_row = evaluate_one_dataset(dataset_name, settings)
        all_rule_rows.extend(rule_rows)
        all_group_rows.extend(group_rows)
        evaluation_rows.append(evaluation_row)
        print()

    choice_table = pd.DataFrame(all_choice_rows)
    rule_table = pd.DataFrame(all_rule_rows)
    group_table = pd.DataFrame(all_group_rows)
    evaluation_table = pd.DataFrame(evaluation_rows)

    print("Evidence rules, chosen on the validation part:")
    print(choice_table.to_string(index=False))
    print()
    print("Share of test applicants caught by each rule:")
    print(rule_table.to_string(index=False))
    print()
    print("Deferral rate per age group:")
    print(group_table.to_string(index=False))
    print()
    print("Is the model less accurate on the deferred cases?")
    print(evaluation_table.to_string(index=False))
    print()

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    tables = {
        "deferral_rule_choice.csv": choice_table,
        "deferral_rules.csv": rule_table,
        "deferral_by_age_group.csv": group_table,
        "deferral_evaluation.csv": evaluation_table,
    }
    for file_name, table in tables.items():
        output_file = TABLES_FOLDER / file_name
        table.to_csv(output_file, index=False)
        print("Saved", output_file.relative_to(PROJECT_ROOT))

    return evaluation_table