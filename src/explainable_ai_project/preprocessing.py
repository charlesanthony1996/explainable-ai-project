"""
Inference pathway, block 2A, steps 1 - 3: applicant input, check the input,
prepare like training.

    1 applicant input       one applicant record (a dictionary or a JSON file)
    2 check the input       required fields present and valid; values outside
                            the training 1st - 99th percentile are flagged as unusual
    3 prepare like training same category labels as training, add loan-to-income
                            and age group, so the saved preprocessor can use it

Nothing here is trained: the category maps come from step 6 (harmonize.py), the
allowed ranges from step 7 (cleaning.py) and the usual ranges from step 27
(models/deferral_settings_dataset_X.json).

Example of an applicant record (amounts and income per year, term in months):

    {
        "age": 35,
        "annual_income": 60000,
        "credit_score": 700,
        "loan_amount": 15000,
        "loan_term_months": 36,
        "marital_status": "Married",
        "education": "Undergraduate",
        "employment_status": "Employed",
        "loan_purpose": "Home",
        "dependents": 1
    }
"""

import json
import math
from pathlib import Path

import pandas as pd

from explainable_ai_project.add_features import add_age_group
from explainable_ai_project.add_features import add_loan_to_income
from explainable_ai_project.cleaning import RANGE_RULES
from explainable_ai_project.deferral import load_deferral_settings
from explainable_ai_project.harmonize import CATEGORY_MAPS
from explainable_ai_project.load_split import load_split


# ------------------------------------------------------------------
# The applicant fields
# ------------------------------------------------------------------
PRIMARY_DATASET = "dataset_1"     # model A: its training data sets the usual ranges

NUMBER_FIELDS = ["age", "dependents", "annual_income", "credit_score", "loan_amount", "loan_term_months"]
CATEGORY_FIELDS = ["marital_status", "education", "employment_status", "loan_purpose"]
REQUIRED_FIELDS = NUMBER_FIELDS + CATEGORY_FIELDS

# Plain-word names, used in messages and in the contract
FIELD_LABELS = {
    "age": "age",
    "dependents": "number of dependents",
    "annual_income": "annual income",
    "credit_score": "credit score",
    "loan_amount": "requested loan amount",
    "loan_term_months": "loan term (months)",
    "marital_status": "marital status",
    "education": "education",
    "employment_status": "employment status",
    "loan_purpose": "loan purpose",
}

# Fields that must be whole numbers
WHOLE_NUMBER_FIELDS = ["age", "dependents", "credit_score", "loan_term_months"]


def shared_labels():
    """Every label the models were trained with, per category field.

    Both datasets' raw labels are accepted too ("Graduate", "Bachelor", "Auto", ...),
    and mapped exactly like in training (step 6).
    """
    labels = {}
    raw_to_shared = {}
    for field in CATEGORY_FIELDS:
        labels[field] = []
        raw_to_shared[field] = {}
        for dataset_name in CATEGORY_MAPS:
            for raw_value, shared_value in CATEGORY_MAPS[dataset_name][field].items():
                raw_to_shared[field][raw_value.lower()] = shared_value
                raw_to_shared[field][shared_value.lower()] = shared_value
                if shared_value not in labels[field]:
                    labels[field].append(shared_value)
    return labels, raw_to_shared


ALLOWED_LABELS, RAW_TO_SHARED = shared_labels()


# ------------------------------------------------------------------
# Step 1: applicant input
# ------------------------------------------------------------------

def read_applicant(source):
    """Return the applicant as a dictionary. source = a dictionary or a path to a JSON file."""

    if isinstance(source, dict):
        return dict(source)

    file_path = Path(source)
    with open(file_path, "r", encoding="utf-8") as file:
        applicant = json.load(file)
    if not isinstance(applicant, dict):
        raise ValueError(str(file_path) + " must contain one JSON object (one applicant)")
    return applicant


# ------------------------------------------------------------------
# Step 2: check the input
# ------------------------------------------------------------------

def is_empty(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def to_number(value):
    """'60,000' or '60000' or 60000 -> 60000.0. Returns None if it is not a number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace(",", "").replace("_", "")
        for symbol in ["€", "$", "EUR", "USD"]:
            text = text.replace(symbol, "")
        try:
            number = float(text.strip())
        except ValueError:
            return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def check_input(applicant):
    """Check one applicant record.

    Returns a dictionary:
        record          the cleaned record (numbers as numbers, shared category labels)
        missing         required fields that are empty (case is deferred, step 7)
        errors          values that are impossible (the system cannot answer at all)
        unusual         values outside the training 1st - 99th percentile (deferred, step 7)
        notes           other things worth telling (e.g. an unknown category)
        flags           all of the above as short texts, for the contract
    """

    record = {}
    missing = []
    errors = []
    unusual = []
    notes = []

    unknown_fields = [field for field in applicant if field not in REQUIRED_FIELDS]
    if unknown_fields:
        notes.append("ignored fields the models do not use: " + ", ".join(sorted(unknown_fields)))

    # Numbers: present, a number, inside the allowed range of step 7
    for field in NUMBER_FIELDS:
        value = applicant.get(field)
        if is_empty(value):
            missing.append(field)
            record[field] = None
            continue

        number = to_number(value)
        if number is None:
            errors.append(FIELD_LABELS[field] + " is not a number: " + repr(value))
            record[field] = None
            continue

        if field in WHOLE_NUMBER_FIELDS and number != int(number):
            errors.append(FIELD_LABELS[field] + " must be a whole number: " + repr(value))

        lowest, highest = RANGE_RULES[field]
        if lowest is not None and number < lowest:
            errors.append(FIELD_LABELS[field] + " is below the allowed minimum " + str(lowest) + ": " + repr(value))
        if highest is not None and number > highest:
            errors.append(FIELD_LABELS[field] + " is above the allowed maximum " + str(highest) + ": " + repr(value))

        record[field] = number

    # Categories: present and one of the labels the models know
    for field in CATEGORY_FIELDS:
        value = applicant.get(field)
        if is_empty(value):
            missing.append(field)
            record[field] = None
            continue

        key = str(value).strip().lower()
        if key in RAW_TO_SHARED[field]:
            record[field] = RAW_TO_SHARED[field][key]
        else:
            # Training also turned unknown labels into "Other" (step 6)
            record[field] = "Other"
            allowed = ", ".join(ALLOWED_LABELS[field])
            notes.append(FIELD_LABELS[field] + " '" + str(value) + "' is not a known label (" + allowed + "), used 'Other'")

    # Unusual: outside the 1st - 99th percentile of model A's training part
    if not errors:
        usual_ranges = load_deferral_settings(PRIMARY_DATASET)["usual_ranges"]
        for field, (low, high) in usual_ranges.items():
            number = record.get(field)
            if number is None:
                continue
            if number < low or number > high:
                unusual.append({
                    "field": field,
                    "value": number,
                    "usual_low": round(low, 2),
                    "usual_high": round(high, 2),
                })

    flags = []
    for field in missing:
        flags.append("missing: " + FIELD_LABELS[field])
    for item in unusual:
        text = (FIELD_LABELS[item["field"]] + " " + format_value(item["value"]) + " is outside the usual range "
                + format_value(item["usual_low"]) + " to " + format_value(item["usual_high"]))
        flags.append("unusual: " + text)
    for note in notes:
        flags.append("note: " + note)

    return {
        "record": record,
        "missing": missing,
        "errors": errors,
        "unusual": unusual,
        "notes": notes,
        "flags": flags,
    }


def format_value(number):
    if number is None:
        return "missing"
    if float(number) == int(number):
        return str(int(number))
    return str(round(float(number), 2))


# ------------------------------------------------------------------
# Step 3: prepare like training
# ------------------------------------------------------------------
_training_medians = None


def training_medians():
    """Median of every number field in model A's TRAINING part.

    Only used to fill a missing field, so the models can still give an estimate.
    Such a case is always deferred to a human (rule 3, missing data).
    """
    global _training_medians
    if _training_medians is None:
        train_rows = load_split(PRIMARY_DATASET, "train")["rows"]
        _training_medians = {}
        for field in NUMBER_FIELDS:
            _training_medians[field] = float(train_rows[field].median())
    return _training_medians


def prepare_rows(record):
    """One-row table in exactly the format the training steps produced.

    Missing numbers are filled with the training median, missing categories with
    'Other' (the one-hot encoder treats it like training did).
    """

    filled = dict(record)
    medians = training_medians()
    for field in NUMBER_FIELDS:
        if filled.get(field) is None:
            filled[field] = medians[field]
    for field in CATEGORY_FIELDS:
        if filled.get(field) is None:
            filled[field] = "Other"

    rows = pd.DataFrame([filled], columns=REQUIRED_FIELDS)
    for field in NUMBER_FIELDS:
        rows[field] = rows[field].astype(float)

    rows = add_loan_to_income(rows)
    rows = add_age_group(rows)
    return rows
