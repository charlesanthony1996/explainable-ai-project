"""
Inference pathway, step 8: the contract, and the checks that use it.

THE IDEA
--------
The language model must never compute a number. Every number it is allowed to
say is collected here first, in one JSON object: the applicant's own values, the
two models' probabilities, the estimated maximum amount, the SHAP drivers and
the deferral decision.

That turns "the model should not hallucinate figures" from a hope into a
measurement. After the model answers, `check_numbers` pulls every number out of
the text and tests whether it appears in the contract. A number that does not is
a hallucination - found automatically, with no human reading the answer.

This is the single most useful metric in the evaluation, because it separates
configuration A (no contract: cannot know any figure) from B and C (contract:
should get every figure right) without any subjective judgement.

WHAT COUNTS AS A MATCH
----------------------
Numbers are matched with a relative tolerance (1 % by default), so "about
31,000" still matches 31,200 and a rounded percentage still matches its
probability. Probabilities are stored twice, as 0.74 and as 74, because an
answer may phrase either one.

Small integers (0 to 10) are ignored by default: they are almost always list
markers ("1.", "2.") or counts of reasons, not claims about the applicant.
`check_numbers(..., ignore_small_integers=False)` turns that off.

Usage
-----
    from explainable_ai_project.contract import build_contract, check_numbers

    contract = build_contract({"age": 34, "annual_income": 72000, ...})
    report = check_numbers(answer_text, contract)
"""

import json
import re

import numpy as np
import pandas as pd

from explainable_ai_project.add_features import add_age_group
from explainable_ai_project.add_features import add_loan_to_income
from explainable_ai_project.deferral import deferral
from explainable_ai_project.deferral import load_deferral_settings
from explainable_ai_project.explain import explain_applicant
from explainable_ai_project.test_set_sweep import estimate_max_amount


DATASET_NAMES = ["dataset_1", "dataset_2"]
PRIMARY_DATASET = "dataset_1"
DEFAULT_THRESHOLD = 0.8

# The values an applicant has to supply (same list the deferral rules use)
REQUIRED_INPUTS = [
    "age",
    "dependents",
    "annual_income",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    "marital_status",
    "education",
    "employment_status",
    "loan_purpose",
]

# A hyphen BETWEEN two digits is a range ("31-45", "18-30"), not a minus sign.
# Without the lookbehind, the age group "31-45" reads as 31 and -45, and both
# then count as numbers the contract does not contain - a false hallucination.
NUMBER_PATTERN = re.compile(
    r"(?<![\d.])-?\d+(?:,\d{3})*(?:\.\d+)?|(?<![\d.])-?\.\d+"
)


# ------------------------------------------------------------------
# Turning a dictionary into a row the models understand
# ------------------------------------------------------------------

def make_applicant_row(values):
    """One applicant as a one-row table, prepared exactly as in training.

    loan_to_income and age_group are added with the SAME functions the training
    pathway used (step 8), so inference can never group or divide differently.
    """

    missing = [name for name in REQUIRED_INPUTS if name not in values]
    if missing:
        raise ValueError("Missing applicant values: " + ", ".join(missing))

    row = pd.DataFrame([dict(values)])
    row = add_loan_to_income(row)
    row = add_age_group(row)

    if "applicant_id" not in row.columns:
        row["applicant_id"] = "inference-1"

    return row


# ------------------------------------------------------------------
# What one model says about this applicant
# ------------------------------------------------------------------

def model_result(applicant_row, dataset_name):
    """Probability at the requested amount, and the estimated maximum."""

    sweep = estimate_max_amount(applicant_row, dataset_name)
    sweep_row = sweep.iloc[0]

    flags = deferral(applicant_row, dataset_name)
    flags_row = flags.iloc[0]

    def as_number(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)

    result = {
        "p_approve_at_requested": as_number(flags_row["p_approve"]),
        "max_amount_0.7": as_number(sweep_row["max_amount_0.7"]),
        "max_amount_0.8": as_number(sweep_row["max_amount_0.8"]),
        "max_amount_0.9": as_number(sweep_row["max_amount_0.9"]),
        "max_is_upper_limit": bool(sweep_row["max_is_upper_limit"]),
        "sweep_lowest_amount": as_number(sweep_row["sweep_lowest_amount"]),
        "sweep_highest_amount": as_number(sweep_row["sweep_highest_amount"]),
    }
    return result


def deferral_result(applicant_row, dataset_name):
    """The deferral decision and the reasons behind it."""

    settings = load_deferral_settings(dataset_name)
    flags = deferral(applicant_row, dataset_name)
    flags_row = flags.iloc[0]

    reasons_text = flags_row["reasons"]
    reasons = [part.strip() for part in reasons_text.split(",") if part.strip()]

    return {
        "flagged": bool(flags_row["deferred"]),
        "reasons": reasons,
        "threshold": settings["threshold"],
        "enabled_rules": settings["enabled_rules"],
    }


# ------------------------------------------------------------------
# The contract
# ------------------------------------------------------------------

def build_contract(values, primary_dataset=PRIMARY_DATASET, top_features=5):
    """Everything the language model is allowed to state, in one object."""

    applicant_row = make_applicant_row(values)

    models = {}
    for dataset_name in DATASET_NAMES:
        models[dataset_name] = model_result(applicant_row, dataset_name)

    primary = models[primary_dataset]
    maximum_amount = primary["max_amount_" + str(DEFAULT_THRESHOLD)]

    shap_table = explain_applicant(applicant_row, primary_dataset, top_n=top_features)
    drivers = []
    for _, feature_row in shap_table.iterrows():
        drivers.append({
            "feature": feature_row["feature"],
            "value": float(feature_row["value"]),
            "contribution": float(feature_row["contribution"]),
            "direction": feature_row["direction"],
        })

    applicant_values = dict(values)
    applicant_values["loan_to_income"] = round(
        float(applicant_row.iloc[0]["loan_to_income"]), 4
    )
    applicant_values["age_group"] = str(applicant_row.iloc[0]["age_group"])

    contract = {
        "applicant": applicant_values,
        "models": models,
        "estimated_maximum": {
            "threshold": DEFAULT_THRESHOLD,
            "amount": maximum_amount,
            "from_model": primary_dataset,
            "is_upper_limit": primary["max_is_upper_limit"],
            "no_amount_found": maximum_amount is None,
        },
        "top_features": drivers,
        "deferral": deferral_result(applicant_row, primary_dataset),
        "explanation_scale": "SHAP contributions are in the raw forest probability scale",
    }

    contract["allowed_numbers"] = allowed_numbers(contract)
    return contract


def validate_contract(contract):
    """Problems that must be fixed before the model sees the contract.

    Returns a list of messages; an empty list means the contract is usable.
    """

    problems = []

    for key in ["applicant", "models", "estimated_maximum", "top_features", "deferral"]:
        if key not in contract:
            problems.append("missing section: " + key)

    applicant = contract.get("applicant", {})
    for name in REQUIRED_INPUTS:
        if applicant.get(name) is None:
            problems.append("missing applicant value: " + name)

    maximum = contract.get("estimated_maximum", {})
    amount = maximum.get("amount")

    if amount is not None:
        primary_name = maximum.get("from_model")
        primary = contract.get("models", {}).get(primary_name, {})
        highest = primary.get("sweep_highest_amount")
        if highest is not None and amount > highest + 1:
            problems.append("the maximum is above the highest amount the sweep may try")

    if maximum.get("no_amount_found") and amount is not None:
        problems.append("no_amount_found is set but an amount is given")

    if not contract.get("allowed_numbers"):
        problems.append("allowed_numbers is empty, so the answer cannot be checked")

    return problems


# ------------------------------------------------------------------
# Numbers: what may be said, and what was said
# ------------------------------------------------------------------

def allowed_numbers(contract):
    """Every number the model may use.

    Probabilities are added twice (0.74 and 74), because an answer may phrase
    either the probability or the percentage.
    """

    collected = []

    def add(value):
        if value is None:
            return
        if isinstance(value, bool):
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if np.isnan(number):
            return
        collected.append(round(number, 6))
        if 0.0 < number <= 1.0:
            collected.append(round(number * 100, 6))

    for value in contract.get("applicant", {}).values():
        add(value)

    # The age group is a range like "31-45". Both edges are real facts about
    # this applicant, so an answer may name them.
    age_group = contract.get("applicant", {}).get("age_group")
    if isinstance(age_group, str):
        for part in re.findall(r"\d+", age_group):
            add(float(part))

    for model in contract.get("models", {}).values():
        for value in model.values():
            add(value)

    maximum = contract.get("estimated_maximum", {})
    add(maximum.get("amount"))
    add(maximum.get("threshold"))

    for driver in contract.get("top_features", []):
        add(driver.get("value"))
        add(driver.get("contribution"))

    add(contract.get("deferral", {}).get("threshold"))

    unique = sorted(set(collected))
    return unique


def numbers_in(text):
    """Every number that appears in a piece of text."""

    found = []
    for match in NUMBER_PATTERN.findall(text):
        cleaned = match.replace(",", "")
        try:
            found.append(float(cleaned))
        except ValueError:
            continue
    return found


def check_numbers(text, contract, relative_tolerance=0.01,
                  absolute_tolerance=0.005, ignore_small_integers=True,
                  extra_allowed=None):
    """Does every number in the answer come from the contract?

    extra_allowed widens the yardstick with numbers that are legitimate but do
    not belong to the contract. The retrieved passages are the case this exists
    for: an answer that quotes a figure out of the fairness report is doing the
    right thing, yet that figure is not in the contract and would otherwise be
    counted as invented. Passing the passage numbers in turns this into a test
    of whether the answer stayed inside the material it was shown, which is a
    different question from whether its figures match the contract. Run it both
    ways and report both; do not quietly replace one with the other.

    Returns:
        numbers_checked     how many numbers were tested
        numbers_supported   how many were found in the contract
        unsupported         the ones that were not (possible hallucinations)
        share_supported     numbers_supported / numbers_checked
        passed              True if nothing was unsupported
    """

    permitted = list(contract.get("allowed_numbers", []))
    if extra_allowed:
        permitted = permitted + [float(value) for value in extra_allowed]
    said = numbers_in(text)

    checked = []
    unsupported = []

    for number in said:
        if ignore_small_integers and float(number).is_integer() and abs(number) <= 10:
            continue

        checked.append(number)

        supported = False
        for allowed in permitted:
            difference = abs(number - allowed)
            limit = max(absolute_tolerance, relative_tolerance * abs(allowed))
            if difference <= limit:
                supported = True
                break

        if not supported:
            unsupported.append(number)

    count = len(checked)
    supported_count = count - len(unsupported)
    share = (supported_count / count) if count > 0 else 1.0

    return {
        "numbers_checked": count,
        "numbers_supported": supported_count,
        "unsupported": unsupported,
        "share_supported": round(share, 4),
        "passed": len(unsupported) == 0,
    }


def check_citations(text, chunks):
    """Does every source the answer names belong to a chunk that was retrieved?

    Looks for the document ids of the retrieved chunks, and for the bracket
    markers [1] ... [n] used in the prompt.
    """

    retrieved_documents = {chunk["doc_id"] for chunk in chunks}
    retrieved_markers = {str(number) for number in range(1, len(chunks) + 1)}

    named_documents = {
        document for document in retrieved_documents if document in text
    }

    markers_used = set(re.findall(r"\[(\d+)\]", text))
    invalid_markers = sorted(markers_used - retrieved_markers)

    cites_something = bool(named_documents) or bool(markers_used - set(invalid_markers))

    return {
        "documents_named": sorted(named_documents),
        "markers_used": sorted(markers_used),
        "invalid_markers": invalid_markers,
        "cites_a_retrieved_source": cites_something,
        "passed": len(invalid_markers) == 0 and cites_something,
    }


def check_deferral_mentioned(text, contract):
    """If the case was flagged, does the answer say a human has to decide?"""

    flagged = contract.get("deferral", {}).get("flagged", False)
    if not flagged:
        return {"flagged": False, "mentions_human_review": None, "passed": True}

    lowered = text.lower()
    phrases = [
        "human review", "human reviewer", "a human", "manual review",
        "reviewed by a person", "human decision", "referred to a",
        "cannot give a definite", "not a final decision",
    ]
    mentions = any(phrase in lowered for phrase in phrases)

    return {
        "flagged": True,
        "mentions_human_review": mentions,
        "passed": mentions,
    }


def contract_as_json(contract):
    """The contract as text for the prompt (without the helper list of numbers)."""

    for_prompt = {key: value for key, value in contract.items()
                  if key != "allowed_numbers"}
    return json.dumps(for_prompt, indent=2, ensure_ascii=False, default=str)
