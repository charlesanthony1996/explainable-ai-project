"""
Inference pathway, block 2A, step 8: build the contract.

Every number the LLM may use goes into ONE JSON object, the "contract".
validate_contract() checks it before the LLM ever sees it.

Why: a language model is good at explaining, but bad at arithmetic and happy
to invent numbers. So all numbers are computed here, in code, including the
simple ones (like the difference between the requested amount and the
maximum). The prompt tells Qwen to use only numbers from the contract, and
step 13 checks the answer against it.
"""

import datetime
import json
import math

from explainable_ai_project.models import MODEL_DATASETS
from explainable_ai_project.models import PRIMARY_MODEL
from explainable_ai_project.preprocessing import FIELD_LABELS
from explainable_ai_project.preprocessing import REQUIRED_FIELDS


CONTRACT_VERSION = "1"


class ContractError(ValueError):
    """The contract is not safe to give to the LLM."""


def clean_number(value):
    """Plain Python numbers for JSON (numpy numbers are not JSON)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) or hasattr(value, "item"):
        number = float(value)
        if number == int(number) and abs(number) < 1e15:
            return int(number)
        return round(number, 4)
    return value


def build_contract(input_check, rows, probabilities, sweep, explanation, gate):
    """Step 8: put everything from steps 2 - 7 into one checked JSON object."""

    record = input_check["record"]
    prepared = rows.iloc[0]
    threshold = sweep["default_threshold"]
    threshold_key = str(threshold)

    applicant = {}
    for field in REQUIRED_FIELDS:
        applicant[field] = clean_number(record.get(field))
    applicant["loan_to_income"] = round(float(prepared["loan_to_income"]), 3)
    applicant["age_group"] = str(prepared["age_group"])

    filled_in = {}
    for field in input_check["missing"]:
        filled_in[field] = clean_number(prepared[field])

    requested = clean_number(prepared["loan_amount"])
    main_sweep = sweep[PRIMARY_MODEL]
    main_maximum = main_sweep["maximum_amount"][threshold_key]
    main_probability = probabilities[PRIMARY_MODEL]

    if main_maximum is None:
        within = None
        difference = None
        size_of_difference = None
    else:
        within = requested <= main_maximum
        difference = clean_number(requested - main_maximum)
        size_of_difference = clean_number(abs(requested - main_maximum))

    comparison = comparison_text(requested, main_maximum, size_of_difference, threshold)
    maximum_meaning = maximum_meaning_text(main_sweep, main_maximum, threshold)
    models_agree, second_opinion = second_opinion_text(probabilities, sweep, threshold)

    models = {}
    for model_name, dataset_name in MODEL_DATASETS.items():
        model_sweep = sweep[model_name]
        models[model_name] = {
            "trained_on": dataset_name,
            "role": "main model" if model_name == PRIMARY_MODEL else "second opinion",
            "p_approve_at_requested_amount": probabilities[model_name],
            "maximum_amount_by_threshold": dict(model_sweep["maximum_amount"]),
            "amounts_tried_from": model_sweep["amounts_tried_from"],
            "amounts_tried_to": model_sweep["amounts_tried_to"],
            "maximum_is_upper_limit": model_sweep["maximum_is_upper_limit"],
        }

    contract = {
        "contract_version": CONTRACT_VERSION,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "requested_amount": requested,
            "estimated_maximum_amount": main_maximum,
            "approval_threshold": threshold,
            "p_approve_at_requested_amount": main_probability,
            "likely_approved_at_requested_amount": main_probability >= threshold,
            "requested_amount_within_maximum": within,
            "requested_minus_maximum": difference,
            "size_of_difference": size_of_difference,
            "comparison": comparison,
            "maximum_meaning": maximum_meaning,
            "models_agree_at_requested_amount": models_agree,
            "second_opinion": second_opinion,
            "no_amount_reaches_threshold": main_maximum is None,
            "maximum_is_upper_limit": main_sweep["maximum_is_upper_limit"],
            "human_review_required": gate["deferred"],
        },
        "applicant": applicant,
        "field_labels": dict(FIELD_LABELS),
        "filled_in_fields": filled_in,
        "input_flags": list(input_check["flags"]),
        "models": models,
        "explanation": {
            "model": explanation["model"],
            "method": explanation["method"],
            "base_value": explanation["base_value"],
            "top_features": explanation["top_features"],
        },
        "deferral": {
            "human_review_required": gate["deferred"],
            "reasons": [reason["text"] for reason in gate["reasons"]],
            "rules": [reason["rule"] for reason in gate["reasons"]],
        },
        "limitations": [
            "The estimate comes from patterns in two historical, partly synthetic datasets, not from a lender's policy.",
            "It is not a loan offer or a lending decision.",
            "The maximum is the highest amount tried with P(approve) at or above the threshold; amounts outside the training range are not tried.",
        ],
    }

    validate_contract(contract)
    return contract


def amount_text(value):
    return "{:,.0f}".format(value)


def comparison_text(requested, maximum, size_of_difference, threshold):
    """The comparison of requested amount and maximum, written in code.

    Comparing two numbers is arithmetic too, and a small LLM gets it wrong when the
    question assumes the opposite ("why is my maximum lower ...?" when it is higher).
    So the answer to "higher or lower?" is given as text.
    """
    if maximum is None:
        return ("No amount in the tried range reaches the " + str(threshold)
                + " approval threshold, so there is no estimated maximum.")
    if requested < maximum:
        return ("The estimated maximum (" + amount_text(maximum) + ") is HIGHER than the requested amount ("
                + amount_text(requested) + "), by " + amount_text(size_of_difference)
                + ". The requested amount is within the maximum.")
    if requested == maximum:
        return ("The estimated maximum is EQUAL to the requested amount (" + amount_text(requested) + ").")
    return ("The estimated maximum (" + amount_text(maximum) + ") is LOWER than the requested amount ("
            + amount_text(requested) + "), by " + amount_text(size_of_difference)
            + ". The requested amount is above the maximum.")


def maximum_meaning_text(main_sweep, maximum, threshold):
    """What the maximum means, in words (the yes/no field alone was misread by Qwen)."""
    low = amount_text(main_sweep["amounts_tried_from"])
    high = amount_text(main_sweep["amounts_tried_to"])
    tried = "Amounts from " + low + " to " + high + " were tried. "
    if maximum is None:
        return tried + "None of them reaches the " + str(threshold) + " approval threshold."
    if main_sweep["maximum_is_upper_limit"]:
        return (tried + "Even the highest tried amount (" + high + ") reaches the threshold, so the real "
                "maximum may be higher than " + amount_text(maximum) + ", but the data cannot show it.")
    return (tried + "Amounts above " + amount_text(maximum) + " fall below the " + str(threshold)
            + " approval threshold, so " + amount_text(maximum) + " is the highest amount that reaches it.")


def second_opinion_text(probabilities, sweep, threshold):
    """Do models A and B agree at the requested amount? Answered in code, as text."""
    main_model = PRIMARY_MODEL
    other_model = [name for name in MODEL_DATASETS if name != main_model][0]
    main_probability = probabilities[main_model]
    other_probability = probabilities[other_model]
    main_passes = main_probability >= threshold
    other_passes = other_probability >= threshold
    agree = main_passes == other_passes

    def side(passes):
        return "at or above" if passes else "below"

    other_maximum = sweep[other_model]["maximum_amount"][str(threshold)]
    other_maximum_text = amount_text(other_maximum) if other_maximum is not None else "none found"

    text = ("Model " + other_model + " (second opinion, trained on a different dataset) gives P(approve) "
            + str(other_probability) + " at the requested amount; model " + main_model + " gives "
            + str(main_probability) + ". Model " + main_model + " is " + side(main_passes) + " and model "
            + other_model + " is " + side(other_passes) + " the " + str(threshold) + " threshold, so the two models "
            + ("AGREE" if agree else "DO NOT AGREE") + " on the requested amount. Model " + other_model
            + "'s estimated maximum is " + other_maximum_text + ".")
    return agree, text


# ------------------------------------------------------------------
# validate_contract: the check before the LLM sees anything
# ------------------------------------------------------------------

def all_numbers(value, path="contract"):
    """Every number in the contract with its path, to check they are finite."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(all_numbers(item, path + "." + str(key)))
    elif isinstance(value, list):
        for position, item in enumerate(value):
            found.extend(all_numbers(item, path + "[" + str(position) + "]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        found.append((path, value))
    return found


def validate_contract(contract):
    """Raise ContractError listing every problem; return True if the contract is safe."""

    problems = []

    for key in ["summary", "applicant", "models", "explanation", "deferral", "limitations"]:
        if key not in contract:
            problems.append("missing section: " + key)
    if problems:
        raise ContractError("; ".join(problems))

    try:
        json.dumps(contract)
    except (TypeError, ValueError) as error:
        problems.append("not valid JSON: " + str(error))

    for path, number in all_numbers(contract):
        if isinstance(number, float) and (math.isnan(number) or math.isinf(number)):
            problems.append(path + " is not a finite number")

    summary = contract["summary"]
    threshold = summary["approval_threshold"]
    if not 0 < threshold < 1:
        problems.append("approval_threshold must be between 0 and 1")

    for model_name, model in contract["models"].items():
        probability = model["p_approve_at_requested_amount"]
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            problems.append(model_name + ": P(approve) must be between 0 and 1")

        low = model["amounts_tried_from"]
        high = model["amounts_tried_to"]
        if low is None or high is None or low > high:
            problems.append(model_name + ": the range of tried amounts is not valid")
            continue

        # A higher threshold can never allow a higher maximum
        previous = None
        for threshold_key in sorted(model["maximum_amount_by_threshold"], key=float):
            maximum = model["maximum_amount_by_threshold"][threshold_key]
            if maximum is None:
                previous = -1
                continue
            if not isinstance(maximum, int):
                problems.append(model_name + ": maximum at " + threshold_key + " is not a whole amount")
            if maximum < low - 1 or maximum > high + 1:
                problems.append(model_name + ": maximum at " + threshold_key + " is outside the tried range")
            if previous == -1:
                problems.append(model_name + ": an amount was found at " + threshold_key + " but not at a lower threshold")
            elif previous is not None and maximum > previous:
                problems.append(model_name + ": the maximum rises with the threshold (" + threshold_key + ")")
            previous = maximum

    main = contract["models"][PRIMARY_MODEL]
    if summary["estimated_maximum_amount"] != main["maximum_amount_by_threshold"][str(threshold)]:
        problems.append("summary maximum does not match model " + PRIMARY_MODEL)
    if summary["p_approve_at_requested_amount"] != main["p_approve_at_requested_amount"]:
        problems.append("summary P(approve) does not match model " + PRIMARY_MODEL)

    maximum = summary["estimated_maximum_amount"]
    requested = summary["requested_amount"]
    if maximum is not None and abs(summary["requested_minus_maximum"] - (requested - maximum)) > 0.01:
        problems.append("requested_minus_maximum is not requested - maximum")
    if maximum is not None:
        word = "HIGHER" if requested < maximum else ("EQUAL" if requested == maximum else "LOWER")
        if word not in summary["comparison"]:
            problems.append("the comparison text does not match the numbers")
    agree_word = "DO NOT AGREE" if not summary["models_agree_at_requested_amount"] else "AGREE"
    if agree_word not in summary["second_opinion"] or (summary["models_agree_at_requested_amount"] and "DO NOT AGREE" in summary["second_opinion"]):
        problems.append("the second-opinion text does not match models_agree_at_requested_amount")
    if (maximum is None) != summary["no_amount_reaches_threshold"]:
        problems.append("no_amount_reaches_threshold does not match the maximum")

    deferral = contract["deferral"]
    if deferral["human_review_required"] != (len(deferral["reasons"]) > 0):
        problems.append("human_review_required must be true exactly when there are reasons")
    if summary["human_review_required"] != deferral["human_review_required"]:
        problems.append("summary and deferral disagree on human review")

    top_features = contract["explanation"]["top_features"]
    if not 1 <= len(top_features) <= 5:
        problems.append("there must be 1 to 5 top features")
    for feature in top_features:
        if feature["effect"] not in ["raises approval", "lowers approval", "no effect"]:
            problems.append("unknown effect for " + str(feature.get("field")))

    if problems:
        raise ContractError("; ".join(problems))
    return True
