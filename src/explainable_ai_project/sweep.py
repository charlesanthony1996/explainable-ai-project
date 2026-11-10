"""
Inference pathway, block 2A, step 5: amount sweep.

Tries 100 amounts across the applicant's allowed range (same limits as training,
models/sweep_limits_dataset_X.json), recomputing loan-to-income each time.
The highest amount with P(approve) >= 0.8 is the estimated maximum; 0.7 and 0.9
are reported too.

This uses the same sweep function as training step 22 (test_set_sweep.py), so
inference and the test-set results can never drift apart.
"""

import math

from explainable_ai_project.models import MODEL_DATASETS
from explainable_ai_project.models import load_model_parts
from explainable_ai_project.test_set_sweep import DEFAULT_THRESHOLD
from explainable_ai_project.test_set_sweep import THRESHOLDS
from explainable_ai_project.test_set_sweep import load_sweep_limits
from explainable_ai_project.test_set_sweep import sweep_applicants


def as_amount(value):
    """Round to a whole amount; 'no amount found' (NaN) becomes None."""
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return int(round(value))


def sweep_one_model(rows, model_name):
    parts = load_model_parts(model_name)
    limits = load_sweep_limits(parts["dataset"])

    table = sweep_applicants(rows, parts["model"], parts["preprocessor"], parts["feature_names"], limits)
    row = table.iloc[0]

    maximum = {}
    for threshold in THRESHOLDS:
        maximum[str(threshold)] = as_amount(row["max_amount_" + str(threshold)])

    return {
        "maximum_amount": maximum,
        "amounts_tried_from": as_amount(row["sweep_lowest_amount"]),
        "amounts_tried_to": as_amount(row["sweep_highest_amount"]),
        # True: even the highest amount we may try passes, so the real maximum may be higher
        "maximum_is_upper_limit": bool(row["max_is_upper_limit"]),
    }


def amount_sweep(rows):
    """Step 5 for ONE applicant, with both models."""

    result = {"default_threshold": DEFAULT_THRESHOLD, "thresholds": list(THRESHOLDS)}
    for model_name in MODEL_DATASETS:
        result[model_name] = sweep_one_model(rows, model_name)
    return result
