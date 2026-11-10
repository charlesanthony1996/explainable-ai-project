"""
Inference pathway, block 2A, step 6: SHAP explanation.

The top 5 features pushing approval up or down for this applicant, at the
amount they asked for.

What exactly is explained:
    SHAP's TreeExplainer works on the trees themselves, so it explains model A's
    Random Forest BEFORE calibration. Calibration (step 19) only rescales the
    probability with a curve that always rises, so it never changes whether a
    feature pushes approval up or down, or which features matter most.

One-hot columns are added back together, so "marital_status_Married" and
"marital_status_Single" count as one feature, "marital status".

If the shap package cannot be installed (it needs a compiled extension), the
same kind of numbers are computed with the "Saabas" method: follow the
applicant's path through every tree and give each split's change in
P(approve) to the feature it split on. The contract records which method was used.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from explainable_ai_project.load_split import make_features
from explainable_ai_project.models import PRIMARY_MODEL
from explainable_ai_project.models import load_model_parts
from explainable_ai_project.preprocessing import CATEGORY_FIELDS
from explainable_ai_project.preprocessing import FIELD_LABELS


TOP_COUNT = 5

FIELD_LABELS_WITH_RATIO = dict(FIELD_LABELS)
FIELD_LABELS_WITH_RATIO["loan_to_income"] = "loan-to-income ratio"

_explainers = {}


def underlying_forest(calibrated_model):
    """The Random Forest inside the calibrated model (CalibratedClassifierCV(FrozenEstimator(forest)))."""

    # FrozenEstimator passes attribute lookups through to the forest, so check the
    # class itself and unwrap until we reach the real Random Forest
    estimator = calibrated_model.calibrated_classifiers_[0].estimator
    for _ in range(5):
        if isinstance(estimator, RandomForestClassifier):
            return estimator
        estimator = estimator.estimator
    raise TypeError("could not find the Random Forest inside the calibrated model")


def original_field(feature_name):
    """'marital_status_Married' -> 'marital_status'; number features stay as they are."""
    for field in CATEGORY_FIELDS:
        if feature_name.startswith(field + "_"):
            return field
    return feature_name


# ------------------------------------------------------------------
# Two ways to get per-feature contributions
# ------------------------------------------------------------------

def shap_contributions(forest, features, model_name):
    import shap

    if model_name not in _explainers:
        _explainers[model_name] = shap.TreeExplainer(forest)
    explainer = _explainers[model_name]

    values = explainer.shap_values(features)
    expected = explainer.expected_value

    # Different shap versions return a list per class, or one array with a class axis
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]

    expected = np.asarray(expected).reshape(-1)
    base_value = float(expected[-1])
    return values, base_value


def class_one_share(tree, node):
    counts = tree.value[node][0]
    total = counts.sum()
    if total == 0:
        return 0.0
    return float(counts[1] / total)


def saabas_contributions(forest, features):
    """Contributions from the decision paths (used only when shap is not available)."""

    values = features.to_numpy(dtype=float)
    contributions = np.zeros(values.shape)
    base_total = 0.0

    for tree_model in forest.estimators_:
        tree = tree_model.tree_
        base_total += class_one_share(tree, 0)
        for row_number in range(values.shape[0]):
            node = 0
            while tree.children_left[node] != -1:
                feature = tree.feature[node]
                if values[row_number, feature] <= tree.threshold[node]:
                    next_node = tree.children_left[node]
                else:
                    next_node = tree.children_right[node]
                change = class_one_share(tree, next_node) - class_one_share(tree, node)
                contributions[row_number, feature] += change
                node = next_node

    tree_count = len(forest.estimators_)
    return contributions / tree_count, base_total / tree_count


def feature_contributions(rows, model_name=PRIMARY_MODEL):
    """Contribution of every ORIGINAL field to P(approve), for every row.

    Returns (list of {field: contribution} per row, base value, method name).
    """

    parts = load_model_parts(model_name)
    forest = underlying_forest(parts["model"])
    features = make_features(rows, parts["preprocessor"], parts["feature_names"])

    try:
        values, base_value = shap_contributions(forest, features, model_name)
        method = "SHAP TreeExplainer (Random Forest before calibration)"
    except ImportError:
        values, base_value = saabas_contributions(forest, features)
        method = "Saabas path contributions (shap not installed)"

    per_row = []
    for row_number in range(values.shape[0]):
        totals = {}
        for column_number, feature_name in enumerate(parts["feature_names"]):
            field = original_field(feature_name)
            totals[field] = totals.get(field, 0.0) + float(values[row_number, column_number])
        per_row.append(totals)

    return per_row, base_value, method


def shap_explanation(rows, model_name=PRIMARY_MODEL):
    """Step 6 for ONE applicant: the top 5 fields by size of their effect."""

    per_row, base_value, method = feature_contributions(rows, model_name)
    totals = per_row[0]

    ranked = sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)

    top_features = []
    for field, contribution in ranked[:TOP_COUNT]:
        value = rows.iloc[0][field]
        if isinstance(value, (float, np.floating)):
            value = round(float(value), 3) if field == "loan_to_income" else round(float(value), 2)
            if value == int(value) and field != "loan_to_income":
                value = int(value)
        if contribution > 0:
            direction = "raises approval"
        elif contribution < 0:
            direction = "lowers approval"
        else:
            direction = "no effect"
        top_features.append({
            "field": field,
            "label": FIELD_LABELS_WITH_RATIO.get(field, field),
            "applicant_value": value,
            "effect": direction,
            "contribution": round(contribution, 4),
        })

    return {
        "model": model_name,
        "method": method,
        "base_value": round(base_value, 4),
        "top_features": top_features,
    }
