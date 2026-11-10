"""
Inference pathway, block 2A, step 4: models A and B.

The calibrated monotonic Random Forests from training (step 19) give
P(approve) for the applicant at the amount they asked for.

    model A   trained on dataset 1 (the course dataset)  - the main model
    model B   trained on dataset 2 (Kaggle)              - the second opinion

Both use the same input columns, so the same applicant goes through both.
The saved files are loaded once and kept in memory, because the amount sweep
(step 5) and SHAP (step 6) use them again.
"""

from explainable_ai_project.calibrate import load_calibrated_model
from explainable_ai_project.load_split import make_features
from explainable_ai_project.load_split import read_feature_names
from explainable_ai_project.load_split import read_preprocessor


MODEL_DATASETS = {"A": "dataset_1", "B": "dataset_2"}
PRIMARY_MODEL = "A"

_loaded = {}


def load_model_parts(model_name):
    """The calibrated model, preprocessor and feature names of model A or B (loaded once)."""

    if model_name not in _loaded:
        dataset_name = MODEL_DATASETS[model_name]
        _loaded[model_name] = {
            "dataset": dataset_name,
            "model": load_calibrated_model(dataset_name),
            "preprocessor": read_preprocessor(dataset_name),
            "feature_names": read_feature_names(dataset_name),
        }
    return _loaded[model_name]


def probability_of_approval(rows, model_name):
    """P(approve) for every row, from model A or B."""

    parts = load_model_parts(model_name)
    features = make_features(rows, parts["preprocessor"], parts["feature_names"])
    probability_table = parts["model"].predict_proba(features)
    return probability_table[:, 1]


def approval_probabilities(rows):
    """Step 4: P(approve) at the requested amount from both models, for ONE applicant."""

    result = {}
    for model_name in MODEL_DATASETS:
        probability = probability_of_approval(rows, model_name)[0]
        result[model_name] = round(float(probability), 4)
    return result
