"""
Training pathway, block 1B, step 15: monotonic Random Forest A.

Fixes the sweep problem from step 14. A monotonic constraint tells the
Random Forest one rule it must always follow:

    asking for a LARGER loan (or a larger loan compared to income)
    can never make approval MORE likely.

In scikit-learn this is the monotonic_cst setting: one number per feature,
    -1  the prediction may only go DOWN when this feature goes up
     0  no rule (the model learns freely)
    +1  the prediction may only go UP when this feature goes up
Here: -1 for loan_amount and loan_to_income, 0 for every other feature.

This step trains model A on dataset 1, scores it on validation, and repeats the
step 14 check to prove the curves no longer rise. The functions take a dataset
name, so step 17 reuses them for model B on dataset 2.

Put this file in:  src/explainable_ai_project/monotonic_forest.py

Writes:
    models/rf_monotonic_dataset_1_v1.joblib
    results/figures/monotonic_fix_dataset_1.png    unconstrained vs monotonic, side by side
    results/tables/model_results.csv               (adds the monotonic Random Forest row)
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from explainable_ai_project.load_split import load_split
from explainable_ai_project.load_split import read_feature_names
from explainable_ai_project.load_split import read_preprocessor
from explainable_ai_project.logistic_baseline import save_model_result
from explainable_ai_project.logistic_baseline import score_predictions
from explainable_ai_project.sweep_problem import amount_grid
from explainable_ai_project.sweep_problem import count_rising_curves
from explainable_ai_project.sweep_problem import probability_curves


# ------------------------------------------------------------------
# Where the files go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

MODELS_FOLDER = PROJECT_ROOT / "models"
FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"

SEED = 42
VERSION = "v1"
MODEL_NAME = "random_forest_monotonic"

# Features where "more" may only LOWER the approval probability
DECREASING_FEATURES = ["loan_amount", "loan_to_income"]

# Same settings as the unconstrained forest in step 14, so the comparison is fair.
# Step 16 tunes them.
DEFAULT_SETTINGS = {
    "n_estimators": 300,
    "min_samples_leaf": 20,
    "max_depth": None,
    "max_features": "sqrt",
}

APPLICANT_COUNT = 300
CURVES_TO_DRAW = 30


# ------------------------------------------------------------------
# The constraint
# ------------------------------------------------------------------

def make_monotonic_constraint(feature_names):
    """One number per feature: -1 for the decreasing features, 0 for the rest.

    The order must match the feature columns exactly, which is why it is built
    from the feature names saved in step 10.
    """

    constraint = []
    for name in feature_names:
        if name in DECREASING_FEATURES:
            constraint.append(-1)
        else:
            constraint.append(0)

    # Stop if a decreasing feature is missing: the constraint would silently do nothing
    for name in DECREASING_FEATURES:
        if name not in feature_names:
            raise ValueError("Feature " + name + " not found, so the constraint cannot be applied")

    return constraint


def build_monotonic_forest(feature_names, settings):
    constraint = make_monotonic_constraint(feature_names)

    model = RandomForestClassifier(
        n_estimators=settings["n_estimators"],
        min_samples_leaf=settings["min_samples_leaf"],
        max_depth=settings["max_depth"],
        max_features=settings["max_features"],
        monotonic_cst=constraint,
        n_jobs=-1,
        random_state=SEED,
    )
    return model


# ------------------------------------------------------------------
# Training and checking one model (used for A here and for B in step 17)
# ------------------------------------------------------------------

def monotonic_model_file(dataset_name):
    file_name = "rf_monotonic_" + dataset_name + "_" + VERSION + ".joblib"
    return MODELS_FOLDER / file_name


def train_monotonic_forest(dataset_name, settings):
    """Train on the training part, score on validation, save. Returns the model."""

    train = load_split(dataset_name, "train")
    validation = load_split(dataset_name, "validation")
    feature_names = list(train["features"].columns)

    model = build_monotonic_forest(feature_names, settings)
    model.fit(train["features"], train["labels"])

    probabilities = model.predict_proba(validation["features"])
    approval_probabilities = probabilities[:, 1]
    scores = score_predictions(validation["labels"], approval_probabilities)
    save_model_result(MODEL_NAME, dataset_name, "validation", scores)
    print("  validation:", scores)

    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    model_file = monotonic_model_file(dataset_name)
    joblib.dump(model, model_file)
    print("  saved", model_file.relative_to(PROJECT_ROOT))

    return model


def sweep_curves_for(model, dataset_name):
    """The step 14 check: change only the amount for 300 validation applicants."""

    train = load_split(dataset_name, "train")
    validation = load_split(dataset_name, "validation")
    preprocessor = read_preprocessor(dataset_name)
    feature_names = read_feature_names(dataset_name)

    sample_rows = validation["rows"].sample(n=APPLICANT_COUNT, random_state=SEED)
    grid = amount_grid(train["rows"])
    curve_table = probability_curves(model, sample_rows, grid, preprocessor, feature_names)
    return grid, curve_table


def draw_before_and_after(dataset_name, grid, before_curves, after_curves):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    panels = [
        (axes[0], before_curves, "Unconstrained (step 14)", "#C0392B"),
        (axes[1], after_curves, "Monotonic (step 15)", "#2F8F5B"),
    ]

    for axis, curve_table, title, colour in panels:
        for column_number in range(CURVES_TO_DRAW):
            single_curve = curve_table.iloc[:, column_number]
            axis.plot(grid, single_curve, color="#B8C2CC", linewidth=0.8)

        average_curve = curve_table.mean(axis=1)
        axis.plot(grid, average_curve, color=colour, linewidth=2.5, label="average of 300 applicants")

        axis.set_title(title)
        axis.set_xlabel("Requested loan amount (everything else unchanged)")
        axis.set_ylim(0, 1)
        axis.legend(loc="best")

    axes[0].set_ylabel("Predicted approval probability")
    figure.suptitle(dataset_name + ": the monotonic constraint stops approval rising with the amount")
    figure.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / ("monotonic_fix_" + dataset_name + ".png")
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    print("  saved", figure_file.relative_to(PROJECT_ROOT))


def train_and_check(dataset_name, settings):
    """Train the monotonic forest, then show the sweep is fixed."""

    print(dataset_name)

    model = train_monotonic_forest(dataset_name, settings)
    grid, after_curves = sweep_curves_for(model, dataset_name)

    share_ends_higher, share_rises_anywhere = count_rising_curves(after_curves)
    print("  applicants whose approval is higher at the largest amount:", share_ends_higher)
    print("  applicants whose approval rises at least once:", share_rises_anywhere)

    if share_rises_anywhere > 0:
        print("  WARNING: some curves still rise - check the constraint")
    else:
        print("  OK: no curve rises as the amount grows")

    # Before-and-after figure, using the unconstrained model saved in step 14
    unconstrained_file = MODELS_FOLDER / ("rf_unconstrained_" + dataset_name + "_" + VERSION + ".joblib")
    unconstrained_exists = unconstrained_file.exists()
    if unconstrained_exists:
        unconstrained_model = joblib.load(unconstrained_file)
        grid, before_curves = sweep_curves_for(unconstrained_model, dataset_name)
        draw_before_and_after(dataset_name, grid, before_curves, after_curves)
    else:
        print("  (run step 14 first to get the before-and-after figure)")

    print()
    return model


def train_monotonic_forest_a():
    """Step 15: model A on dataset 1."""

    model_a = train_and_check("dataset_1", DEFAULT_SETTINGS)
    return model_a