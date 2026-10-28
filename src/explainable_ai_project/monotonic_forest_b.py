"""
Training pathway, block 1B, step 17: monotonic Random Forest B.

The same as model A (steps 15 and 16), but on dataset 2:
    1. tune: try the same 24 settings on dataset 2's validation part
    2. train model B with the best settings
    3. check that no applicant's approval rises as the loan amount grows,
       and draw the before-and-after figure (unconstrained vs monotonic)

Everything is reused from monotonic_forest.py and tune_forest.py,
so A and B are built in exactly the same way.

Put this file in:  src/explainable_ai_project/monotonic_forest_b.py

Writes:
    results/tables/tuning_log_dataset_2.csv
    models/best_settings_dataset_2.json
    models/rf_monotonic_dataset_2_v1.joblib
    results/figures/monotonic_fix_dataset_2.png
    results/tables/model_results.csv            (adds the dataset 2 row)
"""

from explainable_ai_project.monotonic_forest_a import train_and_check
from explainable_ai_project.tune_forest import tune_monotonic_forest


DATASET_NAME = "dataset_2"


def train_monotonic_forest_b():
    """Step 17: tune, train and check model B on dataset 2. Returns model B."""

    # 1. Tune on dataset 2's validation part
    best_settings = tune_monotonic_forest(DATASET_NAME)

    # 2 and 3. Train with the best settings, score, save, and check the sweep
    print("Training model B with the best settings")
    model_b = train_and_check(DATASET_NAME, best_settings)

    return model_b