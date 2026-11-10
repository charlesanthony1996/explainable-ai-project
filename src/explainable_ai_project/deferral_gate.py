"""
Inference pathway, block 2A, step 7: deferral gate (Topic 3: deferral).

Flags the case for human review if any switched-on rule of training step 27
applies: near the threshold, unusual input, missing fields, models A and B
disagree, or an age group the model cannot rank reliably.

The rules and their settings are NOT chosen here: this calls apply_rules()
from deferral.py with the saved models/deferral_settings_dataset_1.json, so
inference uses exactly the rules that were chosen and evaluated in training.

One addition: rule 3 (missing data) is checked on the applicant's ORIGINAL
input. The models only ever see a filled-in copy (step 3), so without this the
gate could not know a field was empty.
"""

from explainable_ai_project.deferral import apply_rules
from explainable_ai_project.deferral import load_deferral_settings
from explainable_ai_project.models import MODEL_DATASETS
from explainable_ai_project.models import PRIMARY_MODEL


def reason_text(rule, settings, rows):
    if rule == "borderline":
        return ("P(approve) at the requested amount is within " + str(settings["borderline_margin"])
                + " of the " + str(settings["threshold"]) + " threshold, so a small change could flip the answer")
    if rule == "unusual_input":
        return "some values are outside the range the model saw in training (1st to 99th percentile)"
    if rule == "missing_data":
        return "required fields are missing, so the estimate used typical values instead"
    if rule == "disagreement":
        return ("models A and B, trained on different data, disagree by more than "
                + str(settings["disagreement_limit"]) + " in P(approve)")
    if rule == "weak_group":
        group = rows.iloc[0]["age_group"]
        return "the model cannot rank applicants reliably in the " + str(group) + " age group (validation AUC below 0.6)"
    return rule


def deferral_gate(rows, input_check):
    """Step 7 for ONE applicant.

    rows        the prepared one-row table (step 3)
    input_check the result of check_input (step 2)
    """

    dataset_name = MODEL_DATASETS[PRIMARY_MODEL]
    settings = load_deferral_settings(dataset_name)
    flags = apply_rules(rows, dataset_name, settings).iloc[0]

    enabled_rules = list(settings["enabled_rules"])
    rules_that_apply = []
    for rule in enabled_rules:
        if bool(flags[rule]):
            rules_that_apply.append(rule)

    # Missing data, judged on the original input (see the note at the top)
    if input_check["missing"] and "missing_data" not in rules_that_apply:
        rules_that_apply.append("missing_data")

    reasons = []
    for rule in rules_that_apply:
        reasons.append({"rule": rule, "text": reason_text(rule, settings, rows)})

    # Rules that would apply but are switched off (no evidence they help): reported, not used
    switched_off_but_true = []
    for rule in ["borderline", "disagreement"]:
        if rule not in enabled_rules and bool(flags[rule]):
            switched_off_but_true.append(rule)

    return {
        "deferred": len(reasons) > 0,
        "reasons": reasons,
        "enabled_rules": enabled_rules,
        "rules_true_but_switched_off": switched_off_but_true,
    }
