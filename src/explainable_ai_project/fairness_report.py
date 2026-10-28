"""
Training pathway, block 1C, step 29: fairness report (Topic 3).

Collects the results of steps 24 to 28 into one readable document:
    docs/fairness_report.md

The same file is copied to rag_docs/, so the assistant (Qwen) can retrieve
and cite it when a user asks about fairness.

Rules for this report:
    - AGGREGATED NUMBERS ONLY: rates and counts per age group, never a single
      applicant. The step checks this and stops if an applicant ID appears.
    - The key findings are WRITTEN FROM THE NUMBERS each time it runs,
      so the text always matches the current results.
    - Every section starts with a plain sentence saying what it covers,
      so a retrieved section makes sense on its own.

Put this file in:  src/explainable_ai_project/fairness_report.py

Reads (from steps 24 to 28, in results/tables/):
    fairness_by_age_group.csv, fairness_gaps.csv,
    max_amount_by_age_group.csv, max_amount_adjusted.csv,
    calibration_by_age_group.csv,
    deferral_rule_choice.csv, deferral_by_age_group.csv, deferral_evaluation.csv,
    intervention_remove_age.csv, intervention_by_age_group.csv

Writes:
    docs/fairness_report.md
    rag_docs/fairness_report.md
"""

import datetime
import re
import shutil
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
# Where everything is
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"
DOCS_FOLDER = PROJECT_ROOT / "docs"
RAG_DOCS_FOLDER = PROJECT_ROOT / "rag_docs"

REPORT_FILE = DOCS_FOLDER / "fairness_report.md"
RAG_COPY_FILE = RAG_DOCS_FOLDER / "fairness_report.md"

DATASET_NAMES = ["dataset_1", "dataset_2"]
MODEL_LABELS = {
    "dataset_1": "Model A (dataset 1)",
    "dataset_2": "Model B (dataset 2)",
}

# Applicant IDs look like D1-123 or D2-45. None may appear in the report.
APPLICANT_ID_PATTERN = r"\bD[12]-\d+\b"


# ==================================================================
# Helpers
# ==================================================================

def read_table(file_name):
    file_path = TABLES_FOLDER / file_name
    if not file_path.exists():
        print("  missing", file_name, "- run the step that makes it first")
        return None
    table = pd.read_csv(file_path)
    return table


def table_to_markdown(table):
    columns = list(table.columns)

    header_cells = []
    separator_cells = []
    for column in columns:
        header_cells.append(str(column))
        separator_cells.append("---")

    header = "| " + " | ".join(header_cells) + " |"
    separator = "| " + " | ".join(separator_cells) + " |"
    lines = [header, separator]

    for row in table.itertuples(index=False):
        cells = []
        for value in row:
            cells.append(str(value))
        line = "| " + " | ".join(cells) + " |"
        lines.append(line)

    text = "\n".join(lines)
    return text


def percent(value):
    """0.153 -> '15.3 %'"""
    number = round(value * 100, 1)
    text = str(number) + " %"
    return text


def rows_for(table, dataset_name, extra_filters=None):
    """Rows of one dataset, optionally filtered further: {"column": value}."""

    keep = table["dataset"] == dataset_name
    if extra_filters is not None:
        for column, value in extra_filters.items():
            matches = table[column] == value
            keep = keep & matches
    selected = table[keep]
    return selected


def value_for(table, dataset_name, filters, column):
    selected = rows_for(table, dataset_name, filters)
    if len(selected) == 0:
        return None
    value = selected[column].iloc[0]
    return value


def add_findings(lines, findings):
    lines.append("**Key findings:**")
    lines.append("")
    for finding in findings:
        lines.append("- " + finding)
    lines.append("")


def add_table(lines, title, table):
    lines.append("*" + title + "*")
    lines.append("")
    if table is None:
        lines.append("Not available: run the step that makes this table.")
    else:
        lines.append(table_to_markdown(table))
    lines.append("")


# ==================================================================
# Sections
# ==================================================================

def section_introduction(lines):
    lines.append("## 1. What this report covers")
    lines.append("")
    lines.append("This fairness report checks whether the loan assessment models treat applicants "
                 "of different ages differently. Age groups are 18-30, 31-45, 46-60 and 61+. "
                 "Model A was trained on dataset 1 (the course dataset), model B on dataset 2 "
                 "(Financial Risk for Loan Approval). All numbers come from the untouched test part "
                 "and are aggregated per age group; no individual applicant is shown. "
                 "Age is used for these checks only and is a model input in the original models.")
    lines.append("")


def section_approval_rates(lines):
    lines.append("## 2. Approval rates by age group")
    lines.append("")
    lines.append("This section compares, per age group, how often applicants were really approved "
                 "and how often each model would approve them.")
    lines.append("")

    groups = read_table("fairness_by_age_group.csv")
    gaps = read_table("fairness_gaps.csv")
    findings = []

    if groups is not None:
        for dataset_name in DATASET_NAMES:
            filters = {"age_group": "61+", "cut_off": 0.5}
            really = value_for(groups, dataset_name, filters, "actual_approval_rate")
            predicted = value_for(groups, dataset_name, filters, "predicted_approval_rate")
            if really is None:
                continue
            model_label = MODEL_LABELS[dataset_name]
            predicted_text = percent(predicted)
            really_text = percent(really)
            sentence = f"{model_label} approves {predicted_text} of applicants aged 61+ at cut-off 0.5; "
            sentence = sentence + f"{really_text} of them were really approved."
            findings.append(sentence)

    if gaps is not None:
        for dataset_name in DATASET_NAMES:
            filters = {"cut_off": 0.5}
            ratio = value_for(gaps, dataset_name, filters, "demographic_parity_ratio")
            difference = value_for(gaps, dataset_name, filters, "equal_opportunity_difference")
            if ratio is None:
                continue
            if ratio < 0.8:
                warning = " (below 0.8, a common warning sign)"
            else:
                warning = ""
            model_label = MODEL_LABELS[dataset_name]
            sentence = f"{model_label}: demographic parity ratio {ratio}{warning}, "
            sentence = sentence + f"equal opportunity difference {difference}."
            findings.append(sentence)

    add_findings(lines, findings)

    if groups is not None:
        columns = [
            "dataset", "cut_off", "age_group", "applicants",
            "actual_approval_rate", "predicted_approval_rate", "true_positive_rate", "precision",
        ]
        add_table(lines, "Per age group", groups[columns])
    add_table(lines, "Gaps between age groups", gaps)


def section_maximum_amount(lines):
    lines.append("## 3. Maximum loan amount by age group")
    lines.append("")
    lines.append("This section checks which age groups receive an estimated maximum loan amount "
                 "(approval probability at least 0.8), and whether income or credit score explain "
                 "the differences. 'Adjusted' rates assume every age group had the same mix of "
                 "incomes and credit scores.")
    lines.append("")

    summary = read_table("max_amount_by_age_group.csv")
    adjusted = read_table("max_amount_adjusted.csv")
    findings = []

    if adjusted is not None:
        for dataset_name in DATASET_NAMES:
            raw_gap = value_for(adjusted, dataset_name, {"adjustment": "not adjusted"}, "gap_between_groups")
            both_gap = value_for(adjusted, dataset_name, {"adjustment": "same income and credit score mix"},
                                 "gap_between_groups")
            income_gap = value_for(adjusted, dataset_name, {"adjustment": "same income mix"}, "gap_between_groups")
            if raw_gap is None:
                continue

            credit_gap = value_for(adjusted, dataset_name, {"adjustment": "same credit score mix"},
                                   "gap_between_groups")

            # Which adjustment shrinks the gap the most?
            candidate_gaps = {
                "income": income_gap,
                "credit score": credit_gap,
                "income and credit score": both_gap,
            }
            best_name = "income"
            smallest_gap = income_gap
            for name, gap in candidate_gaps.items():
                if gap < smallest_gap:
                    smallest_gap = gap
                    best_name = name

            explained_share = 0.0
            if raw_gap > 0:
                explained_share = (raw_gap - smallest_gap) / raw_gap
            explained_text = percent(explained_share)

            if explained_share >= 0.5:
                verdict = f"mostly explained by {best_name} (about {explained_text} of it)"
            else:
                verdict = f"NOT explained by income or credit score (at most {explained_text} of it)"

            model_label = MODEL_LABELS[dataset_name]
            sentence = f"{model_label}: the gap between age groups in getting an amount is {raw_gap}; "
            sentence = sentence + f"with the same income mix it is {income_gap}, "
            sentence = sentence + f"with the same income and credit score mix {both_gap}. "
            sentence = sentence + f"The gap is {verdict}."
            findings.append(sentence)

    add_findings(lines, findings)

    if summary is not None:
        columns = [
            "dataset", "age_group", "applicants", "share_getting_an_amount",
            "median_maximum", "median_income", "median_credit_score",
        ]
        add_table(lines, "Per age group", summary[columns])
    add_table(lines, "Adjusted rates", adjusted)


def section_calibration(lines):
    lines.append("## 4. Calibration by age group")
    lines.append("")
    lines.append("This section checks whether the approval probabilities are equally trustworthy "
                 "for every age group. The calibration gap is the average predicted probability "
                 "minus the share really approved; the 95 % interval comes from 1,000 bootstrap redraws.")
    lines.append("")

    calibration = read_table("calibration_by_age_group.csv")
    findings = []

    if calibration is not None:
        for row in calibration.itertuples(index=False):
            if row.age_group == "all":
                continue
            interval_excludes_zero = row.gap_95_lower > 0 or row.gap_95_upper < 0
            if interval_excludes_zero:
                if row.calibration_gap > 0:
                    direction = "too optimistic"
                else:
                    direction = "too pessimistic"
                model_label = MODEL_LABELS[row.dataset]
                sentence = f"{model_label} is slightly {direction} for age group {row.age_group} "
                sentence = sentence + f"(gap {row.calibration_gap}, 95 % interval "
                sentence = sentence + f"{row.gap_95_lower} to {row.gap_95_upper})."
                findings.append(sentence)

        if len(findings) == 0:
            findings.append("Both models are well calibrated in every age group: every 95 % interval includes 0.")
        else:
            findings.append("All other age groups are well calibrated (their 95 % intervals include 0).")

    add_findings(lines, findings)

    if calibration is not None:
        columns = [
            "dataset", "age_group", "applicants", "mean_predicted", "really_approved",
            "calibration_gap", "gap_95_lower", "gap_95_upper", "brier",
        ]
        add_table(lines, "Per age group", calibration[columns])


def section_deferral(lines):
    lines.append("## 5. Deferral to a human reviewer")
    lines.append("")
    lines.append("This section describes when the system sends a case to a human instead of answering. "
                 "Safety rules are always on: unusual input, missing data, and an age group the model "
                 "cannot rank reliably. Evidence rules (borderline probability, disagreement between "
                 "models A and B) are only switched on if the model is clearly less accurate on the "
                 "cases they catch.")
    lines.append("")

    choice = read_table("deferral_rule_choice.csv")
    by_group = read_table("deferral_by_age_group.csv")
    evaluation = read_table("deferral_evaluation.csv")
    findings = []

    if evaluation is not None:
        for row in evaluation.itertuples(index=False):
            model_label = MODEL_LABELS[row.dataset]
            deferred_text = percent(row.deferral_rate)
            kept_text = percent(row.accuracy_kept)
            deferred_accuracy_text = percent(row.accuracy_deferred)
            sentence = f"{model_label} sends {deferred_text} of applicants to a human. "
            sentence = sentence + f"Its accuracy is {kept_text} on the cases it decides "
            sentence = sentence + f"and {deferred_accuracy_text} on the cases it defers."
            findings.append(sentence)

    if by_group is not None:
        for dataset_name in DATASET_NAMES:
            rate = value_for(by_group, dataset_name, {"age_group": "61+"}, "deferral_rate")
            if rate is None:
                continue
            model_label = MODEL_LABELS[dataset_name]
            rate_text = percent(rate)
            sentence = f"{model_label} defers {rate_text} of applicants aged 61+."
            findings.append(sentence)

    add_findings(lines, findings)
    add_table(lines, "Evidence rules, chosen on the validation part", choice)
    add_table(lines, "Deferral rate per age group", by_group)
    add_table(lines, "Accuracy on decided and deferred cases", evaluation)


def section_intervention(lines):
    lines.append("## 6. Intervention: removing age from the model")
    lines.append("")
    lines.append("This section tests whether removing age as a model input reduces the differences "
                 "between age groups. Both models were retrained without age, with the same settings.")
    lines.append("")

    summary = read_table("intervention_remove_age.csv")
    by_group = read_table("intervention_by_age_group.csv")
    findings = []

    if summary is not None and by_group is not None:
        for dataset_name in DATASET_NAMES:
            auc_with = value_for(summary, dataset_name, {"version": "with age"}, "auc")
            auc_without = value_for(summary, dataset_name, {"version": "without age"}, "auc")
            ratio_with = value_for(summary, dataset_name, {"version": "with age"}, "demographic_parity_ratio_at_0.5")
            ratio_without = value_for(summary, dataset_name, {"version": "without age"},
                                      "demographic_parity_ratio_at_0.5")
            old_with = value_for(by_group, dataset_name, {"version": "with age", "age_group": "61+"},
                                 "share_getting_an_amount_at_0.8")
            old_without = value_for(by_group, dataset_name, {"version": "without age", "age_group": "61+"},
                                    "share_getting_an_amount_at_0.8")
            if auc_with is None:
                continue

            model_label = MODEL_LABELS[dataset_name]
            old_with_text = percent(old_with)
            old_without_text = percent(old_without)
            sentence = f"{model_label}: AUC {auc_with} with age, {auc_without} without; "
            sentence = sentence + f"demographic parity ratio {ratio_with} with age, {ratio_without} without; "
            sentence = sentence + f"applicants aged 61+ getting an amount: {old_with_text} with age, "
            sentence = sentence + f"{old_without_text} without."
            findings.append(sentence)

            change = old_without - old_with
            if abs(change) < 0.02:
                sentence = f"{model_label}: removing age did not change the outcome for applicants aged 61+, "
                sentence = sentence + "so the difference comes through other features linked to age, "
                sentence = sentence + "not from age itself."
                findings.append(sentence)

    add_findings(lines, findings)
    add_table(lines, "Accuracy and fairness, with and without age", summary)

    if by_group is not None:
        columns = [
            "dataset", "version", "age_group", "model_approves_at_0.5",
            "within_group_auc", "share_getting_an_amount_at_0.8",
        ]
        add_table(lines, "Per age group", by_group[columns])


def section_limitations(lines):
    lines.append("## 7. Limitations")
    lines.append("")
    lines.append("This section lists what the fairness checks cannot show.")
    lines.append("")
    lines.append("- Both datasets are synthetic, so the results describe these datasets, not real lending.")
    lines.append("- Only age was checked. Gender was nearly balanced in dataset 1 and is not in dataset 2; "
                 "combinations such as older applicants with low income were not checked separately.")
    lines.append("- The 61+ group is small in dataset 2 (about 120 test applicants), so its numbers have "
                 "wide intervals.")
    lines.append("- In dataset 1 there are no applicants aged 61+ with a good credit score, so the data "
                 "cannot show how the model would treat them.")
    lines.append("- Deferral sends a case to a human; it does not guarantee a fair human decision, "
                 "and older applicants are deferred more often.")
    lines.append("")


# ==================================================================
# The step
# ==================================================================

def check_no_applicant_ids(text):
    """Stop if the report contains anything that looks like an applicant ID."""

    found_ids = re.findall(APPLICANT_ID_PATTERN, text)
    if len(found_ids) > 0:
        message = "The fairness report contains applicant IDs: " + str(found_ids[:5])
        raise ValueError(message)


def write_fairness_report():
    """Step 29: write docs/fairness_report.md and copy it to rag_docs/."""

    now = datetime.datetime.now()
    date_text = now.strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append("# Fairness report: loan assessment models")
    lines.append("")
    lines.append("Written automatically on " + date_text + " from the results of the Topic 3 checks "
                 "(training steps 24 to 28). Aggregated numbers only.")
    lines.append("")

    section_introduction(lines)
    section_approval_rates(lines)
    section_maximum_amount(lines)
    section_calibration(lines)
    section_deferral(lines)
    section_intervention(lines)
    section_limitations(lines)

    text = "\n".join(lines)
    check_no_applicant_ids(text)

    DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(text, encoding="utf-8")
    print("Saved", REPORT_FILE.relative_to(PROJECT_ROOT))

    RAG_DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPORT_FILE, RAG_COPY_FILE)
    print("Copied to", RAG_COPY_FILE.relative_to(PROJECT_ROOT), "(for the RAG index)")

    return REPORT_FILE