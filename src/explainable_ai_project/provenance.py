"""The minimum provenance record, and the governance decision behind it.

WHY A JSON WHEN THERE IS ALREADY A GOVERNANCE LOG
-------------------------------------------------
`results/governance_log.csv` lists every data and model file with its md5,
which answers "did this file change". It does not answer the questions a
reviewer actually asks first: where did the data come from, who may use it
for what, which model is in production, what was it measured on, and who
decided the trade-off it embodies.

This writes one file that answers those, with the hashes carried through
from the log so the two cannot drift apart.

THE METRIC CONFLICT IS RECORDED AS A DECISION, NOT AS A RESULT
--------------------------------------------------------------
Model A satisfies predictive parity almost exactly (0.005) and fails
demographic parity completely (ratio 0.000). That is not a bug to be fixed;
Chouldechova and Kleinberg et al. showed in 2016-17 that the two cannot
both hold when base rates differ, and here they differ sharply. So the
project has to choose which one it holds itself to, say so, and say who
decided. A conflict that is only described is not governed.

Usage
-----
    uv run main.py provenance
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent

RESULTS_FOLDER = PROJECT_ROOT / "results"
TABLES_FOLDER = RESULTS_FOLDER / "tables"
DOCS_FOLDER = PROJECT_ROOT / "docs"

GOVERNANCE_LOG = RESULTS_FOLDER / "governance_log.csv"
SOURCES_CSV = PROJECT_ROOT / "rag_docs" / "sources.csv"

PROVENANCE_FILE = RESULTS_FOLDER / "provenance.json"
DECISION_FILE = DOCS_FOLDER / "governance_decision.md"

# Answered by the project team on 11.11.2026.
ACCOUNTABLE = "Michael Aichinger and Charles Anthony, jointly"
REVIEW = (
    "none. The project ran for about two weeks and ends at submission; no "
    "operational owner exists afterwards and no review is scheduled. Naming "
    "a future date would imply a responsibility nobody holds. In a "
    "deployment this row would carry a named risk owner and a recurring "
    "review, and its absence here is itself a governance finding."
)


def hash_of(path):
    if not path.exists():
        return None
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_log():
    if not GOVERNANCE_LOG.exists():
        return pd.DataFrame()
    return pd.read_csv(GOVERNANCE_LOG)


def entry_for(log, needle):
    """The governance-log row for a file, by the tail of its path."""
    if log.empty:
        return {}
    hits = log[log["file"].str.replace("\\", "/", regex=False).str.endswith(needle)]
    if hits.empty:
        return {}
    row = hits.iloc[-1]
    return {"md5": row["md5"], "size_kb": float(row["size_kb"]),
            "recorded_on": row["date"], "step": row["step"]}


def build_provenance():
    log = read_log()

    datasets = [
        {
            "id": "dataset_1",
            "role": "model A, the working model for the assistant",
            "origin": "course-supplied loan application data",
            "synthetic": True,
            "rows": 52000,
            "file": "datasets/root/loan_dataset_1.csv",
            "licence": "course material, not redistributable outside the course",
            "consent": "not applicable - no real applicants, no personal data",
            "known_issues": [
                "loan_to_income capped at 0.40, which is implausibly tight for "
                "real lending and bounds what the amount sweep can find",
                "applicants aged 61+ hold a median credit score of 438 against "
                "757 for the 31-45 group; the age gap in the model is mediated "
                "almost entirely by this",
            ],
        },
        {
            "id": "dataset_2",
            "role": "model B, the second opinion",
            "origin": "public Kaggle loan dataset",
            "synthetic": True,
            "rows": 20000,
            "file": "datasets/root/loan_dataset_2.csv",
            "licence": "as published on Kaggle; verify before any use beyond "
                       "this course",
            "consent": "not applicable - no real applicants, no personal data",
            "known_issues": [
                "RiskScore reaches a single-feature AUC of 0.9873 and was "
                "excluded as target leakage; it looks like the quantity the "
                "label was generated from",
                "the 61+ group holds 120 test rows, which is why four of its "
                "eight parity metrics come out undecided",
            ],
        },
    ]
    for one in datasets:
        one.update(entry_for(log, one["file"].split("/")[-1]))

    models = [
        {
            "id": "model_A",
            "file": "models/rf_calibrated_dataset_1_v1.joblib",
            "kind": "RandomForestClassifier, 400 trees, monotonic constraints "
                    "on loan amount and loan-to-income, wrapped in "
                    "CalibratedClassifierCV with isotonic regression",
            "trained_on": "dataset_1, training split only",
            "in_production": True,
        },
        {
            "id": "model_B",
            "file": "models/rf_calibrated_dataset_2_v1.joblib",
            "kind": "RandomForestClassifier, 200 trees, same constraints and "
                    "calibration",
            "trained_on": "dataset_2, training split only",
            "in_production": True,
        },
        {
            "id": "ebm_dataset_1",
            "file": "models/ebm_dataset_1_v1.joblib",
            "kind": "Explainable Boosting Machine",
            "trained_on": "dataset_1, training split only",
            "in_production": False,
            "note": "more accurate than model A on both datasets and NOT "
                    "chosen: the random forest takes monotonic constraints "
                    "and TreeSHAP computes exactly on it. The trade is "
                    "deliberate and documented in the model card.",
        },
    ]
    for one in models:
        one.update(entry_for(log, one["file"].split("/")[-1]))

    knowledge_base = []
    if SOURCES_CSV.exists():
        for _, row in pd.read_csv(SOURCES_CSV).iterrows():
            knowledge_base.append({
                "id": row["id"], "kind": row["kind"], "title": row["title"],
                "source": row["source"], "md5": row["md5"],
                "checked_on": row["checked_on"],
            })

    language_model = {
        "id": "qwen2.5:7b-instruct",
        "runs": "locally through Ollama on 127.0.0.1, never over the network",
        "weights": "downloaded from the Ollama registry, not fine-tuned",
        "settings": {"temperature": 0.0, "seed": 42, "num_ctx": 8192},
        "role": "wording only - it is given every figure it may state and is "
                "checked against that list afterwards; it never computes one",
        "data_sent_to_third_parties": "none",
    }

    documents = {}
    for name in ["model_card.md", "datasheet.md", "data_quality.md",
                 "fairness_report.md"]:
        path = DOCS_FOLDER / name
        documents[name] = {"md5": hash_of(path),
                           "present": path.exists(),
                           "indexed_for_retrieval": True}

    return {
        "record": "minimum provenance for the trustworthy loan assessment "
                  "assistant",
        "written_on": str(date.today()),
        "accountable": ACCOUNTABLE,
        "review": REVIEW,
        "personal_data": "none. Both datasets are synthetic; no natural person "
                         "is described, so GDPR data-subject rights do not "
                         "arise. The architecture would still have to be "
                         "re-examined before any real applicant data is used.",
        "datasets": datasets,
        "models": models,
        "language_model": language_model,
        "knowledge_base": knowledge_base,
        "generated_documents": documents,
        "hash_log": {
            "file": "results/governance_log.csv",
            "entries": int(len(read_log())),
            "covers": "every data and model file written by the pipeline",
        },
        "reproduce": {
            "command": "uv run main.py train",
            "seed": 42,
            "split": "70 / 15 / 15, stratified by approval and age group",
            "not_reproducible": [
                "isotonic calibration resampling is not seeded, so parity "
                "figures move in the third decimal between runs",
                "the language model is deterministic only while it stays "
                "loaded; a cold start changes the answer",
            ],
        },
    }


# ------------------------------------------------------------------
# The governance decision
# ------------------------------------------------------------------

DECISION = """# Governance decision: which fairness criterion this project holds itself to

Written on {today}. Accountable: {accountable}.

## The conflict, in numbers

Model A on dataset 1, cut-off 0.5, test split, 95 percent bootstrap intervals
from `results/tables/parity_table_with_intervals.csv`:

| Criterion | Value | Interval | Tolerance | Verdict |
|---|---|---|---|---|
| Predictive parity difference | 0.005 | 0.004 - 0.038 | at most 0.10 | within tolerance |
| Demographic parity ratio | 0.000 | 0.000 - 0.000 | at least 0.80 | breaches |
| Demographic parity difference | 0.834 | 0.822 - 0.845 | at most 0.10 | breaches |
| Equal opportunity difference | 0.966 | 0.959 - 0.973 | at most 0.10 | breaches |

The model is near-identical in precision across age groups and approves none
of the applicants aged 61 or over, of whom 15.4 percent were in fact approved.

This is not a defect to be repaired. Chouldechova (2017) and Kleinberg,
Mullainathan and Raghavan (2016) show that calibration, predictive parity and
equal odds cannot hold together when base rates differ between groups, and
here they differ sharply: the median credit score is 438 in the 61+ group
against 757 in the 31-45 group. Any model on this data must fail at least one
criterion. The project therefore has to choose which, and say so.

## The decision

**The project holds itself to equal opportunity and reports the breach rather
than removing it.**

Approving nobody in a group whose real approval rate is 15.4 percent is a
harm to individuals who would have been approved. Predictive parity holding
at the same time does not undo that; it only means the few approvals the
model does hand out are equally reliable across groups.

We did not repair the breach, for a reason the measurements support: removing
`age` from the model changes nothing, because the credit score carries the
same information (step 28). A fix that leaves the mechanism in place and only
hides the variable would make the metric look better while changing no
applicant's outcome.

Instead the breach is declared, routed and documented:

- every applicant in the 61+ group is deferred to a human, because the model's
  within-group AUC there is 0.524, which is chance;
- the fairness report naming this is part of the knowledge base the assistant
  retrieves, so the system can be asked about its own bias and answers from
  the measurement;
- the model card states that the model is not fit for use on applicants aged
  61 or over.

## Affected groups and error costs

| Group | n (test) | What the breach costs them |
|---|---|---|
| 61+ | 390 | Every applicant is refused automatically and sent to a human. Those who would have been approved - about 15 in 100 - depend entirely on that reviewer. |
| 46-60 | 1,533 | Approved below their real rate; the gap is small and inside the interval. |
| 18-30, 31-45 | 5,877 | No measured disadvantage. |

The cost is asymmetric. A false refusal denies credit to someone who would
have repaid; a false approval creates a loan that may default. Both are real,
they fall on different people, and this project does not have the data to
price them against each other. Saying so is more honest than inventing a
ratio.

## Tolerated gap

| Metric | Limit | Where it comes from |
|---|---|---|
| Demographic parity ratio | at least 0.80 | four-fifths rule, EEOC 29 CFR 1607.4D. A convention from US employment law with no standing in EU credit law, used as a warning line. |
| Demographic parity difference | at most 0.10 | **declared by this project.** No external authority. |
| Equal opportunity difference | at most 0.10 | **declared by this project.** No external authority. |
| Predictive parity difference | at most 0.10 | **declared by this project.** No external authority. |

A metric counts as within tolerance only when its whole confidence interval
sits inside the limit, and as a breach only when all of it sits outside.
Anything else is undecided at this sample size - which is the verdict for
four of dataset 2's eight metric-and-cut-off combinations, because its 61+
group holds 120 rows.

## Remediation trigger

Any of the following would require the model to be withdrawn from automatic
use for the affected group:

1. the demographic parity ratio interval moving below 0.80 for a group not
   already deferred;
2. within-group AUC falling below 0.60 for a group not already deferred, the
   same rule that currently catches 61+;
3. deferral exceeding 25 percent of all cases, at which point the automation
   no longer carries its own weight;
4. any use of real applicant data, which invalidates this record entirely -
   it rests throughout on both datasets being synthetic.

## Review

{review}

## What was rejected, and why

A second control was considered and **rejected on evidence**: deferring cases
where models A and B disagree by more than 0.5. The rule was tested on the
validation split and switched off, because the model is not worse on the
cases it catches. On dataset 1 it is better - 0.8551 against 0.8501 overall -
and on dataset 2 the drop of 0.0126 falls short of the 0.03 a rule must clear.
Switching it on would have sent 2,257 of about 7,800 validation cases to
human review for no measured gain.

The intuition that disagreement means difficulty is reasonable and, here,
wrong. `results/tables/deferral_rule_choice.csv` holds the figures.
"""


def run_provenance():
    print("Provenance record and governance decision")

    record = build_provenance()
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    PROVENANCE_FILE.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
    DECISION_FILE.write_text(
        DECISION.format(today=date.today(), accountable=ACCOUNTABLE,
                        review=REVIEW),
        encoding="utf-8")

    print("  datasets        :", len(record["datasets"]))
    print("  models          :", len(record["models"]),
          "(", sum(1 for one in record["models"] if one["in_production"]),
          "in production )")
    print("  knowledge base  :", len(record["knowledge_base"]), "sources")
    print("  hash log entries:", record["hash_log"]["entries"])

    missing = [name for name, one in record["generated_documents"].items()
               if not one["present"]]
    if missing:
        print("  MISSING documents:", ", ".join(missing))

    print()
    print("  written:", PROVENANCE_FILE.relative_to(PROJECT_ROOT))
    print("  written:", DECISION_FILE.relative_to(PROJECT_ROOT))
    return 0
