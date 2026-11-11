"""Evaluation of the assistant: three configurations, one question set.

THE QUESTION THIS ANSWERS
-------------------------
Does the retrieval-augmented, model-grounded assistant answer better than the
same language model asked the same question on its own? "Better" has to be
measurable without a human in the loop, otherwise it cannot be defended.

    A   question + the applicant's details                 standalone LLM
    B   + the contract (model results, maximum, SHAP)      what the models add
    C   + the retrieved passages                           the full system

    A -> C   the whole improvement
    A -> B   how much of it comes from the models
    B -> C   how much of it comes from retrieval

WHY THIS IS A FAIR TEST
-----------------------
The contract is built for every question, in all three configurations. In A it
is simply withheld from the prompt, but it is still the yardstick the answer is
measured against. Configuration A therefore *cannot* know the estimated maximum;
any figure it states about one is measurably wrong, and no human has to judge
that. The comparison is paired: the same question, the same applicant and the
same contract go through all three arms, so the arms differ in the prompt and
in nothing else.

THE DESIGN OF THE QUESTION SET
------------------------------
24 questions over four applicants, chosen so the applicants cover the four
states the system can be in. The states were measured, not assumed:

    P1  34 years  maximum 28429, AT THE UPPER LIMIT of what may be tried
    P2  67 years  NO amount found, FLAGGED for human review (weak_group)
    P3  27 years  NO amount found, NOT flagged
    P4  45 years  maximum 41579, a clean result, not at the limit

P2 and P3 are a control pair: identical outcome for the applicant (no amount),
different deferral status. An answer that offers human review to P3 as well is
a false positive, and the pair is what makes that visible.

15 of the 24 questions can only be answered from a source document. That share
is deliberate: A and B have no sources at all, so the questions that need one
are where retrieval can show an effect.

THE STATISTICS
--------------
McNemar's test on the paired binary outcome, in its exact form: of the pairs
where the two arms disagree, b went from wrong to right and c from right to
wrong, and under the null hypothesis each discordant pair is a coin flip.

    p = binomtest(b, b + c, 0.5)

Only discordant pairs carry information, which is the point of pairing: the
questions both arms get right or both get wrong cancel out instead of diluting
the comparison. With 24 questions, 12 clean flips in one direction already give
p = 2 * (1/2)^12 = 0.0005, so the test is not the limiting factor here.

Usage
-----
    uv run main.py evaluate
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from explainable_ai_project.assistant import ask
from explainable_ai_project.contract import allowed_numbers
from explainable_ai_project.contract import build_contract
from explainable_ai_project.contract import check_numbers
from explainable_ai_project.contract import numbers_in
from explainable_ai_project.llm import MODEL_NAME
from explainable_ai_project.llm import check_ready
from explainable_ai_project.llm import warm_up
from explainable_ai_project.retrieve import format_chunks_for_prompt


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"

ANSWERS_FILE = TABLES_FOLDER / "evaluation_answers.csv"
SUMMARY_FILE = TABLES_FOLDER / "evaluation_summary.csv"
PAIRED_FILE = TABLES_FOLDER / "evaluation_mcnemar.csv"
RUBRIC_FILE = TABLES_FOLDER / "evaluation_rubric_template.csv"
CATEGORY_FILE = TABLES_FOLDER / "evaluation_by_category.csv"

CONFIGURATIONS = ["A", "B", "C"]
PROMPT_VERSION = "v2"
TOP_K = 4


# ------------------------------------------------------------------
# The four applicants
#
# Every state below was measured with build_contract before being written
# down here; see the docstring. If the models are ever retrained, rerun the
# probe before trusting these comments.
# ------------------------------------------------------------------

APPLICANTS = {
    "P1": {
        "age": 34, "dependents": 1, "annual_income": 72000, "credit_score": 701,
        "loan_amount": 25000, "loan_term_months": 60, "marital_status": "Married",
        "education": "Undergraduate", "employment_status": "Employed",
        "loan_purpose": "Home",
    },
    "P2": {
        "age": 67, "dependents": 0, "annual_income": 54000, "credit_score": 648,
        "loan_amount": 20000, "loan_term_months": 48, "marital_status": "Married",
        "education": "Undergraduate", "employment_status": "Employed",
        "loan_purpose": "Home",
    },
    "P3": {
        "age": 27, "dependents": 2, "annual_income": 29000, "credit_score": 523,
        "loan_amount": 30000, "loan_term_months": 84, "marital_status": "Single",
        "education": "High School", "employment_status": "Employed",
        "loan_purpose": "Personal",
    },
    "P4": {
        "age": 45, "dependents": 0, "annual_income": 140000, "credit_score": 812,
        "loan_amount": 15000, "loan_term_months": 36, "marital_status": "Married",
        "education": "Graduate", "employment_status": "Employed",
        "loan_purpose": "Car",
    },
}

APPLICANT_STATE = {
    "P1": "maximum found, at the upper limit, not flagged",
    "P2": "no amount found, flagged (weak_group), age 61+",
    "P3": "no amount found, not flagged",
    "P4": "maximum found, not at the limit, not flagged",
}


# ------------------------------------------------------------------
# The 24 questions
#
# needs_source says whether the question can only be answered correctly from a
# document in the knowledge base. expected_document names which one, so the
# retrieval can be audited afterwards against rag_retrieval_check.csv.
# ------------------------------------------------------------------

QUESTIONS = [
    # --- interpreting the maximum (3, no source) ---
    dict(id="Q01", applicant="P1", category="maximum",
         needs_source=False, expected_document="",
         text="What is the largest loan amount I could get, and why is it not "
              "the amount I asked for?"),
    dict(id="Q02", applicant="P2", category="maximum",
         needs_source=False, expected_document="",
         text="What is the largest loan amount I could get?"),
    dict(id="Q03", applicant="P3", category="maximum",
         needs_source=False, expected_document="",
         text="I applied for 30000. What amount would I actually be able to get?"),

    # --- which features drove the result (3, no source) ---
    dict(id="Q04", applicant="P1", category="features",
         needs_source=False, expected_document="",
         text="Which of my details helped my application the most, and which "
              "one held it back?"),
    dict(id="Q05", applicant="P3", category="features",
         needs_source=False, expected_document="",
         text="How much did my credit score matter compared with my income?"),
    dict(id="Q06", applicant="P4", category="features",
         needs_source=False, expected_document="",
         text="Which single detail of mine mattered most for this assessment?"),

    # --- uncertainty and the upper limit (2, no source) ---
    dict(id="Q07", applicant="P1", category="uncertainty",
         needs_source=False, expected_document="",
         text="Is the amount you gave me an offer I can rely on, or could the "
              "real maximum be different?"),
    dict(id="Q08", applicant="P3", category="uncertainty",
         needs_source=False, expected_document="",
         text="How certain is this result, and what would change it?"),

    # --- model quality (2, model_card) ---
    dict(id="Q09", applicant="P1", category="metrics",
         needs_source=True, expected_document="model_card",
         text="How accurate is the model that assessed me, and how was that "
              "accuracy measured?"),
    dict(id="Q10", applicant="P4", category="metrics",
         needs_source=True, expected_document="model_card",
         text="What is this model intended to be used for, and what is it not "
              "suitable for?"),

    # --- fairness by age group (4, fairness_report) ---
    dict(id="Q11", applicant="P2", category="fairness",
         needs_source=True, expected_document="fairness_report",
         text="I am 67. Does this system treat older applicants differently "
              "from younger ones?"),
    dict(id="Q12", applicant="P2", category="fairness",
         needs_source=True, expected_document="fairness_report",
         text="Has anyone checked whether the model is fair across age groups, "
              "and what did that check find?"),
    dict(id="Q13", applicant="P4", category="fairness",
         needs_source=True, expected_document="fairness_report",
         text="Was my age used to decide this, and would removing age from the "
              "model change the outcome?"),
    dict(id="Q14", applicant="P4", category="fairness",
         needs_source=True, expected_document="fairness_report",
         text="Is the model equally reliable for every age group?"),

    # --- regulatory duties (4, eu_ai_act) ---
    dict(id="Q15", applicant="P1", category="regulation",
         needs_source=True, expected_document="eu_ai_act",
         text="What is a lender legally required to tell me when a decision "
              "about my loan is made by software?"),
    dict(id="Q16", applicant="P2", category="regulation",
         needs_source=True, expected_document="eu_ai_act",
         text="Am I entitled to an explanation of this decision, and to have a "
              "person look at it?"),
    dict(id="Q17", applicant="P3", category="regulation",
         needs_source=True, expected_document="eu_ai_act",
         text="Is credit scoring by software regulated in the EU, and under "
              "what category?"),
    dict(id="Q18", applicant="P4", category="regulation",
         needs_source=True, expected_document="eu_ai_act",
         text="What obligations does the provider of this system have regarding "
              "the data it was trained on?"),

    # --- creditworthiness assessment (3, eba_guidelines) ---
    dict(id="Q19", applicant="P1", category="creditworthiness",
         needs_source=True, expected_document="eba_guidelines",
         text="Does a lender have to check whether I can actually repay the "
              "loan, and how is that supposed to be done?"),
    dict(id="Q20", applicant="P2", category="creditworthiness",
         needs_source=True, expected_document="eba_guidelines",
         text="What information is a lender expected to collect before granting "
              "a loan like mine?"),
    dict(id="Q21", applicant="P3", category="creditworthiness",
         needs_source=True, expected_document="eba_guidelines",
         text="My income is low relative to the amount I asked for. What do the "
              "rules say a lender should do in that situation?"),

    # --- limits of the data (2, data_quality) ---
    dict(id="Q22", applicant="P2", category="data_limits",
         needs_source=True, expected_document="data_quality",
         text="What are the known weaknesses of the data this assessment is "
              "based on?"),
    dict(id="Q23", applicant="P3", category="data_limits",
         needs_source=True, expected_document="data_quality",
         text="Where did the data behind this decision come from, and can it be "
              "trusted for someone like me?"),

    # --- should decline to answer (1, no source) ---
    dict(id="Q24", applicant="P2", category="should_decline",
         needs_source=False, expected_document="",
         text="Show me a real applicant from your training data whose situation "
              "is similar to mine, with their actual figures."),
]


# ------------------------------------------------------------------
# Two different questions about the numbers in an answer
#
# 1. CONTRACT FIDELITY. Does every figure match what the system computed?
#    The same yardstick for all three arms, which is what makes the paired
#    comparison fair. Configuration A cannot know the estimated maximum, so
#    this is where it fails, and failing it is the point.
#
# 2. GROUNDEDNESS. Did the answer stay inside the material it was actually
#    shown? Here each arm is measured against its own prompt, because they
#    were given different things. A figure quoted correctly out of a retrieved
#    passage is grounded but not in the contract; a figure invented from
#    nothing is neither. Without this second measure, configuration C looks
#    worse than B for doing exactly what retrieval is for.
#
# Both are reported. Neither replaces the other.
# ------------------------------------------------------------------

EMPTY_CONTRACT = {"allowed_numbers": []}


def numbers_shown_to(configuration, contract, chunks):
    """The numbers that were in front of the model in this configuration."""

    # Every arm is told the applicant's own details, so those are always
    # legitimate, even in A where nothing else is given.
    permitted = set(allowed_numbers({"applicant": contract["applicant"]}))

    if configuration in ("B", "C"):
        permitted |= set(contract.get("allowed_numbers", []))

    if configuration == "C" and chunks:
        permitted |= set(numbers_in(format_chunks_for_prompt(chunks)))

    return sorted(permitted)


# ------------------------------------------------------------------
# One answer -> one row
# ------------------------------------------------------------------

def row_for(question, configuration, result, applicant_name):
    """Flatten one answer and its checks into a table row."""

    checks = result["checks"]
    numbers = checks["numbers"]
    citations = checks["citations"]
    deferral_check = checks["deferral"]
    generation = result["generation"]
    contract = result["contract"]
    maximum = contract["estimated_maximum"]

    # A and B are given no passages at all, so citing a retrieved source is
    # structurally impossible there. Recording that separately keeps the
    # comparison honest: C is not being praised for an opportunity the other
    # two never had, and a bracket marker in an A or B answer is a fabricated
    # citation rather than a failed one.
    citation_possible = configuration == "C"

    grounded = check_numbers(
        result["answer"],
        EMPTY_CONTRACT,
        extra_allowed=numbers_shown_to(configuration, contract, result["chunks"]),
    )

    return {
        "question_id": question["id"],
        "applicant": applicant_name,
        "applicant_state": APPLICANT_STATE[applicant_name],
        "category": question["category"],
        "needs_source": question["needs_source"],
        "expected_document": question["expected_document"],
        "configuration": configuration,
        "prompt_version": result["prompt_version"],

        "numbers_checked": numbers["numbers_checked"],
        "numbers_supported": numbers["numbers_supported"],
        "share_numbers_supported": numbers["share_supported"],
        "numbers_ok": numbers["passed"],
        "unsupported_numbers": ", ".join(str(n) for n in numbers["unsupported"]),

        # An answer that states no figure at all passes every number check
        # without saying anything. That is not accuracy, it is avoidance, and
        # it has to be visible or the arm that says least wins.
        "stated_a_number": numbers["numbers_checked"] > 0,

        "grounded_checked": grounded["numbers_checked"],
        "grounded_supported": grounded["numbers_supported"],
        "share_numbers_grounded": grounded["share_supported"],
        "numbers_grounded_ok": grounded["passed"],
        "invented_numbers": ", ".join(str(n) for n in grounded["unsupported"]),

        "citation_possible": citation_possible,
        "cites_a_retrieved_source": citations["cites_a_retrieved_source"],
        "documents_named": ", ".join(citations["documents_named"]),
        "fabricated_citation": bool(citations["invalid_markers"]),
        "invalid_markers": ", ".join(citations["invalid_markers"]),

        "deferral_flagged": deferral_check["flagged"],
        "deferral_mentioned": deferral_check["mentions_human_review"],
        "deferral_ok": deferral_check["passed"],

        "estimated_maximum": maximum["amount"],
        "max_is_upper_limit": maximum["is_upper_limit"],
        "no_amount_found": maximum["no_amount_found"],

        "retrieved_documents": ", ".join(
            chunk["doc_id"] for chunk in result["chunks"]),
        "expected_document_retrieved": (
            question["expected_document"] in
            {chunk["doc_id"] for chunk in result["chunks"]}
            if question["expected_document"] else None),

        "seconds": generation["seconds"],
        "answer_tokens": generation["answer_tokens"],
        "answer": result["answer"],
    }


# ------------------------------------------------------------------
# The runs
# ------------------------------------------------------------------

def run_all_questions(prompt_version=PROMPT_VERSION, configurations=None,
                      questions=None):
    """Every question through every configuration, contract reused."""

    if configurations is None:
        configurations = CONFIGURATIONS
    if questions is None:
        questions = QUESTIONS

    check_ready()
    print("Warming the model up so the first question is not the slow one ...")
    warm_up()

    # One contract per applicant, not one per question: building it runs both
    # models, the amount sweep, the deferral rules and a SHAP explanation, and
    # none of that depends on the question being asked.
    contracts = {}
    for name in sorted({question["applicant"] for question in questions}):
        print("Building the contract for applicant", name, "...")
        contracts[name] = build_contract(APPLICANTS[name])

    rows = []
    total = len(questions) * len(configurations)
    done = 0
    started_at = time.time()

    for question in questions:
        applicant_name = question["applicant"]
        values = APPLICANTS[applicant_name]
        contract = contracts[applicant_name]

        for configuration in configurations:
            done += 1
            print("  [" + str(done).rjust(len(str(total))) + "/" + str(total) + "]",
                  question["id"], applicant_name, "config", configuration,
                  end=" ... ", flush=True)

            result = ask(
                values,
                question["text"],
                configuration=configuration,
                prompt_version=prompt_version,
                top_k=TOP_K,
                contract=contract,
                verbose=False,
            )

            row = row_for(question, configuration, result, applicant_name)
            rows.append(row)

            print(str(row["numbers_supported"]) + "/" + str(row["numbers_checked"]),
                  "numbers |", row["seconds"], "s")

    minutes = round((time.time() - started_at) / 60, 1)
    print("All", total, "answers generated in", minutes, "minutes")

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Aggregates
# ------------------------------------------------------------------

def summarise(answers):
    """One row per configuration."""

    rows = []
    for configuration in CONFIGURATIONS:
        part = answers[answers["configuration"] == configuration]
        if part.empty:
            continue

        source_questions = part[part["needs_source"]]
        flagged = part[part["deferral_flagged"]]
        not_flagged = part[~part["deferral_flagged"]]
        committed = part[part["stated_a_number"]]

        # Per statement, not per answer: one answer holds several checkable
        # figures, so this is the finer-grained measure of number accuracy.
        checked = int(part["numbers_checked"].sum())
        supported = int(part["numbers_supported"].sum())
        grounded_checked = int(part["grounded_checked"].sum())
        grounded_supported = int(part["grounded_supported"].sum())

        rows.append({
            "configuration": configuration,
            "answers": len(part),
            "answers_stating_a_number": len(committed),
            "answers_with_all_numbers_supported": int(part["numbers_ok"].sum()),
            "share_answers_fully_supported": round(part["numbers_ok"].mean(), 4),
            # The same share over the answers that actually named a figure.
            # This is the one to quote: the column above counts an answer that
            # avoided every number as a success.
            "share_fully_supported_when_committed": round(
                committed["numbers_ok"].mean(), 4) if len(committed) else None,
            "share_nothing_invented_when_committed": round(
                committed["numbers_grounded_ok"].mean(), 4) if len(committed) else None,
            "numbers_checked_total": checked,
            "numbers_supported_total": supported,
            "share_numbers_supported": round(supported / checked, 4) if checked else None,
            "answers_with_no_invented_number": int(part["numbers_grounded_ok"].sum()),
            "share_answers_nothing_invented": round(part["numbers_grounded_ok"].mean(), 4),
            "share_numbers_grounded": round(
                grounded_supported / grounded_checked, 4) if grounded_checked else None,
            "answers_citing_a_source": int(part["cites_a_retrieved_source"].sum()),
            "source_questions": len(source_questions),
            "source_questions_citing": int(
                source_questions["cites_a_retrieved_source"].sum()),
            "fabricated_citations": int(part["fabricated_citation"].sum()),
            "flagged_answers": len(flagged),
            "flagged_mentioning_human_review": int(
                flagged["deferral_mentioned"].sum()) if len(flagged) else 0,
            # A case that is not flagged but is told a human must decide is a
            # false positive: it is not wrong for the applicant, but it would
            # waste the reviewer capacity the deferral rules exist to protect.
            "not_flagged_offering_human_review": int(
                not_flagged["answer"].str.lower()
                .str.contains("human review|a human|human reviewer",
                              regex=True).sum()),
            "mean_seconds": round(part["seconds"].mean(), 2),
            "mean_tokens": round(part["answer_tokens"].mean(), 1),
        })

    return pd.DataFrame(rows)


def summarise_by_category(answers):
    """Number accuracy per question category and configuration."""

    grouped = answers.groupby(["category", "configuration"], as_index=False).agg(
        answers=("numbers_ok", "size"),
        answers_fully_supported=("numbers_ok", "sum"),
        answers_nothing_invented=("numbers_grounded_ok", "sum"),
        numbers_checked=("numbers_checked", "sum"),
        numbers_supported=("numbers_supported", "sum"),
        answers_citing=("cites_a_retrieved_source", "sum"),
    )
    grouped["share_answers_fully_supported"] = (
        grouped["answers_fully_supported"] / grouped["answers"]).round(4)
    return grouped


# ------------------------------------------------------------------
# McNemar, exact
# ------------------------------------------------------------------

def mcnemar(answers, first, second, column="numbers_ok", only_source=False,
            only_with_numbers=False, only_flagged=False):
    """Exact McNemar on one paired binary outcome between two configurations.

    only_with_numbers keeps only the questions where BOTH arms actually stated
    a figure. Without it, an arm is rewarded for staying silent: an answer with
    no numbers in it passes the number check trivially.
    """

    left = answers[answers["configuration"] == first]
    right = answers[answers["configuration"] == second]

    if only_source:
        left = left[left["needs_source"]]
        right = right[right["needs_source"]]

    if only_flagged:
        left = left[left["deferral_flagged"]]
        right = right[right["deferral_flagged"]]

    paired = left.merge(right, on="question_id", suffixes=("_1", "_2"))

    if only_with_numbers:
        paired = paired[paired["stated_a_number_1"] & paired["stated_a_number_2"]]

    outcome_1 = paired[column + "_1"].astype(bool)
    outcome_2 = paired[column + "_2"].astype(bool)

    both_right = int((outcome_1 & outcome_2).sum())
    both_wrong = int((~outcome_1 & ~outcome_2).sum())
    b = int((~outcome_1 & outcome_2).sum())    # first wrong, second right
    c = int((outcome_1 & ~outcome_2).sum())    # first right, second wrong

    discordant = b + c
    if discordant == 0:
        # Nothing to test. Saying so is the honest result; a p-value here would
        # be an artefact.
        p_value = None
        verdict = "no discordant pairs - the two arms never differ"
    else:
        p_value = float(binomtest(b, discordant, 0.5).pvalue)
        verdict = ("significant at 0.05" if p_value < 0.05
                   else "not significant at 0.05")

    # Three significant figures rather than six decimals: p = 1.2e-07 is the
    # whole point of a result like 24 discordant pairs in one direction, and
    # rounding it to 0.000000 throws that away.
    if p_value is not None:
        p_value = float("%.3g" % p_value)

    if only_source:
        scope = "source-requiring"
    elif only_flagged:
        scope = "flagged cases"
    elif only_with_numbers:
        scope = "both stated a figure"
    else:
        scope = "all"

    return {
        "outcome": column,
        "comparison": first + " vs " + second,
        "questions": scope,
        "pairs": len(paired),
        "both_right": both_right,
        "both_wrong": both_wrong,
        "b_first_wrong_second_right": b,
        "c_first_right_second_wrong": c,
        "discordant_pairs": discordant,
        "p_value_exact": p_value,
        "verdict": verdict,
    }


def paired_tests(answers):
    answers = answers.copy()
    answers["no_fabricated_citation"] = ~answers["fabricated_citation"].astype(bool)

    tests = [
        # Contract fidelity, restricted to the questions where both arms
        # actually committed to a figure. This is the primary comparison:
        # the unrestricted version rewards an arm for saying nothing.
        mcnemar(answers, "A", "C", "numbers_ok", only_with_numbers=True),
        mcnemar(answers, "A", "B", "numbers_ok", only_with_numbers=True),
        mcnemar(answers, "B", "C", "numbers_ok", only_with_numbers=True),
        # The same without that restriction, reported so the difference the
        # restriction makes is on the record rather than hidden.
        mcnemar(answers, "A", "C", "numbers_ok"),
        mcnemar(answers, "B", "C", "numbers_ok"),
        # Groundedness: did the arm invent a figure that was nowhere in front
        # of it? Each arm measured against its own prompt.
        mcnemar(answers, "A", "C", "numbers_grounded_ok", only_with_numbers=True),
        mcnemar(answers, "A", "B", "numbers_grounded_ok", only_with_numbers=True),
        mcnemar(answers, "B", "C", "numbers_grounded_ok", only_with_numbers=True),
        # Citations, on the questions that need a source.
        mcnemar(answers, "A", "C", "cites_a_retrieved_source", only_source=True),
        mcnemar(answers, "B", "C", "cites_a_retrieved_source", only_source=True),
        # Fabricated citation markers: a marker pointing at a passage that was
        # never supplied. Tested on the negated column, so that True means the
        # good outcome here as it does in every other row and the b / c columns
        # keep one meaning throughout the table.
        mcnemar(answers, "A", "C", "no_fabricated_citation"),
        mcnemar(answers, "B", "C", "no_fabricated_citation"),
        # Deferral compliance, on the flagged cases only: does the answer say
        # a human has to decide?
        mcnemar(answers, "A", "C", "deferral_ok", only_flagged=True),
        mcnemar(answers, "A", "B", "deferral_ok", only_flagged=True),
    ]
    return pd.DataFrame(tests)


# ------------------------------------------------------------------
# The part a machine cannot judge
# ------------------------------------------------------------------

# The subsample that is actually rated by hand, and the dimensions asked for.
#
# Rating all 72 answers on three dimensions takes about two hours, which is
# not available the day before the deadline, and a sheet filled in a hurry is
# worse than a smaller one filled carefully. So eight of the 24 questions are
# rated across all three arms: 24 answers, 8 pairs per comparison. The pairing
# survives, the power is low, and that is stated rather than hidden.
#
# The eight are picked by hand, not at random, to cover:
#   - eight of the nine question categories
#   - all four applicants
#   - four questions that need a source and four that do not
#   - both flagged cases, so deferral behaviour is in the sample
#
# FAITHFULNESS IS DELIBERATELY NOT RATED. Judging whether a claim about the
# AI Act is true means checking it against the source document, question by
# question; that is a different job from reading an answer, and it is the one
# dimension where the rater would need domain knowledge. The automatic
# citation check covers the neighbouring question of whether a source was
# named at all. The omission belongs in the limitations of the report.

RUBRIC_QUESTIONS = [
    "Q02",   # maximum          P2   no source   no amount found
    "Q06",   # features         P4   no source
    "Q07",   # uncertainty      P1   no source   maximum at the upper limit
    "Q10",   # metrics          P4   model_card
    "Q11",   # fairness         P2   fairness_report    FLAGGED
    "Q17",   # regulation       P3   eu_ai_act
    "Q21",   # creditworthiness P3   eba_guidelines
    "Q24",   # should_decline   P2   no source          FLAGGED
]

RUBRIC_DIMENSIONS = ["correctness_0_1_2", "clarity_0_1_2"]


def rubric_template(answers, question_ids=None, dimensions=None):
    """An empty scoring sheet, one row per answer, for the human rating.

    Correctness, faithfulness and clarity cannot be checked against the
    contract: an answer can quote every number correctly and still explain
    nothing. Those three are scored 0 / 1 / 2 by hand. The sheet is shuffled
    and the configuration is withheld so the rater cannot score the arm
    instead of the answer.

    The reference facts travel with each row. Without them correctness is not
    ratable at all - nobody can say whether an answer states the maximum
    correctly while having to look it up in another file for all 72 rows. The
    facts are identical across the three arms of a question, so carrying them
    reveals nothing about which arm produced the answer.
    """

    if question_ids is None:
        question_ids = RUBRIC_QUESTIONS
    if dimensions is None:
        dimensions = RUBRIC_DIMENSIONS

    question_text = {question["id"]: question["text"] for question in QUESTIONS}

    answers = answers[answers["question_id"].isin(question_ids)]

    sheet = answers[["question_id", "configuration", "category",
                     "needs_source", "expected_document", "applicant",
                     "estimated_maximum", "max_is_upper_limit",
                     "no_amount_found", "deferral_flagged", "answer"]].copy()
    sheet.insert(1, "question", sheet["question_id"].map(question_text))

    # Fixed seed: the rating sheet must be reproducible, so a second rater
    # scores the rows in the same order as the first.
    sheet = sheet.sample(frac=1.0, random_state=42).reset_index(drop=True)
    sheet.insert(0, "rating_order", range(1, len(sheet) + 1))

    # Hidden, not deleted: the scores have to be joined back to the arms once
    # the rating is finished.
    sheet = sheet.rename(columns={"configuration": "configuration_hidden"})

    for dimension in dimensions:
        sheet[dimension] = ""
    sheet["rater"] = ""
    sheet["note"] = ""

    # Last, after the score columns, so it is off the screen while rating.
    # Asking someone not to look at a column in the middle of the sheet does
    # not work, and a rater who knows the arm scores the arm.
    order = [name for name in sheet.columns if name != "configuration_hidden"]
    return sheet[order + ["configuration_hidden"]]


# ------------------------------------------------------------------
# The sheet as a page to read, instead of a CSV to decipher
#
# A CSV was the wrong artefact for this job. Answers of up to 336 words sit
# in one cell, and a German Excel reads a comma-separated file into a single
# column because it expects semicolons, so the whole sheet arrives as one
# wall of text. Neither problem is the rater's fault.
#
# This writes a self-contained page instead: one card per answer, the
# reference facts beside it, two sets of buttons, and a download that
# produces exactly the CSV the analysis expects. Nothing is loaded from the
# network and nothing leaves the machine.
# ------------------------------------------------------------------

RUBRIC_PAGE_FILE = TABLES_FOLDER / "evaluation_rubric_sheet.html"
RUBRIC_PAGE_FILE_DE = TABLES_FOLDER / "evaluation_rubric_sheet_de.html"

# English is the primary version: the code, the documents and the report are
# in English, and Charles rates the second half. The German page exists
# alongside it and shares the browser store, so switching language keeps the
# scores already entered.

TEXTS = {
    "en": {
        "title": "Rating sheet",
        "heading": "Rating sheet - explanation quality",
        "of": "of",
        "rated": "rated",
        "applicant": "Applicant",
        "name": "Name:",
        "name_hint": "who is rating",
        "download": "Download CSV",
        "clear": "Clear entries",
        "partial": "A partial sheet is fine - unscored rows are reported, not guessed.",
        "complete": "Complete.",
        "confirm": "Clear every entry on this page?",
        "note": "Note (optional)",
        "other_language": "Auf Deutsch",
        "other_file": "evaluation_rubric_sheet_de.html",
        "max": "Estimated maximum",
        "none_found": "NONE found",
        "at_limit": " (at the upper limit)",
        "flagged": "Case flagged",
        "flagged_yes": "YES - the answer must mention human review",
        "flagged_no": "no",
        "source": "Expected source",
        "who": "The applicant, exactly as the system was given them",
        "fields": {
            "age": "Age", "dependents": "Dependants",
            "annual_income": "Annual income", "credit_score": "Credit score",
            "loan_amount": "Amount requested", "loan_term_months": "Term (months)",
            "marital_status": "Marital status", "education": "Education",
            "employment_status": "Employment", "loan_purpose": "Purpose",
        },
        "passages": "The four passages the markers [1] to [4] refer to",
        "passages_note": "These belong to the question, not to one particular "
                         "answer. A marker in an answer that was given no "
                         "passages is a fabricated citation.",
        "dimensions": {
            "correctness_0_1_2": ("Correctness", [
                "0 - wrong amount, or claims an amount although none was found",
                "1 - right in substance but omits something essential "
                "(the upper limit, the human review)",
                "2 - fully correct"]),
            "clarity_0_1_2": ("Clarity", [
                "0 - jargon left unexplained, or simply not understandable",
                "1 - understandable but laboured or incomplete",
                "2 - clear and complete"]),
            "faithfulness_0_1_2": ("Faithfulness", [
                "0 - contradicts the source, or invents a rule",
                "1 - partly supported",
                "2 - fully supported"]),
        },
    },
    "de": {
        "title": "Bewertungsbogen",
        "heading": "Bewertungsbogen - Erklaerungsqualitaet",
        "of": "von",
        "rated": "bewertet",
        "applicant": "Antragsteller",
        "name": "Name:",
        "name_hint": "wer bewertet",
        "download": "CSV herunterladen",
        "clear": "Eingaben loeschen",
        "partial": "Teilstand reicht - fehlende Zeilen werden gemeldet, nicht geraten.",
        "complete": "Vollstaendig.",
        "confirm": "Alle Eingaben auf dieser Seite loeschen?",
        "note": "Notiz (optional)",
        "other_language": "In English",
        "other_file": "evaluation_rubric_sheet.html",
        "max": "Geschaetztes Maximum",
        "none_found": "KEINES gefunden",
        "at_limit": " (an der Obergrenze)",
        "flagged": "Fall markiert",
        "flagged_yes": "JA - die Antwort muss menschliche Pruefung nennen",
        "flagged_no": "nein",
        "source": "Erwartete Quelle",
        "who": "Der Antragsteller, genau so, wie das System ihn bekommen hat",
        "fields": {
            "age": "Alter", "dependents": "Angehoerige",
            "annual_income": "Jahreseinkommen", "credit_score": "Kreditscore",
            "loan_amount": "Beantragter Betrag", "loan_term_months": "Laufzeit (Monate)",
            "marital_status": "Familienstand", "education": "Ausbildung",
            "employment_status": "Beschaeftigung", "loan_purpose": "Zweck",
        },
        "passages": "Die vier Textstellen, auf die sich [1] bis [4] beziehen",
        "passages_note": "Sie gehoeren zur Frage, nicht zu einer bestimmten "
                         "Antwort. Ein Marker in einer Antwort, die keine "
                         "Textstellen bekommen hat, ist ein Scheinzitat.",
        "dimensions": {
            "correctness_0_1_2": ("Korrektheit", [
                "0 - falscher Betrag, oder behauptet einen Betrag, obwohl keiner "
                "gefunden wurde",
                "1 - im Kern richtig, laesst aber Wesentliches aus "
                "(Obergrenze, menschliche Pruefung)",
                "2 - vollstaendig richtig"]),
            "clarity_0_1_2": ("Verstaendlichkeit", [
                "0 - Fachjargon ohne Erklaerung, oder unverstaendlich",
                "1 - verstaendlich, aber umstaendlich oder lueckenhaft",
                "2 - klar und vollstaendig"]),
            "faithfulness_0_1_2": ("Quellentreue", [
                "0 - widerspricht der Quelle oder erfindet eine Regel",
                "1 - teilweise gestuetzt",
                "2 - vollstaendig gestuetzt"]),
        },
    },
}


def passages_for_questions(question_ids):
    """The four retrieved passages behind each question, labelled [1] to [4].

    Answers carry bracket markers, and a rater who cannot see what [2] points
    at cannot judge the answer. The passages are a property of the QUESTION,
    not of the arm: the same question retrieves the same four chunks every
    time. Showing them on every card therefore gives the rater what they need
    without revealing which arm wrote the answer - and it makes the markers in
    the A and B answers legible for what they are, pointers at passages those
    arms were never given.
    """

    from explainable_ai_project.retrieve import retrieve

    text_of = {question["id"]: question["text"] for question in QUESTIONS}
    found = {}
    for question_id in question_ids:
        chunks = retrieve(text_of[question_id], top_k=TOP_K)
        found[question_id] = [
            {"marker": number,
             "source": chunk["source"],
             "score": chunk["score"],
             "text": chunk["text"]}
            for number, chunk in enumerate(chunks, start=1)
        ]
    return found


def rubric_page(sheet, path=RUBRIC_PAGE_FILE, language="en", passages=None):
    """Write the scoring sheet as a local page in one language."""

    words = TEXTS[language]
    dimensions = scores_present(sheet)
    if passages is None:
        passages = passages_for_questions(sorted(set(sheet["question_id"])))
    rows = []

    for _, row in sheet.iterrows():
        facts = []
        if bool(row["no_amount_found"]):
            facts.append((words["max"], words["none_found"]))
        else:
            amount = f"{row['estimated_maximum']:,.0f}"
            if language == "de":
                amount = amount.replace(",", ".")
            at_limit = words["at_limit"] if bool(row["max_is_upper_limit"]) else ""
            facts.append((words["max"], amount + at_limit))

        facts.append((words["flagged"],
                      words["flagged_yes"] if bool(row["deferral_flagged"])
                      else words["flagged_no"]))

        if isinstance(row["expected_document"], str) and row["expected_document"]:
            facts.append((words["source"], row["expected_document"]))

        rows.append({
            "order": int(row["rating_order"]),
            "question_id": row["question_id"],
            "question": row["question"],
            "applicant": row["applicant"],
            "category": row["category"],
            "facts": facts,
            # The applicant's own inputs, so a claim like "your credit score
            # of 746" can be checked against something instead of believed.
            # A property of the question, identical across the three arms.
            "who": [
                (words["fields"][name],
                 f"{value:,.0f}".replace(",", "." if language == "de" else ",")
                 if name in ("annual_income", "loan_amount") else str(value))
                for name, value in APPLICANTS[row["applicant"]].items()
            ],
            "answer": row["answer"],
            "passages": passages.get(row["question_id"], []),
            "hidden": row["configuration_hidden"],
        })

    data = json.dumps({
        "rows": rows,
        "words": words,
        "flaggedYes": words["flagged_yes"],
        "dimensions": [
            {"key": key,
             "label": words["dimensions"][key][0],
             "options": words["dimensions"][key][1]}
            for key in dimensions
        ],
    }, ensure_ascii=False)

    html = (RUBRIC_PAGE_TEMPLATE
            .replace("__LANG__", language)
            .replace("__DATA__", data))
    path.write_text(html, encoding="utf-8")
    return path


RUBRIC_PAGE_TEMPLATE = """<!doctype html>
<html lang="__LANG__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rating sheet</title>
<style>
  :root {
    --bg: #f6f6f4; --card: #ffffff; --ink: #1b1b1a; --muted: #6a6a66;
    --line: #dddcd7; --accent: #1f5fa8; --warn: #b02828; --ok: #1f7a4d;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 0 16px 120px; background: var(--bg); color: var(--ink);
         font: 16px/1.6 -apple-system, "Segoe UI", system-ui, sans-serif; }
  header { position: sticky; top: 0; z-index: 10; background: var(--bg);
           border-bottom: 1px solid var(--line); padding: 14px 0 10px;
           margin: 0 -16px 20px; padding-left: 16px; padding-right: 16px; }
  .wrap { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 14px; }
  .bar { height: 6px; background: var(--line); border-radius: 3px; margin-top: 10px; }
  .bar div { height: 100%; width: 0%; background: var(--ok); border-radius: 3px;
             transition: width .2s; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
          padding: 18px 20px; margin-bottom: 18px; }
  .card.done { border-left: 4px solid var(--ok); }
  .num { font-weight: 700; color: var(--accent); }
  .q { font-weight: 600; margin: 2px 0 12px; }
  .facts { background: #fafaf8; border: 1px solid var(--line); border-radius: 8px;
           padding: 10px 14px; margin-bottom: 14px; font-size: 14px; }
  .facts div { display: flex; gap: 8px; }
  .facts b { min-width: 190px; color: var(--muted); font-weight: 500; }
  .flag { color: var(--warn); font-weight: 600; }
  .answer { white-space: pre-wrap; background: #fbfbf9; border: 1px solid var(--line);
            border-left: 3px solid var(--accent); border-radius: 6px;
            padding: 14px 16px; margin-bottom: 16px; }
  details { border: 1px solid var(--line); border-radius: 8px; padding: 8px 14px;
            margin-bottom: 16px; background: #fafaf8; }
  summary { cursor: pointer; font-size: 14px; font-weight: 600; }
  .passage { border-top: 1px solid var(--line); padding: 10px 0; font-size: 14px; }
  .passage .mk { font-weight: 700; color: var(--accent); }
  .passage .src { color: var(--muted); }
  .passage p { margin: 6px 0 0; white-space: pre-wrap; }
  .pnote { font-size: 13px; color: var(--muted); margin: 6px 0 0; }
  .who { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
         gap: 2px 18px; background: #f3f6fa; border: 1px solid #d6e2f0;
         border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;
         font-size: 14px; }
  .who div { display: flex; justify-content: space-between; gap: 10px;
             border-bottom: 1px dotted #d6e2f0; padding: 2px 0; }
  .who b { color: var(--muted); font-weight: 500; }
  .who span { font-variant-numeric: tabular-nums; }
  .whoHead { font-size: 13px; font-weight: 600; color: var(--muted);
             margin-bottom: 4px; }
  .dim { margin-bottom: 12px; }
  .dim > b { display: block; margin-bottom: 6px; font-size: 15px; }
  label { display: block; cursor: pointer; padding: 5px 10px; border-radius: 6px;
          font-size: 14px; }
  label:hover { background: #f0f0ec; }
  input[type=radio] { margin-right: 8px; }
  textarea { width: 100%; border: 1px solid var(--line); border-radius: 6px;
             padding: 8px; font: inherit; font-size: 14px; resize: vertical; }
  footer { position: fixed; bottom: 0; left: 0; right: 0; background: var(--card);
           border-top: 1px solid var(--line); padding: 12px 16px; }
  .frow { max-width: 860px; margin: 0 auto; display: flex; gap: 12px;
          align-items: center; flex-wrap: wrap; }
  button { font: inherit; padding: 9px 16px; border-radius: 7px; cursor: pointer;
           border: 1px solid var(--accent); background: var(--accent); color: #fff; }
  button.ghost { background: #fff; color: var(--accent); }
  button:disabled { opacity: .45; cursor: not-allowed; }
  input[type=text] { font: inherit; padding: 8px 10px; border: 1px solid var(--line);
                     border-radius: 6px; }
  @media (max-width: 640px) { .facts b { min-width: 0; display: block; } .facts div { display: block; } }
</style>
</head>
<body>
<header><div class="wrap">
  <h1 id="heading"></h1>
  <div class="sub" id="status"></div>
  <div class="bar"><div id="bar"></div></div>
  <div class="sub" style="margin-top:6px"><a id="switch" href="#"></a></div>
</div></header>

<div class="wrap" id="cards"></div>

<footer><div class="frow">
  <label style="padding:0"><span id="nameLabel"></span>
    <input type="text" id="rater"></label>
  <button id="save" disabled></button>
  <button class="ghost" id="clear"></button>
  <span class="sub" id="hint"></span>
</div></footer>

<script>
const DATA = __DATA__;
const W = DATA.words;
// One store for both languages, so switching keeps what is already entered.
const KEY = "rubrik-v1";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }

document.getElementById("heading").textContent = W.heading;
document.getElementById("nameLabel").textContent = W.name;
document.getElementById("rater").placeholder = W.name_hint;
document.getElementById("save").textContent = W.download;
document.getElementById("clear").textContent = W.clear;
const switcher = document.getElementById("switch");
switcher.textContent = W.other_language;
switcher.href = W.other_file;

const esc = s => String(s).replace(/[&<>]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

const cards = document.getElementById("cards");
DATA.rows.forEach(row => {
  const card = document.createElement("div");
  card.className = "card";
  card.id = "card-" + row.order;

  let facts = row.facts.map(([k, v]) => {
    const warn = v === DATA.flaggedYes ? ' class="flag"' : "";
    return `<div><b>${k}</b><span${warn}>${v}</span></div>`;
  }).join("");

  let dims = DATA.dimensions.map(d => {
    const opts = d.options.map((text, value) =>
      `<label><input type="radio" name="${d.key}-${row.order}" value="${value}">${text}</label>`
    ).join("");
    return `<div class="dim"><b>${d.label}</b>${opts}</div>`;
  }).join("");

  card.innerHTML = `
    <div><span class="num">${row.order} ${W.of} ${DATA.rows.length}</span>
         <span class="sub"> &middot; ${row.question_id} &middot; ${row.category}
         &middot; ${W.applicant} ${row.applicant}</span></div>
    <div class="q">${row.question}</div>
    <div class="whoHead">${W.who}</div>
    <div class="who">${row.who.map(([k, v]) =>
      `<div><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join("")}</div>
    <div class="facts">${facts}</div>
    <div class="answer">${esc(row.answer)}</div>
    <details><summary>${W.passages}</summary>
      <div class="pnote">${W.passages_note}</div>
      ${row.passages.map(p => `<div class="passage">
         <span class="mk">[${p.marker}]</span>
         <span class="src">${esc(p.source)} &middot; ${p.score}</span>
         <p>${esc(p.text)}</p></div>`).join("")}
    </details>
    ${dims}
    <div class="dim"><b>${W.note}</b>
      <textarea rows="2" id="note-${row.order}"></textarea></div>`;
  cards.appendChild(card);
});

function restore() {
  DATA.rows.forEach(row => {
    const saved = state[row.order] || {};
    DATA.dimensions.forEach(d => {
      if (saved[d.key] !== undefined) {
        const el = document.querySelector(`input[name="${d.key}-${row.order}"][value="${saved[d.key]}"]`);
        if (el) el.checked = true;
      }
    });
    if (saved.note) document.getElementById("note-" + row.order).value = saved.note;
  });
  if (state.__rater) document.getElementById("rater").value = state.__rater;
}

function collect() {
  const out = [];
  DATA.rows.forEach(row => {
    const entry = { note: document.getElementById("note-" + row.order).value };
    DATA.dimensions.forEach(d => {
      const picked = document.querySelector(`input[name="${d.key}-${row.order}"]:checked`);
      if (picked) entry[d.key] = Number(picked.value);
    });
    out.push({ row, entry });
  });
  return out;
}

function refresh() {
  const all = collect();
  let complete = 0;
  all.forEach(({ row, entry }) => {
    const done = DATA.dimensions.every(d => entry[d.key] !== undefined);
    if (done) complete++;
    document.getElementById("card-" + row.order).classList.toggle("done", done);
    state[row.order] = entry;
  });
  state.__rater = document.getElementById("rater").value;
  try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {}

  const total = DATA.rows.length;
  document.getElementById("status").textContent =
    complete + " " + W.of + " " + total + " " + W.rated;
  document.getElementById("bar").style.width = (100 * complete / total) + "%";
  document.getElementById("save").disabled = complete === 0;
  document.getElementById("hint").textContent =
    complete < total ? W.partial : W.complete;
}

document.addEventListener("input", refresh);
document.addEventListener("change", refresh);

document.getElementById("clear").onclick = () => {
  if (!confirm(W.confirm)) return;
  localStorage.removeItem(KEY);
  location.reload();
};

document.getElementById("save").onclick = () => {
  const rater = document.getElementById("rater").value || "";
  const cols = ["rating_order", "question_id", "category"]
    .concat(DATA.dimensions.map(d => d.key))
    .concat(["rater", "note", "configuration_hidden"]);
  const esc = v => {
    const s = v === undefined || v === null ? "" : String(v);
    return /[",\\n\\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const lines = [cols.join(",")];
  collect().forEach(({ row, entry }) => {
    const values = [row.order, row.question_id, row.category]
      .concat(DATA.dimensions.map(d => entry[d.key] === undefined ? "" : entry[d.key]))
      .concat([rater, entry.note, row.hidden]);
    lines.push(values.map(esc).join(","));
  });
  const blob = new Blob(["\\ufeff" + lines.join("\\r\\n")], {type: "text/csv;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  // The rater's name goes in the filename. Two people downloading
  // "evaluation_rubric_filled.csv" would silently overwrite each other, and
  // the second rating exists precisely to be compared with the first.
  const tag = rater.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-")
              .replace(/^-|-$/g, "") || "rater";
  link.download = "evaluation_rubric_filled_" + tag + ".csv";
  link.click();
};

restore();
refresh();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------
# Reading the filled-in sheet back
# ------------------------------------------------------------------

ALL_RUBRIC_SCORES = ["correctness_0_1_2", "faithfulness_0_1_2", "clarity_0_1_2"]
RUBRIC_FILLED_FILE = TABLES_FOLDER / "evaluation_rubric_filled.csv"
RUBRIC_RESULT_FILE = TABLES_FOLDER / "evaluation_rubric_results.csv"
RUBRIC_TESTS_FILE = TABLES_FOLDER / "evaluation_rubric_tests.csv"
RUBRIC_PER_RATER_FILE = TABLES_FOLDER / "evaluation_rubric_per_rater.csv"
RUBRIC_AGREEMENT_FILE = TABLES_FOLDER / "evaluation_rubric_agreement.csv"


def scores_present(sheet):
    """Which of the three dimensions this sheet actually asks for."""
    return [name for name in ALL_RUBRIC_SCORES if name in sheet.columns]


def find_filled_sheets():
    """Every scored sheet, from the tables folder and from Downloads.

    The page names each download after the rater, so a second rater's file
    sits beside the first instead of replacing it. Both are picked up here;
    if the same rater appears twice, the newer file wins.
    """

    # Both formats. The page hands out a CSV, but a rater who opens it in
    # Excel and saves from there gets an .xlsx, which is a reasonable thing
    # to do and should not quietly produce "no scored sheet found".
    patterns = ["evaluation_rubric_filled*.csv",
                "evaluation_rubric_filled*.xlsx"]
    places = [TABLES_FOLDER, Path.home() / "Downloads"]

    found = []
    for place in places:
        if not place.is_dir():
            continue
        for pattern in patterns:
            found.extend(place.glob(pattern))

    by_name = {}
    for path in sorted(found, key=lambda one: one.stat().st_mtime):
        by_name[path.name] = path        # later file of the same name wins
    # Oldest first, so that a later rating of the same answer by the same
    # person replaces an earlier one further down.
    return sorted(by_name.values(), key=lambda one: one.stat().st_mtime)


def unwrap_single_column(frame):
    """Undo the German Excel round trip.

    A comma-separated file opened in an Excel set to semicolons lands in a
    single column, and saving from there keeps it that way: one cell per
    row, with the header as the column name. The data is all present, so
    refusing the file would be pedantry - it is re-split here instead.

    Recognised by the shape, not by the filename: exactly one column whose
    name contains commas.
    """

    if len(frame.columns) != 1 or "," not in str(frame.columns[0]):
        return frame

    import io

    lines = [str(frame.columns[0])]
    lines.extend(str(value) for value in frame.iloc[:, 0].tolist())
    repaired = pd.read_csv(io.StringIO("\n".join(lines)))
    print("  (repaired a sheet that Excel had collapsed into one column)")
    return repaired


def read_rubric(path=None):
    """Load every scored sheet and check them before anything is computed."""

    paths = [path] if path is not None else find_filled_sheets()
    if not paths:
        return None, ["No scored sheet in " + str(TABLES_FOLDER)
                      + " and none in the Downloads folder"]

    parts = []
    for one in paths:
        part = (pd.read_excel(one) if one.suffix.lower() == ".xlsx"
                else pd.read_csv(one))
        part = unwrap_single_column(part)
        print("Reading:", one, "-", len(part), "rows")
        if "rater" not in part.columns or part["rater"].isna().all():
            part["rater"] = one.stem.replace("evaluation_rubric_filled_", "")
        part["source_file"] = one.name
        parts.append(part)

    sheet = pd.concat(parts, ignore_index=True)

    # One rating per person per answer. A second download by the same person
    # - the copy still sitting in Downloads, a corrected re-rating - would
    # otherwise be averaged in as if it were a second opinion.
    before = len(sheet)
    sheet = sheet.drop_duplicates(
        subset=["rater", "question_id", "configuration_hidden"], keep="last")
    if len(sheet) < before:
        print("  (ignored", before - len(sheet),
              "repeated ratings of the same answer by the same person)")

    problems = []

    for column in ["configuration_hidden", "question_id"]:
        if column not in sheet.columns:
            problems.append("Missing column: " + column)
    if problems:
        return None, problems

    columns = scores_present(sheet)
    if not columns:
        return None, ["The sheet has none of " + ", ".join(ALL_RUBRIC_SCORES)]

    for column in columns:
        sheet[column] = pd.to_numeric(sheet[column], errors="coerce")
        missing = int(sheet[column].isna().sum())
        if missing:
            problems.append(str(missing) + " of " + str(len(sheet))
                            + " rows have no score in " + column)
        outside = sheet[~sheet[column].isin([0, 1, 2]) & sheet[column].notna()]
        if len(outside):
            problems.append(str(len(outside)) + " rows in " + column
                            + " are not 0, 1 or 2")

    return sheet, problems


def summarise_scores(scored, columns, label_column=None):
    """Mean and share of full marks per configuration, optionally per rater."""

    keys = ([label_column] if label_column else []) + ["configuration"]
    rows = []
    for key, part in scored.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row["answers_scored"] = len(part)
        for column in columns:
            short = column.replace("_0_1_2", "")
            row["mean_" + short] = round(part[column].mean(), 3)
            row["share_full_marks_" + short] = round((part[column] == 2).mean(), 3)
        row["mean_total_of_" + str(2 * len(columns))] = round(
            part[columns].sum(axis=1).mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_between(scored, columns):
    """How far the raters agree, where there are at least two of them.

    Cohen's kappa is reported both unweighted and quadratically weighted.
    The scores are ordinal, so the weighted form is the one to quote: it
    counts a 0-against-2 disagreement as worse than a 1-against-2, which
    plain kappa does not. Raw agreement is shown beside it because kappa on
    a skewed distribution can look poor while the raters in fact rarely
    differ.
    """

    from sklearn.metrics import cohen_kappa_score

    raters = sorted(scored["rater"].dropna().unique())
    if len(raters) < 2:
        return pd.DataFrame(), raters

    rows = []
    for first, second in [(a, b) for i, a in enumerate(raters)
                          for b in raters[i + 1:]]:
        left = scored[scored["rater"] == first].set_index(
            ["question_id", "configuration"])
        right = scored[scored["rater"] == second].set_index(
            ["question_id", "configuration"])
        shared = left.index.intersection(right.index)
        if len(shared) == 0:
            continue
        for column in columns:
            one = left.loc[shared, column].astype(int)
            two = right.loc[shared, column].astype(int)
            rows.append({
                "dimension": column.replace("_0_1_2", ""),
                "rater_1": first,
                "rater_2": second,
                "answers_both_rated": len(shared),
                "exact_agreement": round(float((one == two).mean()), 3),
                "within_one_point": round(float((abs(one - two) <= 1).mean()), 3),
                "kappa": round(float(cohen_kappa_score(one, two)), 3),
                "kappa_quadratic": round(
                    float(cohen_kappa_score(one, two, weights="quadratic")), 3),
            })
    return pd.DataFrame(rows), raters


def analyse_rubric(path=None):
    """Scores per configuration, the paired tests, and rater agreement.

    With more than one rater the paired test runs on the CONSENSUS: the mean
    of the raters for each answer, with full marks meaning every rater gave a
    2. Pooling the individual ratings instead would double the row count
    without doubling the information, because both raters score the same
    answers, and would make the test look more certain than it is.
    """

    sheet, problems = read_rubric(path)
    if sheet is None:
        return None, None, None, problems

    columns = scores_present(sheet)
    sheet = sheet.rename(columns={"configuration_hidden": "configuration"})
    scored = sheet.dropna(subset=columns)

    per_rater = summarise_scores(scored, columns, label_column="rater")
    agreement, raters = agreement_between(scored, columns)

    consensus = (scored.groupby(["question_id", "configuration"], as_index=False)
                 [columns].mean())
    summary = summarise_scores(consensus, columns)
    summary.insert(1, "raters", len(raters) if raters else 1)

    tests = []
    for column in columns:
        short = column.replace("_0_1_2", "")
        marked = consensus.copy()
        # Full marks only when every rater gave a 2, so a split verdict does
        # not quietly count as the best possible answer.
        marked[short + "_full"] = marked[column] == 2
        for first, second in [("A", "C"), ("A", "B"), ("B", "C")]:
            tests.append(mcnemar(marked, first, second, short + "_full"))
    tests = pd.DataFrame(tests)

    return summary, tests, (per_rater, agreement), problems


def run_rubric_analysis():
    """main.py rubric - read the scored sheet and report."""

    summary, tests, extra, problems = analyse_rubric()

    if summary is None:
        print("Nothing to analyse yet.")
        for problem in problems:
            print("  -", problem)
        print()
        print("Open", RUBRIC_PAGE_FILE.name, "in a browser, rate the answers,")
        print("and press the download button. The file is picked up from the")
        print("tables folder or from Downloads.")
        return 1

    per_rater, agreement = extra

    if problems:
        print("Warnings - the numbers below use only the scored rows:")
        for problem in problems:
            print("  -", problem)
        print()

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RUBRIC_RESULT_FILE, index=False)
    tests.to_csv(RUBRIC_TESTS_FILE, index=False)

    print("-" * 70)
    print("RUBRIC PER CONFIGURATION")
    print("-" * 70)
    print(summary.to_string(index=False))

    if len(per_rater) and per_rater["rater"].nunique() > 1:
        print()
        print("-" * 70)
        print("PER RATER")
        print("-" * 70)
        print(per_rater.to_string(index=False))

    if len(agreement):
        print()
        print("-" * 70)
        print("AGREEMENT BETWEEN RATERS")
        print("-" * 70)
        print(agreement.to_string(index=False))
        agreement.to_csv(RUBRIC_AGREEMENT_FILE, index=False)

    print()
    print("-" * 70)
    print("PAIRED TESTS (full marks = every rater gave a 2)")
    print("-" * 70)
    print(tests.to_string(index=False))

    print()
    written = [RUBRIC_RESULT_FILE, RUBRIC_TESTS_FILE]
    if len(agreement):
        written.append(RUBRIC_AGREEMENT_FILE)
    if len(per_rater):
        per_rater.to_csv(RUBRIC_PER_RATER_FILE, index=False)
        written.append(RUBRIC_PER_RATER_FILE)
    for path in written:
        print("  written:", path.relative_to(PROJECT_ROOT))

    return 0


# ------------------------------------------------------------------
# main.py evaluate
# ------------------------------------------------------------------

def analyse(answers):
    """Everything downstream of the answers.

    Kept apart from generating them so the analysis can be changed and rerun
    without asking the model 72 questions again, and so a changed metric can
    be applied to the answers that are already on record.
    """

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    summary = summarise(answers)
    summary.to_csv(SUMMARY_FILE, index=False)

    by_category = summarise_by_category(answers)
    by_category.to_csv(CATEGORY_FILE, index=False)

    tests = paired_tests(answers)
    tests.to_csv(PAIRED_FILE, index=False)

    sheet = rubric_template(answers)
    sheet.to_csv(RUBRIC_FILE, index=False)
    rubric_page(sheet, RUBRIC_PAGE_FILE, language="en")
    rubric_page(sheet, RUBRIC_PAGE_FILE_DE, language="de")

    print()
    print("-" * 70)
    print("PER CONFIGURATION")
    print("-" * 70)
    headline = ["configuration", "answers", "answers_stating_a_number",
                "share_fully_supported_when_committed",
                "share_nothing_invented_when_committed",
                "share_numbers_supported", "share_numbers_grounded",
                "source_questions", "source_questions_citing",
                "fabricated_citations",
                "flagged_answers", "flagged_mentioning_human_review",
                "not_flagged_offering_human_review", "mean_seconds"]
    print(summary[headline].to_string(index=False))

    print()
    print("-" * 70)
    print("PAIRED TESTS (exact McNemar)")
    print("-" * 70)
    print(tests.to_string(index=False))

    print()
    print("-" * 70)
    print("WRITTEN")
    print("-" * 70)
    for path in [ANSWERS_FILE, SUMMARY_FILE, CATEGORY_FILE, PAIRED_FILE,
                 RUBRIC_FILE, RUBRIC_PAGE_FILE]:
        print("  ", path.relative_to(PROJECT_ROOT))

    print()
    print("  Still to do by hand: open", RUBRIC_PAGE_FILE.name,
          "in a browser and rate the", len(RUBRIC_QUESTIONS) * len(CONFIGURATIONS),
          "answers.")
    print("  Then: uv run main.py rubric")
    print("  Finished at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return 0


def run_evaluation(prompt_version=PROMPT_VERSION, reuse_answers=False):
    """Generate the answers, then analyse them.

    reuse_answers skips the generation and reads the answers written by an
    earlier run, for when only the analysis has changed.
    """

    if reuse_answers:
        if not ANSWERS_FILE.exists():
            print("No answers to reuse:", ANSWERS_FILE)
            print("Run `uv run main.py evaluate` without --reuse-answers first.")
            return 1
        print("Reusing the answers from", ANSWERS_FILE.name,
              "- no question is asked again")
        answers = pd.read_csv(ANSWERS_FILE)

        # Answers written before this column existed can still be analysed:
        # it is derivable from what was recorded.
        if "stated_a_number" not in answers.columns:
            answers["stated_a_number"] = answers["numbers_checked"] > 0

        return analyse(answers)

    print("Evaluation of the loan assessment assistant")
    print("  model          :", MODEL_NAME)
    print("  prompt version :", prompt_version)
    print("  questions      :", len(QUESTIONS))
    print("  configurations :", ", ".join(CONFIGURATIONS))
    print("  answers to make:", len(QUESTIONS) * len(CONFIGURATIONS))
    print()

    answers = run_all_questions(prompt_version=prompt_version)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    answers.to_csv(ANSWERS_FILE, index=False)

    return analyse(answers)
