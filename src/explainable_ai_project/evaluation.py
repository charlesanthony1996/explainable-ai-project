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
# Reading the filled-in sheet back
# ------------------------------------------------------------------

ALL_RUBRIC_SCORES = ["correctness_0_1_2", "faithfulness_0_1_2", "clarity_0_1_2"]
RUBRIC_FILLED_FILE = TABLES_FOLDER / "evaluation_rubric_filled.csv"
RUBRIC_RESULT_FILE = TABLES_FOLDER / "evaluation_rubric_results.csv"
RUBRIC_TESTS_FILE = TABLES_FOLDER / "evaluation_rubric_tests.csv"


def scores_present(sheet):
    """Which of the three dimensions this sheet actually asks for."""
    return [name for name in ALL_RUBRIC_SCORES if name in sheet.columns]


def read_rubric(path=RUBRIC_FILLED_FILE):
    """Load the scored sheet and check it before anything is computed."""

    if not path.exists():
        return None, ["No scored sheet at " + str(path)]

    sheet = pd.read_csv(path)
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


def analyse_rubric(path=RUBRIC_FILLED_FILE):
    """Mean score per configuration, plus the paired tests.

    The paired test treats a score as 'good' when it is 2. Using the mean
    alone would hide which answers moved; using the pairing shows it, and it
    is the same exact binomial as everywhere else in this file.
    """

    sheet, problems = read_rubric(path)
    if sheet is None:
        return None, None, problems

    columns = scores_present(sheet)
    sheet = sheet.rename(columns={"configuration_hidden": "configuration"})
    scored = sheet.dropna(subset=columns)

    rows = []
    for configuration in CONFIGURATIONS:
        part = scored[scored["configuration"] == configuration]
        if part.empty:
            continue
        row = {"configuration": configuration, "answers_scored": len(part)}
        for column in columns:
            short = column.replace("_0_1_2", "")
            row["mean_" + short] = round(part[column].mean(), 3)
            row["share_full_marks_" + short] = round((part[column] == 2).mean(), 3)
        row["mean_total_of_" + str(2 * len(columns))] = round(
            part[columns].sum(axis=1).mean(), 3)
        rows.append(row)
    summary = pd.DataFrame(rows)

    tests = []
    for column in columns:
        short = column.replace("_0_1_2", "")
        marked = scored.copy()
        marked[short + "_full"] = marked[column] == 2
        for first, second in [("A", "C"), ("A", "B"), ("B", "C")]:
            tests.append(mcnemar(marked, first, second, short + "_full"))
    tests = pd.DataFrame(tests)

    return summary, tests, problems


def run_rubric_analysis():
    """main.py rubric - read the scored sheet and report."""

    summary, tests, problems = analyse_rubric()

    if summary is None:
        print("Nothing to analyse yet.")
        for problem in problems:
            print("  -", problem)
        print()
        print("Fill in", RUBRIC_FILE.name, "and save it as",
              RUBRIC_FILLED_FILE.name, "in the same folder.")
        return 1

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

    print()
    print("-" * 70)
    print("PAIRED TESTS (full marks = 2)")
    print("-" * 70)
    print(tests.to_string(index=False))

    print()
    for path in [RUBRIC_RESULT_FILE, RUBRIC_TESTS_FILE]:
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
    for path in [ANSWERS_FILE, SUMMARY_FILE, CATEGORY_FILE, PAIRED_FILE, RUBRIC_FILE]:
        print("  ", path.relative_to(PROJECT_ROOT))

    print()
    print("  Still to do by hand:", RUBRIC_FILE.name,
          "- 0/1/2 for correctness, faithfulness and clarity.")
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
