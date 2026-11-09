"""
Inference pathway: one applicant, one question, one checked answer.

This module puts the pieces together:

    applicant values
        -> contract.make_applicant_row          prepare like training
        -> test_set_sweep.estimate_max_amount   the estimated maximum (step 5)
        -> explain.explain_applicant            the SHAP drivers (step 6)
        -> deferral.deferral                    the deferral gate (step 7)
        -> contract.build_contract              every allowed number (step 8)
        -> retrieve.retrieve                    four supporting passages (step 10)
        -> llm.build_prompt + ask_qwen          the answer (steps 11, 12)
        -> contract.check_*                     the answer check (step 13)

THE THREE CONFIGURATIONS
------------------------
    A   question + applicant values
    B   + the contract (model results and SHAP)
    C   + the retrieved passages          <- the full system

The contract is ALWAYS built, in all three configurations. In A it is simply
not put into the prompt - but it is still used to check the answer. That is
what makes the comparison fair and automatic: configuration A cannot know the
estimated maximum, so every figure it states about it is measurably wrong,
and no human has to decide that.

Usage
-----
    uv run main.py ask

    from explainable_ai_project.assistant import ask
    result = ask(values, "Why is my maximum lower than I asked for?")
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from explainable_ai_project.contract import build_contract
from explainable_ai_project.contract import check_citations
from explainable_ai_project.contract import check_deferral_mentioned
from explainable_ai_project.contract import check_numbers
from explainable_ai_project.contract import contract_as_json
from explainable_ai_project.contract import validate_contract
from explainable_ai_project.llm import ask_qwen
from explainable_ai_project.llm import build_prompt
from explainable_ai_project.llm import check_ready
from explainable_ai_project.llm import system_prompt
from explainable_ai_project.llm import warm_up
from explainable_ai_project.retrieve import format_chunks_for_prompt
from explainable_ai_project.retrieve import retrieve


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent

TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"
LOG_FILE = TABLES_FOLDER / "interaction_log.csv"

CONFIGURATIONS = ["A", "B", "C"]
DEFAULT_TOP_K = 4


# ------------------------------------------------------------------
# An example applicant, for `uv run main.py ask`
# ------------------------------------------------------------------

DEMO_APPLICANT = {
    "age": 34,
    "dependents": 1,
    "annual_income": 72000,
    "credit_score": 701,
    "loan_amount": 25000,
    "loan_term_months": 60,
    "marital_status": "Married",
    "education": "Undergraduate",
    "employment_status": "Employed",
    "loan_purpose": "Home",
}

DEMO_QUESTION = "What is the largest loan I could get, and why is it not more?"


# ------------------------------------------------------------------
# The applicant in words (every configuration gets this)
# ------------------------------------------------------------------

def applicant_as_text(values):
    lines = []
    for name, value in values.items():
        lines.append("  " + name + ": " + str(value))
    return "\n".join(lines)


# ------------------------------------------------------------------
# One answer
# ------------------------------------------------------------------

def ask(values, question, configuration="C", prompt_version="v1",
        top_k=DEFAULT_TOP_K, contract=None, verbose=True):
    """Answer one question about one applicant, and check the answer.

    values          the applicant's inputs
    configuration   "A", "B" or "C" (see the module docstring)
    prompt_version  "v0" or "v1"
    contract        a contract built earlier (saves recomputing it for A/B/C)

    Returns a dictionary with the answer, the contract, the retrieved chunks
    and the results of every check.
    """

    if configuration not in CONFIGURATIONS:
        raise ValueError("configuration must be A, B or C")

    check_ready()

    if contract is None:
        if verbose:
            print("Building the contract (models, sweep, SHAP, deferral) ...")
        contract = build_contract(values)

    problems = validate_contract(contract)
    if problems:
        raise ValueError("The contract is not usable:\n  - " + "\n  - ".join(problems))

    # Retrieval only happens in configuration C
    chunks = []
    chunks_text = None
    if configuration == "C":
        if verbose:
            print("Retrieving", top_k, "supporting passages ...")
        chunks = retrieve(question, top_k=top_k)
        chunks_text = format_chunks_for_prompt(chunks)

    # The contract goes into the prompt only in B and C
    contract_text = None
    if configuration in ("B", "C"):
        contract_text = contract_as_json(contract)

    prompt = build_prompt(
        question=question,
        applicant_text=applicant_as_text(values),
        contract_text=contract_text,
        chunks_text=chunks_text,
    )

    if verbose:
        print("Asking the model (configuration", configuration,
              ", prompt", prompt_version + ") ...")

    generation = ask_qwen(prompt, system_prompt(prompt_version))
    answer = generation["answer"]

    # Step 13: check the answer against the contract
    numbers = check_numbers(answer, contract)
    citations = check_citations(answer, chunks)
    deferral_check = check_deferral_mentioned(answer, contract)

    result = {
        "question": question,
        "configuration": configuration,
        "prompt_version": prompt_version,
        "answer": answer,
        "contract": contract,
        "chunks": chunks,
        "generation": generation,
        "checks": {
            "numbers": numbers,
            "citations": citations,
            "deferral": deferral_check,
        },
    }

    return result


# ------------------------------------------------------------------
# Logging (Topic 3: every interaction is recorded)
# ------------------------------------------------------------------

def log_interaction(result):
    """Append one row per answer, so the run can be audited afterwards."""

    contract = result["contract"]
    maximum = contract["estimated_maximum"]
    checks = result["checks"]
    generation = result["generation"]

    row = {
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": result["configuration"],
        "prompt_version": result["prompt_version"],
        "question": result["question"],
        "model": generation["model"],
        "seed": generation["seed"],
        "temperature": generation["temperature"],
        "seconds": generation["seconds"],
        "answer_tokens": generation["answer_tokens"],
        "estimated_maximum": maximum["amount"],
        "max_is_upper_limit": maximum["is_upper_limit"],
        "deferral_flagged": contract["deferral"]["flagged"],
        "deferral_reasons": ", ".join(contract["deferral"]["reasons"]),
        "numbers_checked": checks["numbers"]["numbers_checked"],
        "numbers_supported": checks["numbers"]["numbers_supported"],
        "share_numbers_supported": checks["numbers"]["share_supported"],
        "unsupported_numbers": ", ".join(str(number) for number
                                        in checks["numbers"]["unsupported"]),
        "cites_a_retrieved_source": checks["citations"]["cites_a_retrieved_source"],
        "invalid_citation_markers": ", ".join(checks["citations"]["invalid_markers"]),
        "deferral_mentioned": checks["deferral"]["mentions_human_review"],
        "answer": result["answer"],
    }

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row])

    if LOG_FILE.exists():
        table.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        table.to_csv(LOG_FILE, index=False)

    return row


# ------------------------------------------------------------------
# Printing one result
# ------------------------------------------------------------------

def print_result(result):
    contract = result["contract"]
    maximum = contract["estimated_maximum"]
    deferral_info = contract["deferral"]
    checks = result["checks"]

    print()
    print("-" * 70)
    print("WHAT THE MODELS SAY")
    print("-" * 70)

    for dataset_name, model in contract["models"].items():
        print(" ", dataset_name,
              "| P(approve) at the requested amount:", model["p_approve_at_requested"],
              "| maximum at 0.8:", model["max_amount_0.8"])

    if maximum["no_amount_found"]:
        print("  estimated maximum: none - not even the smallest amount reaches 0.8")
    else:
        note = " (at the upper limit of what may be tried)" if maximum["is_upper_limit"] else ""
        print("  estimated maximum:", maximum["amount"], note)

    print("  deferral:", "FLAGGED" if deferral_info["flagged"] else "not flagged",
          ("- " + ", ".join(deferral_info["reasons"])) if deferral_info["reasons"] else "")

    print()
    print("  main drivers (SHAP, raw forest scale):")
    for driver in contract["top_features"]:
        print("   ", driver["direction"].ljust(5), driver["feature"],
              "=", driver["value"], "-> contribution", driver["contribution"])

    if result["chunks"]:
        print()
        print("  retrieved passages:")
        for number, chunk in enumerate(result["chunks"], start=1):
            print("   ", "[" + str(number) + "]", chunk["source"],
                  "| score", chunk["score"], "|", str(chunk.get("section", ""))[:60])

    print()
    print("-" * 70)
    print("THE ANSWER")
    print("-" * 70)
    print(result["answer"])

    print()
    print("-" * 70)
    print("ANSWER CHECKS")
    print("-" * 70)

    numbers = checks["numbers"]
    verdict = "PASSED" if numbers["passed"] else "FAILED"
    print("  numbers from the contract:", numbers["numbers_supported"], "of",
          numbers["numbers_checked"],
          "(" + str(round(numbers["share_supported"] * 100, 1)) + " %)", verdict)
    if numbers["unsupported"]:
        print("    not in the contract:", numbers["unsupported"])

    citations = checks["citations"]
    if result["chunks"]:
        print("  citations:", "PASSED" if citations["passed"] else "FAILED",
              "| sources named:", citations["documents_named"] or "none",
              "| markers:", citations["markers_used"] or "none")
        if citations["invalid_markers"]:
            print("    markers that were never given:", citations["invalid_markers"])

    deferral_check = checks["deferral"]
    if deferral_check["flagged"]:
        print("  human review mentioned:",
              "YES" if deferral_check["mentions_human_review"] else "NO - RULE BROKEN")

    generation = result["generation"]
    print()
    print("  model:", generation["model"], "| seed:", generation["seed"],
          "| temperature:", generation["temperature"],
          "|", generation["seconds"], "s |", generation["answer_tokens"], "tokens")


# ------------------------------------------------------------------
# main.py ask
# ------------------------------------------------------------------

def run_ask(values=None, question=None, configuration="C", prompt_version="v1"):
    """Answer one question for one applicant and log it."""

    warm_up()   # never let the first real question be the one that loads the model

    if values is None:
        values = DEMO_APPLICANT
        print("Using the example applicant (edit DEMO_APPLICANT in assistant.py):")
        print(applicant_as_text(values))
        print()

    if question is None:
        question = DEMO_QUESTION

    print("Question:", question)
    print()

    result = ask(values, question, configuration=configuration,
                 prompt_version=prompt_version)

    print_result(result)

    log_interaction(result)
    print()
    print("Logged to", LOG_FILE.relative_to(PROJECT_ROOT))

    return result
