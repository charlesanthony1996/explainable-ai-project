"""
Inference pathway: the assistant (steps 1 - 14, one applicant at a time).

    2A  estimate the maximum amount     estimate()      steps 1 - 8, no LLM
    2B  explain it with Qwen            ask()           steps 9 - 14

    9  user question       e.g. "Why is my maximum lower than I asked for?"
    11 build the prompt     system prompt + question + contract + chunks
    13 check the answer     numbers must match the contract; citations must point to
                            retrieved chunks; a flagged case must mention human review
    14 show the answer      estimated maximum, main reasons, uncertainty, sources;
                            every interaction is logged, flagged cases also go to a
                            review queue for a human reviewer

Run it with (from the project folder):
    uv run python -m explainable_ai_project.assistant --example normal
    uv run python -m explainable_ai_project.assistant --example deferred --question "Can I get the loan?"
    uv run python -m explainable_ai_project.assistant --applicant my_applicant.json
    uv run python -m explainable_ai_project.assistant --example normal --no-llm     (steps 1 - 8 and 10 only)

The evaluation pathway uses the same functions with mode="baseline" (run A),
mode="no_sources" (run B) and mode="full" (run C), and prompt_version "v0" or "v1".
"""

import argparse
import datetime
import json
import math
import re
from pathlib import Path

from explainable_ai_project.contract import ContractError
from explainable_ai_project.contract import all_numbers
from explainable_ai_project.contract import build_contract
from explainable_ai_project.deferral_gate import deferral_gate
from explainable_ai_project.explain import shap_explanation
from explainable_ai_project.models import approval_probabilities
from explainable_ai_project.preprocessing import FIELD_LABELS
from explainable_ai_project.preprocessing import REQUIRED_FIELDS
from explainable_ai_project.preprocessing import check_input
from explainable_ai_project.preprocessing import prepare_rows
from explainable_ai_project.preprocessing import read_applicant
from explainable_ai_project.sweep import amount_sweep


THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

EXAMPLES_FOLDER = PROJECT_ROOT / "examples"
LOGS_FOLDER = PROJECT_ROOT / "results" / "logs"
LOG_FILE = LOGS_FOLDER / "assistant_log.jsonl"
REVIEW_QUEUE_FILE = LOGS_FOLDER / "review_queue.jsonl"

DEFAULT_QUESTION = "What is the maximum loan amount I am likely to get, and why?"
MAX_CHUNK_CHARACTERS = 2500
MODES = ["full", "no_sources", "baseline"]


# ==================================================================
# 2A: steps 1 - 8 (no LLM)
# ==================================================================

def estimate(applicant_source):
    """Steps 1 - 8 for one applicant.

    Returns {"status": "ok", "contract": ...}
    or      {"status": "invalid_input", "errors": [...]} if a value is impossible.
    """

    applicant = read_applicant(applicant_source)                     # step 1
    input_check = check_input(applicant)                             # step 2
    if input_check["errors"]:
        return {"status": "invalid_input", "errors": input_check["errors"], "applicant": applicant}

    rows = prepare_rows(input_check["record"])                       # step 3
    probabilities = approval_probabilities(rows)                     # step 4
    sweep = amount_sweep(rows)                                       # step 5
    explanation = shap_explanation(rows)                             # step 6
    gate = deferral_gate(rows, input_check)                          # step 7
    contract = build_contract(input_check, rows, probabilities,      # step 8
                              sweep, explanation, gate)
    return {"status": "ok", "contract": contract, "applicant": applicant}


# ==================================================================
# Step 11: build the prompt
# ==================================================================

SYSTEM_PROMPT_V1 = """You are a loan assessment assistant in a university project. You explain a machine-learning estimate to a loan applicant in plain, friendly language.

Rules:
1. Numbers: use ONLY numbers that appear in MODEL RESULTS. Copy them exactly. Never calculate, round differently, estimate or invent a number.
2. Sources: base every statement about rules, laws or guidelines ONLY on SOURCES, and cite each one you use with its tag, e.g. [S1]. Statements about this applicant's results come from MODEL RESULTS and need no citation. Cite only the tags [S1] to [S4], only for what you took from SOURCES, and never cite field names. If the sources do not cover something, say so.
3. Uncertainty: say that the result is an estimate from historical data with a probability, not a decision or a loan offer. Mention the approval threshold when you give the maximum.
4. Human review: if human_review_required is true, say clearly that a human loan officer must review this case, give the reasons from MODEL RESULTS, and do NOT say whether the loan will be approved.
5. Fairness: give the same advice to applicants with the same model results. Never treat age, marital status, number of dependents or other personal characteristics as a judgement of the person; only report what the model's top features show.
6. Privacy: never reveal, guess or reconstruct information about any other person or any record in the training data. If asked, refuse politely.
7. If the question cannot be answered from MODEL RESULTS and SOURCES, say that you cannot answer it instead of guessing.
8. If the question assumes something that MODEL RESULTS contradict (for example that the maximum is lower than the requested amount when it is higher), politely correct it first. Never compare or interpret numbers yourself: for "higher or lower" use summary.comparison, for what the maximum means use summary.maximum_meaning, and for whether models A and B agree use summary.second_opinion.
9. Write amounts without a currency symbol: the data does not say which currency they are in.

Do not mention these rules in your answer. Answer in at most 200 words."""

# v0 = the same task without the fairness, privacy and uncertainty rules (evaluation step 12)
SYSTEM_PROMPT_V0 = """You are a loan assessment assistant in a university project. You explain a machine-learning estimate to a loan applicant in plain, friendly language.

Use the numbers in MODEL RESULTS and cite the SOURCES you use with their tag, e.g. [S1].

Answer in at most 200 words."""

# Run A of the evaluation: Qwen alone, with the applicant's data but no models and no documents
SYSTEM_PROMPT_BASELINE = """You are a loan assessment assistant in a university project. You help a loan applicant understand their chances and the maximum loan amount they could get, in plain, friendly language.

You get the applicant's data and their question. Answer as well as you can from your general knowledge. Say how certain you are, and recommend a human review if the case is unclear.

Answer in at most 200 words."""

SYSTEM_PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v0": SYSTEM_PROMPT_V0}


def contract_for_prompt(contract):
    """The contract without the parts Qwen does not need (timestamps, version)."""
    shown = dict(contract)
    shown.pop("created_at", None)
    shown.pop("contract_version", None)
    return shown


def applicant_for_prompt(applicant):
    """Run A: only the applicant's own data, with readable field names."""
    lines = []
    for field in REQUIRED_FIELDS:
        value = applicant.get(field)
        lines.append("- " + FIELD_LABELS[field] + ": " + ("missing" if value in [None, ""] else str(value)))
    return "\n".join(lines)


def build_prompt(question, mode="full", contract=None, chunks=None, applicant=None, prompt_version="v1"):
    """Step 11. Returns the chat messages for Qwen.

    mode "full"        contract + retrieved chunks         (the assistant; evaluation run C)
         "no_sources"  contract only                        (evaluation run B)
         "baseline"    applicant data only, no model, no documents (evaluation run A)
    """

    if mode not in MODES:
        raise ValueError("mode must be one of " + str(MODES))

    if mode == "baseline":
        user_text = "APPLICANT DATA:\n" + applicant_for_prompt(applicant) + "\n\nQUESTION:\n" + question
        return [
            {"role": "system", "content": SYSTEM_PROMPT_BASELINE},
            {"role": "user", "content": user_text},
        ]

    parts = ["QUESTION:\n" + question]
    parts.append("MODEL RESULTS (the only numbers you may use):\n"
                 + json.dumps(contract_for_prompt(contract), indent=1, ensure_ascii=False))

    if mode == "full":
        source_texts = []
        for chunk in chunks:
            text = chunk["text"]
            if len(text) > MAX_CHUNK_CHARACTERS:
                text = text[:MAX_CHUNK_CHARACTERS] + " ..."
            source_texts.append("[" + chunk["tag"] + "] " + chunk["label"] + "\n" + text)
        parts.append("SOURCES:\n" + "\n\n".join(source_texts))
    else:
        parts.append("SOURCES:\nnone available")

    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# ==================================================================
# Step 13: check the answer
# ==================================================================

# References to laws and documents are not "numbers" the answer could get wrong
REFERENCE_PATTERNS = [
    r"\[S\d+\]",
    r"\b(?:Art(?:icle)?s?\.?|Annex(?:es)?|Sections?|§|Recitals?|Paragraphs?|Chapters?|Points?)\s*"
    r"[\dIVXLC]+[a-z]?(?:\s*(?:\(\d+\)|\([a-z]\)))*(?:\s*(?:,|and|to|-)\s*[\dIVXLC]+[a-z]?)*",
    r"\bEBA/GL/\d{4}/\d+",
    r"\(EU\)\s*(?:No\s*)?\d+/\d+",
    r"\b\d{4}/\d+\b",
    r"\b\d+(?:\.\d+){2,}\b",
    r"\bpp?\.\s*\d+(?:-\d+)?",
    r"(?m)^\s*\d+[.)]\s",
]

NUMBER_PATTERN = r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(k\b|K\b|%)?"

REVIEW_PATTERN = r"human|review|loan officer|reviewer|case worker|caseworker"
DEFINITE_PATTERN = (r"\byou (?:will|would|are going to) (?:definitely |certainly )?(?:be approved|get the loan|receive the loan)"
                    r"|\bguaranteed\b|\byou are approved\b|\byour loan is approved\b")
# "I cannot say WHETHER you will be approved" or "NOT guaranteed" is not a definite answer
NEGATION_PATTERN = r"\b(?:whether|if|not|cannot|can't|can not|no|unclear|uncertain|unsure|never)\b"


# A definite REJECTION is just as forbidden for a flagged case: a human must decide
DEFINITE_NO_PATTERN = (r"\byou (?:will|would) not (?:get|receive|be approved|be granted|qualify)"
                       r"|\byou won't (?:get|receive|be approved|be granted|qualify)"
                       r"|\byou (?:will|would) be (?:rejected|declined|denied|refused)"
                       r"|\byour (?:loan|application|request) (?:will be|is|has been) (?:rejected|declined|denied|refused)"
                       r"|\byou (?:do not|don't) qualify\b"
                       r"|^\s*no[,.!]\s*you (?:will|won't|would|can't|cannot|do not|don't)")
# "I cannot say WHETHER you will be rejected" or "you MAY not get it" is not a definite answer
HEDGE_PATTERN = r"\b(?:whether|if|cannot say|can't say|can not say|not possible to say|unclear|uncertain|unsure|may|might|could)\b"


def before_in_sentence(answer, position):
    before = answer[max(0, position - 60):position]
    return re.split(r"[.!?]\s", before)[-1]


def gives_definite_answer(answer):
    """True if the answer says the loan WILL or WILL NOT be granted."""
    for match in re.finditer(DEFINITE_PATTERN, answer, flags=re.IGNORECASE):
        if not re.search(NEGATION_PATTERN, before_in_sentence(answer, match.start()) + " " + match.group(0),
                         flags=re.IGNORECASE):
            return True
    for match in re.finditer(DEFINITE_NO_PATTERN, answer, flags=re.IGNORECASE | re.MULTILINE):
        if not re.search(HEDGE_PATTERN, before_in_sentence(answer, match.start()), flags=re.IGNORECASE):
            return True
    return False


def numbers_in_text(text):
    """All numbers in a text, ignoring references like 'Article 14' or '[S2]'."""
    for pattern in REFERENCE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    found = []
    for match in re.finditer(NUMBER_PATTERN, text):
        whole, decimals, suffix = match.group(1), match.group(2), match.group(3)
        number = float(whole.replace(",", "") + ("." + decimals if decimals else ""))
        if suffix in ["k", "K"]:
            number = number * 1000
        found.append({"text": match.group(0).strip(), "value": number, "percent": suffix == "%"})
    return found


def rounded(value, digits):
    if value == 0:
        return 0.0
    return round(value, digits - 1 - int(math.floor(math.log10(abs(value)))))


def allowed_values(contract, question, chunks):
    """Every number the answer may contain, in the forms a person would write it."""

    values = set()
    # Numbers inside texts of the contract too, e.g. the age group "61+" or a deferral reason
    for item in numbers_in_text(json.dumps(contract_for_prompt(contract), ensure_ascii=False)):
        values.add(item["value"])
    for _, number in all_numbers(contract):
        number = float(number)
        values.add(number)
        values.add(abs(number))
        for digits in [2, 3]:
            values.add(rounded(number, digits))
        if 0 < abs(number) <= 1:
            values.add(round(number * 100, 1))
            values.add(round(number * 100))
    for threshold in [0.7, 0.8, 0.9]:
        values.add(threshold)
        values.add(threshold * 100)
    for small in range(0, 11):
        values.add(float(small))           # "two models", "top 5 features", ...
    for item in numbers_in_text(question):
        values.add(item["value"])
    for chunk in chunks or []:
        for item in numbers_in_text(chunk["text"]):
            values.add(item["value"])
    return values


def is_supported(item, values):
    number = item["value"]
    for value in values:
        tolerance = max(0.005 * abs(value), 0.011)
        if abs(number - value) <= tolerance:
            return True
        if item["percent"] and abs(number - value * 100) <= 0.5:
            return True
    return False


def check_answer(answer, contract, chunks, question):
    """Step 13. Returns what is wrong with the answer (if anything)."""

    unsupported = []
    values = allowed_values(contract, question, chunks)
    for item in numbers_in_text(answer):
        if not is_supported(item, values):
            unsupported.append(item["text"])

    available_tags = [chunk["tag"] for chunk in chunks or []]
    cited = set()
    other = set()
    for text in re.findall(r"\[([^\[\]]{1,60})\]", answer):
        parts = [part.strip() for part in re.split(r"[,;]|\band\b", text) if part.strip()]
        if parts and all(re.fullmatch(r"S\d+", part) for part in parts):
            cited.update(parts)              # "[S1]" or "[S1, S2]"
        else:
            other.add(text)                  # e.g. "[summary.comparison]": not a source
    cited_tags = sorted(cited)
    bad_citations = [tag for tag in cited_tags if tag not in available_tags]
    other_brackets = sorted(other)
    # Recorded for the evaluation, but NOT a reason to retry: an answer that only explains
    # MODEL RESULTS needs no source, and forcing one made Qwen add citations that did not fit
    missing_citation = bool(available_tags) and len(cited_tags) == 0

    review_required = contract["summary"]["human_review_required"]
    mentions_review = re.search(REVIEW_PATTERN, answer, flags=re.IGNORECASE) is not None
    definite_answer = gives_definite_answer(answer)

    problems = []
    if unsupported:
        problems.append("these numbers are not in MODEL RESULTS or SOURCES: " + ", ".join(unsupported))
    if bad_citations:
        problems.append("these source tags do not exist: " + ", ".join(bad_citations))
    if other_brackets:
        problems.append("these are not source tags: " + ", ".join("[" + text + "]" for text in other_brackets)
                        + "; cite only [S1] to [S4], and give no citation for MODEL RESULTS")
    if review_required and not mentions_review:
        problems.append("human_review_required is true, but the answer does not say a human must review the case")
    if review_required and definite_answer:
        problems.append("human_review_required is true, but the answer says the loan will or will not be granted; "
                        "a human must decide, so do not give a yes or no")

    return {
        "passed": len(problems) == 0,
        "problems": problems,
        "unsupported_numbers": unsupported,
        "cited_tags": cited_tags,
        "bad_citations": bad_citations,
        "not_source_tags": other_brackets,
        "missing_citation": missing_citation,
        "review_required": review_required,
        "mentions_review": mentions_review,
        "definite_answer": definite_answer,
    }


REVIEW_SENTENCE = ("Important: this case has been flagged for review by a human loan officer, "
                   "so this is not a final answer.")


def review_fallback(contract):
    """A fixed answer for a flagged case, written from the contract (no LLM)."""
    reasons = "; ".join(contract["deferral"]["reasons"])
    return ("A human loan officer must review this application before any decision can be made. "
            "Reasons: " + reasons + ". The model's estimate is shown above, but it is not a final answer "
            "and this is not a loan offer.")


def make_safe(answer, check, contract):
    """Last safety net after one retry: never show a flagged case with a definite answer or
    without the review notice, and never show unverifiable numbers without a warning."""
    if check["review_required"] and check["definite_answer"]:
        return review_fallback(contract)

    notes = []
    if check["review_required"] and not check["mentions_review"]:
        notes.append(REVIEW_SENTENCE)
    if check["unsupported_numbers"]:
        notes.append("Note: these numbers could not be checked against the model results: "
                     + ", ".join(check["unsupported_numbers"]) + ".")
    if not notes:
        return answer
    return answer.rstrip() + "\n\n" + "\n".join(notes)


# ==================================================================
# Steps 9 - 14 together
# ==================================================================

def retrieve(question, k=4):
    """Step 10, with tags [S1].. [S4] and readable labels."""
    from explainable_ai_project.rag import search
    from explainable_ai_project.rag import source_label

    chunks = search(question, k=k)
    for position, chunk in enumerate(chunks, start=1):
        chunk["tag"] = "S" + str(position)
        chunk["label"] = source_label(chunk)
    return chunks


def ask(applicant_source, question=DEFAULT_QUESTION, mode="full", prompt_version="v1",
        retry=True, log=True, estimated=None, chunks=None):
    """Steps 1 - 14 for one applicant and one question. Returns everything as one dictionary.

    estimated / chunks can be passed in to reuse steps 1 - 8 and step 10
    (the evaluation asks several questions about the same applicant).
    """

    from explainable_ai_project.llm import generate
    from explainable_ai_project.llm import settings as llm_settings

    result = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,                                      # step 9
        "mode": mode,
        "prompt_version": prompt_version,
        "llm": llm_settings(),
    }

    if estimated is None:
        estimated = estimate(applicant_source)
    result["applicant"] = estimated["applicant"]

    if estimated["status"] != "ok":
        result["status"] = "invalid_input"
        result["errors"] = estimated["errors"]
        result["check"] = None
        result["answer"] = ("I cannot assess this application because some values are not valid: "
                            + "; ".join(estimated["errors"]) + ". Please correct them and try again.")
        if log:
            write_log(result)
        return result

    contract = estimated["contract"]
    result["contract"] = contract

    if mode == "full" and chunks is None:
        chunks = retrieve(question)                                 # step 10
    if mode != "full":
        chunks = []
    result["chunks"] = chunks

    messages = build_prompt(question, mode, contract, chunks,       # step 11
                            estimated["applicant"], prompt_version)

    draft = generate(messages)                                      # step 12
    result["draft_answer"] = draft["text"]
    result["seconds"] = draft["seconds"]
    result["tokens"] = {"prompt": draft["prompt_tokens"], "answer": draft["answer_tokens"]}

    if mode == "baseline":
        # Run A has no contract to check against: this is the plain LLM
        result["status"] = "ok"
        result["answer"] = draft["text"]
        result["check"] = None
        if log:
            write_log(result)
        return result

    check = check_answer(draft["text"], contract, chunks, question)  # step 13
    result["retried"] = False
    answer = draft["text"]

    if not check["passed"] and retry:
        feedback = ("Your answer broke these rules:\n- " + "\n- ".join(check["problems"])
                    + "\nWrite the whole answer again and follow all rules.")
        retry_messages = messages + [
            {"role": "assistant", "content": draft["text"]},
            {"role": "user", "content": feedback},
        ]
        second = generate(retry_messages)
        result["retried"] = True
        result["first_check"] = check
        result["seconds"] = round(result["seconds"] + second["seconds"], 2)
        answer = second["text"]
        check = check_answer(answer, contract, chunks, question)

    result["check"] = check
    result["answer"] = make_safe(answer, check, contract)
    result["answer_replaced_by_safety_net"] = result["answer"] != answer
    result["status"] = "ok"

    if log:
        write_log(result)
    return result


# ==================================================================
# Step 14: show the answer and log it
# ==================================================================

def write_log(result):
    """Every interaction -> results/logs/assistant_log.jsonl (gitignored: it holds applicant data).
    Flagged cases also -> results/logs/review_queue.jsonl, for the human reviewer."""

    LOGS_FOLDER.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    contract = result.get("contract")
    if contract and contract["summary"]["human_review_required"]:
        entry = {
            "time": result["time"],
            "question": result["question"],
            "applicant": result["applicant"],
            "reasons": contract["deferral"]["reasons"],
            "assistant_answer": result.get("answer"),
            "status": "waiting for human review",
        }
        with open(REVIEW_QUEUE_FILE, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def money(value):
    if value is None:
        return "none found"
    return "{:,.0f}".format(value)


def show_contract(contract):
    summary = contract["summary"]
    models = contract["models"]
    threshold = str(summary["approval_threshold"])

    print("=" * 70)
    print("LOAN ASSESSMENT (estimate from historical data - not a loan offer)")
    print("=" * 70)
    print("Requested amount:           ", money(summary["requested_amount"]))
    print("Estimated maximum (P >= " + threshold + "):", money(summary["estimated_maximum_amount"]),
          "(model A)" + ("  - highest amount tried, may be higher" if summary["maximum_is_upper_limit"] else ""))
    others = models["A"]["maximum_amount_by_threshold"]
    print("   at other thresholds:      0.7 ->", money(others.get("0.7")), "| 0.9 ->", money(others.get("0.9")))
    print("P(approve) at requested:     A =", models["A"]["p_approve_at_requested_amount"],
          "| B =", models["B"]["p_approve_at_requested_amount"])
    print("Main reasons (" + contract["explanation"]["method"] + "):")
    for feature in contract["explanation"]["top_features"]:
        print("   -", feature["label"], "=", feature["applicant_value"], "->", feature["effect"],
              "(" + "{:+.3f}".format(feature["contribution"]) + ")")
    if contract["input_flags"]:
        print("Input flags:")
        for flag in contract["input_flags"]:
            print("   -", flag)
    if summary["human_review_required"]:
        print("HUMAN REVIEW REQUIRED:")
        for reason in contract["deferral"]["reasons"]:
            print("   -", reason)
    else:
        print("Human review: not required")


def show_answer(result):
    """Step 14."""

    if result["status"] != "ok":
        print(result["answer"])
        return

    show_contract(result["contract"])
    print("-" * 70)
    print("QUESTION:", result["question"])
    print("-" * 70)
    print(result["answer"])
    if result.get("chunks"):
        print("-" * 70)
        print("SOURCES:")
        for chunk in result["chunks"]:
            print("   [" + chunk["tag"] + "]", chunk["label"], "(similarity " + str(chunk["score"]) + ")")
    check = result.get("check")
    if check is not None:
        print("-" * 70)
        status = "passed" if check["passed"] else "problems: " + "; ".join(check["problems"])
        print("Answer check:", status, "| retried:", result.get("retried"),
              "| time:", result.get("seconds"), "s")
    print("=" * 70)


# ==================================================================
# Command line
# ==================================================================

def ask_for_applicant():
    """Type an applicant in, field by field."""
    print("Enter the applicant (press Enter to leave a field empty):")
    applicant = {}
    for field in REQUIRED_FIELDS:
        value = input("  " + FIELD_LABELS[field] + ": ").strip()
        applicant[field] = value if value else None
    return applicant


def run_from_command_line(arguments=None):
    parser = argparse.ArgumentParser(description="Loan assessment assistant (inference pathway)")
    parser.add_argument("--applicant", help="JSON file with one applicant")
    parser.add_argument("--example", choices=["normal", "deferred"], help="use an example applicant from examples/")
    parser.add_argument("--question", help="the user's question")
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--prompt", choices=["v1", "v0"], default="v1")
    parser.add_argument("--no-llm", action="store_true", help="only steps 1 - 8 and the retrieval (no Qwen)")
    options = parser.parse_args(arguments)

    if options.example:
        applicant = EXAMPLES_FOLDER / ("applicant_" + options.example + ".json")
    elif options.applicant:
        applicant = options.applicant
    else:
        applicant = ask_for_applicant()

    question = options.question
    if question is None:
        typed = input("Your question (Enter = '" + DEFAULT_QUESTION + "'): ").strip()
        question = typed or DEFAULT_QUESTION

    if options.no_llm:
        estimated = estimate(applicant)
        if estimated["status"] != "ok":
            print("Invalid input:", "; ".join(estimated["errors"]))
            return
        show_contract(estimated["contract"])
        print("-" * 70)
        print("Contract JSON (what Qwen would receive):")
        print(json.dumps(estimated["contract"], indent=2, ensure_ascii=False))
        try:
            chunks = retrieve(question)
            print("-" * 70)
            print("Sources for:", question)
            for chunk in chunks:
                print("   [" + chunk["tag"] + "]", chunk["label"], "(similarity " + str(chunk["score"]) + ")")
        except (FileNotFoundError, ImportError, OSError) as error:
            print("(retrieval skipped:", error, ")")
        return

    try:
        result = ask(applicant, question, mode=options.mode, prompt_version=options.prompt)
    except ContractError as error:
        print("The contract failed its check, so Qwen was not asked:", error)
        return
    show_answer(result)


if __name__ == "__main__":
    run_from_command_line()
