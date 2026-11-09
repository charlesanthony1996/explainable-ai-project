"""
Inference pathway, steps 11 and 12: the prompt and the local language model.

Qwen runs locally through Ollama, which serves an HTTP API on
127.0.0.1:11434. Nothing leaves the machine, which matters here: the prompt
contains an applicant's financial details.

REPRODUCIBILITY
---------------
Every call uses temperature 0 and a fixed seed. Without both, the same question
gives different answers on different days and the evaluation cannot be repeated
- and repeatability is part of what this project is graded on. The model name,
seed and temperature are returned with every answer so they can be logged.

THE TWO SYSTEM PROMPTS
----------------------
v0 states the task only.
v1 adds the rules on numbers, citations, uncertainty, human review and privacy.

Comparing v0 with v1 is the prompt INTERVENTION in Topic 3: it shows whether a
problem in the answer comes from the model itself or from the instructions it
was given. Both versions must stay in the file for that comparison to be
possible later.

CONTEXT WINDOW
--------------
Ollama defaults to 4096 tokens. The contract plus four chunks plus the question
can exceed that, and anything over the limit is silently dropped - which would
look like the model ignoring its sources. num_ctx is therefore set to 8192,
which still fits comfortably on an 8 GB GPU next to a 7B model in Q4.

Usage
-----
    from explainable_ai_project.llm import ask_qwen, system_prompt
    answer = ask_qwen(prompt, system_prompt("v1"))
"""

import time

import requests


OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen2.5:7b-instruct"

TEMPERATURE = 0.0
SEED = 42
CONTEXT_TOKENS = 8192
TIMEOUT_SECONDS = 300

# How long Ollama keeps the model in VRAM after a request. The default is 5
# minutes, which is a real reproducibility trap: if an evaluation run pauses
# longer than that, the model is unloaded and reloaded, and a FRESHLY LOADED
# model can answer the same prompt differently - measured on this project, a
# cold start changed a 467-token answer into a 346-token one with the same
# seed and temperature 0. Once the model is warm, three identical runs of the
# full prompt gave byte-identical answers.
#
# So: keep it loaded for the whole run, and call warm_up() once before
# measuring anything.
KEEP_ALIVE = "30m"


# ------------------------------------------------------------------
# System prompts
# ------------------------------------------------------------------

SYSTEM_PROMPT_V0 = """You are a loan assessment assistant. You help an applicant
understand the largest loan amount they would likely be approved for, and why.
Answer the applicant's question clearly and briefly."""


SYSTEM_PROMPT_V1 = """You are a loan assessment assistant. You help an applicant
understand the largest loan amount they would likely be approved for, and why.

You must follow these rules.

1. NUMBERS. Use only numbers that appear in the CONTRACT below. Never calculate,
   estimate, round up or invent a figure. If a number you would need is not in
   the contract, say that it is not available.

2. SOURCES. When you state anything about regulation, lender obligations or the
   model's own fairness, cite the source it came from, using the bracket marker
   of the retrieved passage, for example [2]. Do not cite a passage that was not
   given to you.

3. UNCERTAINTY. The estimated maximum is an estimate from past decisions in the
   data, not an offer and not a guarantee. Say so. If the contract marks the
   maximum as an upper limit, explain that the real maximum may be higher but
   the data cannot show it. If no amount was found, say that plainly.

4. HUMAN REVIEW. If the contract marks the case as flagged, state clearly that a
   human reviewer has to decide and do not give a definite recommendation.

5. PRIVACY. Never mention, quote or reconstruct the details of any individual
   applicant from the training data. Speak only about this applicant and about
   aggregated figures.

Write in plain language, in short paragraphs. Do not use jargon without
explaining it."""


SYSTEM_PROMPTS = {
    "v0": SYSTEM_PROMPT_V0,
    "v1": SYSTEM_PROMPT_V1,
}


def system_prompt(version="v1"):
    if version not in SYSTEM_PROMPTS:
        raise ValueError("Unknown prompt version: " + str(version))
    return SYSTEM_PROMPTS[version]


# ------------------------------------------------------------------
# Talking to Ollama
# ------------------------------------------------------------------

def server_is_up():
    """True if Ollama answers."""

    try:
        response = requests.get(OLLAMA_URL + "/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def installed_models():
    response = requests.get(OLLAMA_URL + "/api/tags", timeout=10)
    response.raise_for_status()
    models = response.json().get("models", [])
    return [model["name"] for model in models]


def check_ready(model_name=MODEL_NAME):
    """Fail early and clearly if the model cannot be reached."""

    if not server_is_up():
        raise RuntimeError(
            "Ollama is not answering on " + OLLAMA_URL + ".\n"
            "Start it with:  ollama serve   (or open the Ollama app)"
        )

    names = installed_models()
    if model_name not in names:
        raise RuntimeError(
            "The model " + model_name + " is not installed.\n"
            "Install it with:  ollama pull " + model_name + "\n"
            "Installed: " + (", ".join(names) if names else "none")
        )

    return True


def warm_up(model_name=MODEL_NAME, context_tokens=CONTEXT_TOKENS):
    """Load the model into VRAM before measuring anything.

    Call this once at the start of an evaluation run. See the comment on
    KEEP_ALIVE: a cold model can answer differently from a warm one, so the
    first real question must not be the one that loads it.
    """

    check_ready(model_name)

    payload = {
        "model": model_name,
        "prompt": "ok",
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.0, "seed": SEED,
                    "num_ctx": context_tokens, "num_predict": 1},
    }

    started_at = time.time()
    response = requests.post(OLLAMA_URL + "/api/generate", json=payload,
                             timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    seconds = round(time.time() - started_at, 1)

    print("Model", model_name, "is warm (took", seconds, "s, kept for",
          KEEP_ALIVE + ")")
    return seconds


def ask_qwen(prompt, system_text=None, model_name=MODEL_NAME,
             temperature=TEMPERATURE, seed=SEED, context_tokens=CONTEXT_TOKENS):
    """One answer from the local model.

    Returns a dictionary:
        answer        the text
        model, seed, temperature      what produced it (for the log)
        seconds       how long it took
        prompt_tokens, answer_tokens  as reported by Ollama
    """

    if system_text is None:
        system_text = system_prompt("v1")

    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system_text,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": context_tokens,
        },
    }

    started_at = time.time()
    response = requests.post(
        OLLAMA_URL + "/api/generate",
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    seconds = time.time() - started_at

    body = response.json()

    return {
        "answer": body.get("response", "").strip(),
        "model": model_name,
        "seed": seed,
        "temperature": temperature,
        "context_tokens": context_tokens,
        "seconds": round(seconds, 2),
        "prompt_tokens": body.get("prompt_eval_count"),
        "answer_tokens": body.get("eval_count"),
    }


# ------------------------------------------------------------------
# Building the user prompt for the three configurations
# ------------------------------------------------------------------

def build_prompt(question, applicant_text=None, contract_text=None, chunks_text=None):
    """The user part of the prompt.

    Which pieces are present defines the configuration in the evaluation:
        A   question + applicant values                  (no model, no documents)
        B   question + applicant values + contract       (no documents)
        C   question + applicant values + contract + retrieved passages
    """

    parts = []

    if applicant_text:
        parts.append("APPLICANT\n" + applicant_text)

    if contract_text:
        parts.append(
            "CONTRACT (the only numbers you may use)\n" + contract_text
        )

    if chunks_text:
        parts.append(
            "RETRIEVED PASSAGES (cite these by their bracket marker)\n" + chunks_text
        )

    parts.append("QUESTION\n" + question)

    return "\n\n".join(parts)
