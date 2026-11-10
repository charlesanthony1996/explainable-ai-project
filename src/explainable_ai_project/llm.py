"""
Inference pathway, block 2B, step 12: Qwen generates.

Runs Qwen locally through Ollama or LM Studio, with a fixed model, temperature
and seed, so the same prompt gives the same answer (needed for a fair evaluation).

Settings (change them here, or set them as environment variables):

    LLM_BACKEND   "ollama" (default) or "lmstudio"
    LLM_MODEL     Ollama:    "qwen3:4b-instruct" (after: ollama pull qwen3:4b-instruct)
                  LM Studio: the model identifier shown in LM Studio, e.g. "qwen/qwen3-4b-2507"
    LLM_URL       Ollama:    http://localhost:11434   (default)
                  LM Studio: http://localhost:1234    (default; start the server in LM Studio)

Which Qwen: Qwen3-4B-Instruct-2507 (Ollama tag qwen3:4b-instruct). It answers
directly. Ollama's plain "qwen3:4b" tag is the THINKING-only version
(Qwen3-4B-Thinking-2507): it always writes long reasoning first, which made one
answer take over 4 minutes and get cut off, so it is not used.

If a model that can think is used anyway, thinking is switched off in two ways:
    - the text /no_think at the end of the last user message (Qwen3's own switch)
    - "think": false for Ollama (only understood by newer Ollama versions)
and any reasoning that still comes back is cut off before the answer is used.

Check that it works with:
    uv run python -m explainable_ai_project.llm
"""

import os
import re
import time

import requests


LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()

DEFAULT_MODELS = {"ollama": "qwen3:4b-instruct", "lmstudio": "qwen/qwen3-4b-2507"}
DEFAULT_URLS = {"ollama": "http://localhost:11434", "lmstudio": "http://localhost:1234"}

LLM_MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODELS.get(LLM_BACKEND, "qwen3:4b"))
LLM_URL = os.environ.get("LLM_URL", DEFAULT_URLS.get(LLM_BACKEND, "http://localhost:11434")).rstrip("/")

TEMPERATURE = 0.0
SEED = 42
CONTEXT_TOKENS = 8192      # Ollama's default (2048 or 4096) is too small for 4 chunks + the contract
MAX_ANSWER_TOKENS = 700
TIMEOUT_SECONDS = 600


class LLMError(RuntimeError):
    pass


def settings():
    """The exact LLM settings, saved with every answer and in the evaluation results."""
    return {
        "backend": LLM_BACKEND,
        "model": LLM_MODEL,
        "url": LLM_URL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "context_tokens": CONTEXT_TOKENS,
        "max_answer_tokens": MAX_ANSWER_TOKENS,
        "thinking": False,
    }


def remove_thinking(text):
    """Drop the model's reasoning if it wrote some anyway.

    Ollama sometimes leaves out the opening <think> tag, so everything up to the
    LAST </think> is removed, not only what sits between the two tags.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def add_no_think(messages):
    """A copy of the messages with Qwen3's switch /no_think at the end of the last user message.

    Instruct models never think, so they get the messages unchanged.
    """
    messages = [dict(message) for message in messages]
    if "instruct" in LLM_MODEL.lower():
        return messages
    for message in reversed(messages):
        if message["role"] == "user":
            message["content"] = message["content"] + "\n/no_think"
            break
    return messages


def call_ollama(messages):
    body = {
        "model": LLM_MODEL,
        "messages": add_no_think(messages),
        "stream": False,
        "think": False,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_ctx": CONTEXT_TOKENS,
            "num_predict": MAX_ANSWER_TOKENS,
        },
    }
    response = requests.post(LLM_URL + "/api/chat", json=body, timeout=TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise LLMError("Ollama answered " + str(response.status_code) + ": " + response.text[:300])
    data = response.json()
    return {
        "text": data["message"]["content"],
        "prompt_tokens": data.get("prompt_eval_count"),
        "answer_tokens": data.get("eval_count"),
    }


def call_lmstudio(messages):
    body = {
        "model": LLM_MODEL,
        "messages": add_no_think(messages),
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_tokens": MAX_ANSWER_TOKENS,
        "stream": False,
    }
    response = requests.post(LLM_URL + "/v1/chat/completions", json=body, timeout=TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise LLMError("LM Studio answered " + str(response.status_code) + ": " + response.text[:300])
    data = response.json()
    usage = data.get("usage") or {}
    return {
        "text": data["choices"][0]["message"]["content"],
        "prompt_tokens": usage.get("prompt_tokens"),
        "answer_tokens": usage.get("completion_tokens"),
    }


def generate(messages):
    """Step 12: send the messages to Qwen; return the answer text, token counts and time taken.

    messages = [{"role": "system", "content": ...}, {"role": "user", "content": ...}, ...]
    """

    start = time.perf_counter()
    try:
        if LLM_BACKEND == "ollama":
            result = call_ollama(messages)
        elif LLM_BACKEND == "lmstudio":
            result = call_lmstudio(messages)
        else:
            raise LLMError("LLM_BACKEND must be 'ollama' or 'lmstudio', not " + repr(LLM_BACKEND))
    except requests.exceptions.ConnectionError:
        raise LLMError("Cannot reach " + LLM_BACKEND + " at " + LLM_URL + ". Is it running? "
                       "(Ollama: open the Ollama app. LM Studio: Developer tab -> Start Server.)")

    result["text"] = remove_thinking(result["text"])
    result["seconds"] = round(time.perf_counter() - start, 2)
    result["settings"] = settings()
    return result


def check_llm():
    """Is the server running, and is the model there? Prints what it finds."""

    print("LLM settings:", settings())
    try:
        if LLM_BACKEND == "ollama":
            response = requests.get(LLM_URL + "/api/tags", timeout=10)
            names = [model["name"] for model in response.json().get("models", [])]
        else:
            response = requests.get(LLM_URL + "/v1/models", timeout=10)
            names = [model["id"] for model in response.json().get("data", [])]
    except requests.exceptions.ConnectionError:
        print("Cannot reach", LLM_BACKEND, "at", LLM_URL, "- is it running?")
        return False

    print("models available:", names)
    if LLM_MODEL not in names:
        print("WARNING:", LLM_MODEL, "is not in that list. Ollama: ollama pull", LLM_MODEL,
              "| LM Studio: set LLM_MODEL to one of the names above")
        return False

    answer = generate([{"role": "user", "content": "In one sentence: what is a credit score?"}])
    print("test answer (" + str(answer["seconds"]) + " s):", answer["text"])
    return True


if __name__ == "__main__":
    check_llm()
