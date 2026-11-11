#!/usr/bin/env bash
#
# Run the whole project end to end.
#
#   ./run_all.sh            everything that needs no language model
#   ./run_all.sh --with-llm everything, including the Ollama stages
#
# The training pathway takes roughly 30 to 60 minutes. The LLM part adds
# about 20 minutes and needs Ollama running with the model pulled.
#
# Stops at the first failure rather than carrying on with stale inputs: a
# later stage reading half-written models produces numbers that look fine
# and are wrong.

set -euo pipefail

WITH_LLM=0
[ "${1:-}" = "--with-llm" ] && WITH_LLM=1

cd "$(dirname "$0")"

# uv installs to ~/.local/bin, which a shell opened before the install does
# not have on PATH yet.
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it from https://docs.astral.sh/uv/ and reopen the shell."
    exit 1
fi

started=$(date +%s)

run () {
    echo
    echo "=================================================================="
    echo ">>> uv run main.py $*"
    echo "=================================================================="
    uv run main.py "$@"
}

# --- training: data, models, fairness, knowledge base --------------------
# One command, steps 1 to 35: load and check the data, split it, train and
# calibrate both forests, run the fairness checks, build the FAISS index.
run

# --- governance and verification, seconds each ---------------------------
run verify_fairness  # parity with bootstrap intervals and stated tolerances
run zones            # auto-accept / sample audit / human review / safe-stop
run provenance       # provenance record and the governance decision
run overview         # the three overview diagrams

# --- anything that talks to the language model ---------------------------
if [ "$WITH_LLM" -eq 1 ]; then
    if ! curl -sf --max-time 5 http://127.0.0.1:11434/api/tags >/dev/null; then
        echo
        echo "Ollama is not answering on 127.0.0.1:11434."
        echo "Start it, then run:  ./run_all.sh --with-llm"
        exit 1
    fi
    run ask       # one applicant, one question, one checked answer
    run evaluate  # the 45 questions through runs A, B and C
else
    echo
    echo "Skipped the stages that need Ollama. For those:  ./run_all.sh --with-llm"
fi

minutes=$(( ($(date +%s) - started) / 60 ))
echo
echo "=================================================================="
echo "Finished in about ${minutes} minutes."
echo "  results/tables/   result tables"
echo "  results/figures/  figures"
echo "  docs/             data quality, datasheet, model card, fairness,"
echo "                    governance decision"
echo "  evaluation/       questions, answers, scoring sheets"
echo
echo "Still by hand:  uv run main.py team_rubric   (the blind scoring page)"
echo "=================================================================="
