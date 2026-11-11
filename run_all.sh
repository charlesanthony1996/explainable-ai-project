#!/usr/bin/env bash
#
# Run the whole project end to end.
#
#   ./run_all.sh            everything that needs no language model
#   ./run_all.sh --with-llm everything, including the Ollama stages
#
# The offline part takes roughly 30 to 60 minutes, almost all of it in
# `models`. The LLM part adds about 20 minutes and needs Ollama running with
# the model already pulled.
#
# Stops at the first failure rather than carrying on with stale inputs: a
# later stage reading half-written models would produce numbers that look
# fine and are wrong.

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
    echo ">>> $*"
    echo "=================================================================="
    uv run main.py "$@"
}

# --- offline: data, models, fairness, explanations, knowledge base --------
run data             # steps 1-11   clean, check, split, preprocessor
run models           # steps 12-23  baselines, monotonic forests, calibration
run fairness         # steps 24-29  parity, deferral rules, fairness report
run shap             # topic 2      TreeSHAP, global and per applicant
run rag              # steps 30-35  documents, chunks, embeddings, FAISS

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
    run ask              # one applicant, one question, one checked answer
    run evaluate         # 24 questions through configurations A, B and C
    run counterfactual   # topic 3 check 9: the same applicant at 30 and 67
    run rubric           # read back whatever human scoring exists
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
echo "=================================================================="
