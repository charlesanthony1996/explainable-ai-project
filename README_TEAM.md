# The common version

One branch with everything. Built from `dayas`, with four governance
commands and the scoring page added from `michael`.

## Run it

```bash
./run_all.sh              # Linux / macOS
.\run_all.ps1             # Windows
```

Training pathway plus the governance commands: about 30 to 60 minutes.
Add `--with-llm` / `-WithLlm` for the stages that need Ollama (about 20
minutes more). It stops at the first failure instead of carrying on with
half-written models.

## What came from where

| From `dayas` | |
|---|---|
| The whole training pathway (steps 1–35) | Charles's modules, unchanged |
| Inference: `uv run main.py ask` | contract, sources, answer check, retry |
| Evaluation: `uv run main.py evaluate` | 45 questions, 7 applicants, runs A/B/C |
| All six Topic 3 checks | zero-shot, SHAP vs LIME, age pairs, privacy, deferral compliance, prompt intervention |

| From `michael` | |
|---|---|
| `uv run main.py verify_fairness` | parity metrics with bootstrap confidence intervals, and the definitions written down: protected group, favourable outcome, reference population, tolerance and where each tolerance comes from |
| `uv run main.py zones` | four control zones instead of one deferral flag: auto-accept, sample audit, human review, safe-stop |
| `uv run main.py provenance` | provenance record, and the metric conflict recorded as a governance decision |
| `uv run main.py overview` | three diagrams: the whole programme, the evaluation, the governance |
| `uv run main.py team_rubric` | the blind scoring page for the evaluation sheet |

The two branches built the inference and evaluation pathways in parallel
without knowing: `michael` on 9 Nov 16:25, `dayas` on 10 Nov 11:59. On
9 Nov there was nothing on `dayas` but Charles's commits. `dayas` has the
more complete evaluation, so that is the one here. The counterfactual
age check on `michael` is a second implementation of the same idea and
was left out.

## What still needs a person

**The blind scoring.** `evaluation/llm_scores.csv` has 135 answers with
the automatic number check filled in, and every human column empty:
`correctness`, `faithfulness`, `citations`, `correct_refusal`. Both
age-pair review sheets are empty too.

```bash
uv run main.py team_rubric        # builds evaluation/team_rubric_sheet.html
uv run main.py team_rubric --read # reads the filled sheets back
```

The page rates a stratified subsample — 10 questions across all eight
categories, all three runs each, 30 answers. Seeded, so every rater gets
the same thirty: Q01, Q10, Q11, Q16, Q17, Q22, Q28, Q33, Q39, Q42.

The run that produced each answer is hidden, which the sheet already
supported: `answer_id` is a hash, and `applicant_facts` and
`sources_given` are filled for all 135 rows rather than only for the runs
that received them.

**Two people, and calibrate first.** The rubric says it and it is worth
following: score the first few answers together and agree on every score
before splitting. A first attempt that skipped this produced a Cohen's
kappa of −0.04 — the two raters ranked the systems identically but used
the scale a point apart, and the agreement figure is unusable.

## One known fault

`evaluation/applicants.json` gives the `older_flagged` applicant
`loan_purpose: "Personal"`. The allowed values are `Education`, `Home`,
`Personal or other`, `Vehicle`. The preprocessor runs with
`handle_unknown="ignore"`, so the value is silently encoded as none of
the four and no rule catches it — `missing_data` looks for an empty
field, not a wrong one.

It does not affect the A/B/C comparison, because the same applicant with
the same encoding goes through all three runs. It belongs in the
limitations.
