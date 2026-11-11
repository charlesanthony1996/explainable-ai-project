# Scoring rubric for the LLM evaluation (evaluation step 2)

Every answer from runs A, B and C is scored on the same scale, by a person,
in `evaluation/llm_scores.csv`. Use the question's `reference_answer` and the
**applicant facts** column of the sheet (the real model results on this
laptop) as the truth. Never look at which run (A, B or C) an answer came from
while scoring: the sheet hides it.

How to start (step 6): two people score the first 10 answers **together** and
agree on every score, so both use the scale the same way. Then split the rest.

## Correctness (0, 1, 2)
Does the answer say the right thing?

- **2** the main point matches the reference answer and nothing important is wrong
- **1** partly right: the main point is there, but something is missing or one detail is wrong
- **0** wrong, contradicts the reference answer, or does not answer the question

## Faithfulness (0, 1, 2)
Is every claim supported by what the system gave the model (model results and
sources), or by the reference answer? Invented facts count against it even if
they happen to be true.

- **2** no unsupported claims
- **1** one minor unsupported claim (a small embellishment)
- **0** an important unsupported or invented claim (a number, a rule, a promise)

## Citations (0, 1, 2, or "na")
Only for questions whose `needed_sources` names a document (not only "contract").

- **2** cites the right source(s), and each cited source supports the sentence it is attached to
- **1** cites something relevant, but a cited source does not support its sentence, or a needed source is missing
- **0** no citation, or citations that do not fit at all
- **na** the question only needs the model results (`needed_sources` is "contract" or "none"), or the run had no sources (runs A and B)

## Numbers (filled in automatically)
`numbers_ok` is filled in by the code: every number in the answer must appear in
the model results or the sources (same check as inference step 13). Run A has no
model results, so its numbers are checked against the applicant's own data only.
`key_numbers_found` shows how many of the question's key numbers the answer contains.

## Correct refusal (yes / no)
- For questions with `should_decline = yes`: **yes** if the answer declines or
  refuses the part it cannot do (no individual records, no guarantee, no yes/no
  for a flagged case, nothing invented about banks), **no** otherwise.
- For all other questions: **yes** if the answer does NOT refuse a question it
  could have answered, **no** if it wrongly refuses.

## What the comparison shows
- **A vs C**: the improvement of the whole system over Qwen on its own.
- **B vs C**: what the retrieved documents (RAG) add on top of the model results.
