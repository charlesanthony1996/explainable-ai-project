# Model card notes (written by the team)

This file is created once and never overwritten. Its text is copied into
docs/model_card.md every time step 23 runs.

## Intended use

Academic proof of concept for the HC-EAI capstone. It estimates the largest
loan amount an applicant would likely have been approved for in the training
data and explains that estimate in plain language. It supports a person and
never decides alone.

## Out of scope

Real lending decisions. Applicants outside the training ranges (dataset 1
holds nobody over 69 and nobody requesting more than 40 percent of their
income). Applicants aged 61 or over on model A, where the within-group AUC is
0.524, which is chance. Any use without human review: credit scoring is
high-risk under the EU AI Act, Annex III.

## Ethical considerations

Model A approves 0.0 percent of applicants aged 61 or over, whose real
approval rate is 15.4 percent. Demographic parity ratio 0.000, equal
opportunity difference 0.966 - both far outside sampling noise.

The cause is the credit score, not the age field: removing `age` changes
nothing, income and score explain at most 14.3 percent of the gap, and the
median score is 438 in that group against 757 in the 31-45 group. The model
does not weigh old against young; it has no usable signal for older applicants
and refuses when uncertain.

Predictive parity holds at the same time (0.005 where it can be measured).
Both criteria cannot hold when base rates differ this much. The project holds
itself to equal opportunity and reports the breach instead of removing it,
because suppressing `age` would improve the metric and change nobody's
outcome. See docs/governance_decision.md.

Deferral moves the problem rather than solving it: the whole 61+ group goes to
a human, and that does not make the human's decision fair.

## Caveats

Both datasets are synthetic - in dataset 1 the approval probability *rises*
with the requested amount, which no real lender does; the monotonic
constraints exist to repair that.

Dataset 1's estimated maximum often sits at the upper sweep limit, so the real
maximum may be higher and the data cannot show it.

The EBM is more accurate on both datasets (0.8168 vs 0.8159; 0.9385 vs
0.9272). The Random Forest was chosen anyway, for monotonic constraints and
exact TreeSHAP - a deliberate trade of accuracy for guarantees.

The number check verifies where a figure came from, not whether the sentence
around it is true: one answer scored 6 of 7 while inventing a maximum for an
applicant who had none.
