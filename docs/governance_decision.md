# Governance decision: which fairness criterion this project holds itself to

Written on 2026-11-11. Accountable: Michael Aichinger and Charles Anthony, jointly.

## The conflict, in numbers

Model A on dataset 1, cut-off 0.5, test split, 95 percent bootstrap intervals
from `results/tables/parity_table_with_intervals.csv`:

| Criterion | Value | Interval | Tolerance | Verdict |
|---|---|---|---|---|
| Predictive parity difference | 0.005 | 0.004 - 0.038 | at most 0.10 | within tolerance |
| Demographic parity ratio | 0.000 | 0.000 - 0.000 | at least 0.80 | breaches |
| Demographic parity difference | 0.834 | 0.822 - 0.845 | at most 0.10 | breaches |
| Equal opportunity difference | 0.966 | 0.959 - 0.973 | at most 0.10 | breaches |

The model is near-identical in precision across age groups and approves none
of the applicants aged 61 or over, of whom 15.4 percent were in fact approved.

This is not a defect to be repaired. Chouldechova (2017) and Kleinberg,
Mullainathan and Raghavan (2016) show that calibration, predictive parity and
equal odds cannot hold together when base rates differ between groups, and
here they differ sharply: the median credit score is 438 in the 61+ group
against 757 in the 31-45 group. Any model on this data must fail at least one
criterion. The project therefore has to choose which, and say so.

## The decision

**The project holds itself to equal opportunity and reports the breach rather
than removing it.**

Approving nobody in a group whose real approval rate is 15.4 percent is a
harm to individuals who would have been approved. Predictive parity holding
at the same time does not undo that; it only means the few approvals the
model does hand out are equally reliable across groups.

We did not repair the breach, for a reason the measurements support: removing
`age` from the model changes nothing, because the credit score carries the
same information (step 28). A fix that leaves the mechanism in place and only
hides the variable would make the metric look better while changing no
applicant's outcome.

Instead the breach is declared, routed and documented:

- every applicant in the 61+ group is deferred to a human, because the model's
  within-group AUC there is 0.524, which is chance;
- the fairness report naming this is part of the knowledge base the assistant
  retrieves, so the system can be asked about its own bias and answers from
  the measurement;
- the model card states that the model is not fit for use on applicants aged
  61 or over.

## Affected groups and error costs

| Group | n (test) | What the breach costs them |
|---|---|---|
| 61+ | 390 | Every applicant is refused automatically and sent to a human. Those who would have been approved - about 15 in 100 - depend entirely on that reviewer. |
| 46-60 | 1,533 | Approved below their real rate; the gap is small and inside the interval. |
| 18-30, 31-45 | 5,877 | No measured disadvantage. |

The cost is asymmetric. A false refusal denies credit to someone who would
have repaid; a false approval creates a loan that may default. Both are real,
they fall on different people, and this project does not have the data to
price them against each other. Saying so is more honest than inventing a
ratio.

## Tolerated gap

| Metric | Limit | Where it comes from |
|---|---|---|
| Demographic parity ratio | at least 0.80 | four-fifths rule, EEOC 29 CFR 1607.4D. A convention from US employment law with no standing in EU credit law, used as a warning line. |
| Demographic parity difference | at most 0.10 | **declared by this project.** No external authority. |
| Equal opportunity difference | at most 0.10 | **declared by this project.** No external authority. |
| Predictive parity difference | at most 0.10 | **declared by this project.** No external authority. |

A metric counts as within tolerance only when its whole confidence interval
sits inside the limit, and as a breach only when all of it sits outside.
Anything else is undecided at this sample size - which is the verdict for
four of dataset 2's eight metric-and-cut-off combinations, because its 61+
group holds 120 rows.

## Remediation trigger

Any of the following would require the model to be withdrawn from automatic
use for the affected group:

1. the demographic parity ratio interval moving below 0.80 for a group not
   already deferred;
2. within-group AUC falling below 0.60 for a group not already deferred, the
   same rule that currently catches 61+;
3. deferral exceeding 25 percent of all cases, at which point the automation
   no longer carries its own weight;
4. any use of real applicant data, which invalidates this record entirely -
   it rests throughout on both datasets being synthetic.

## Review

none. The project ran for about two weeks and ends at submission; no operational owner exists afterwards and no review is scheduled. Naming a future date would imply a responsibility nobody holds. In a deployment this row would carry a named risk owner and a recurring review, and its absence here is itself a governance finding.

## What was rejected, and why

A second control was considered and **rejected on evidence**: deferring cases
where models A and B disagree by more than 0.5. The rule was tested on the
validation split and switched off, because the model is not worse on the
cases it catches. On dataset 1 it is better - 0.8551 against 0.8501 overall -
and on dataset 2 the drop of 0.0126 falls short of the 0.03 a rule must clear.
Switching it on would have sent 2,257 of about 7,800 validation cases to
human review for no measured gain.

The intuition that disagreement means difficulty is reasonable and, here,
wrong. `results/tables/deferral_rule_choice.csv` holds the figures.
