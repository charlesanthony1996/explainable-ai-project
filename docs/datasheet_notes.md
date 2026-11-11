# Datasheet notes (written by the team)

This file is created once and never overwritten. Its text is copied into
docs/datasheet.md every time the reports step runs.

## How the approval labels were made

Not documented for either dataset, and we could not reconstruct it. What the
data shows is that both label columns were generated rather than observed. In
dataset 2 a single column, `RiskScore`, predicts approval with an AUC of
0.9873, which is what a label looks like when it was derived from one formula.
In dataset 1 the approval probability rises with the requested amount, which
no real approval process does.

We treat the labels as "what this generated process decided", not as real
lending outcomes.

## Evidence that the data is synthetic

Dataset 1: average approval probability rises from 0.434 at 5,000 to 0.650 at
its peak as the requested amount grows. Nobody requests more than 40 percent
of their income.

Dataset 2: `RiskScore` alone reaches an AUC of 0.9873.

Neither pattern occurs in real lending data.

## Leakage and derived columns

`RiskScore` (dataset 2, single-feature AUC 0.9873) was excluded as target
leakage. Three further columns are calculated from the loan amount and are
also not model inputs: `MonthlyLoanPayment` (R² 0.598 from the amount),
`TotalDebtToIncomeRatio` (0.198) and `BaseInterestRate` (0.138). Using any of
them would have let the model read the answer out of the question.

Dataset 1 showed no leakage; its highest single-feature AUC is 0.8154
(`Credit_Score`).

## Limitations

Currency is not stated in either dataset. The two credit score ranges differ,
so scores are not comparable between them. Neither dataset records a true
maximum loan amount, so the sweep estimates one and often hits the upper limit
of the search range. The 61+ group is small - 390 test rows in dataset 1 and
only 120 in dataset 2, which is why four of dataset 2's eight parity metrics
come out undecided.

## Privacy and consent

Both datasets are synthetic. No natural person is described, so consent does
not arise and GDPR data-subject rights do not apply. No individual record is
written into the RAG documents: the knowledge base holds only aggregated
reports, so the assistant cannot quote a training row even if asked.

This would all have to be reconsidered before any real applicant data is used.

## Licence

Dataset 1 is course material and is not redistributable outside the course.

Dataset 2 comes from Kaggle; **the licence on its Kaggle page still has to be
checked and named here before this datasheet is published anywhere.**
