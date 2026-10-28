# Feature glossary: what every input and output means

This glossary explains, in plain words, every value the loan assessment assistant
uses or reports.

## Inputs (what the applicant provides)

- **Age:** the applicant's age in years. Used by the models; also used to form age
  groups (18-30, 31-45, 46-60, 61+) for the fairness checks.
- **Dependents:** the number of people who depend financially on the applicant.
- **Annual income:** yearly income before tax, in the dataset's currency (not stated in the data).
- **Credit score:** a number summarising the applicant's credit history; higher is better.
  The two datasets use different ranges: about 300-850 in dataset 1 and about 340-710 in dataset 2.
- **Loan amount:** the amount of money requested.
- **Loan term:** how long the loan runs, in months.
- **Marital status, education, employment status, loan purpose:** categories describing
  the applicant and the loan.

## Values the system calculates

- **Loan-to-income:** the loan amount divided by the annual income. 0.3 means the loan
  equals 30 % of one year's income. Higher values make approval less likely.
- **P(approve):** the model's calibrated probability that a loan of this amount would be
  approved for this applicant. 0.8 means: among applicants given 0.8, about 80 % were approved.
- **Estimated maximum amount:** the highest loan amount for which P(approve) is still at
  least 0.8. It is an estimate based on past decisions in the data, not a promise.
- **Upper limit flag:** when the estimated maximum equals the highest amount the system was
  allowed to check, the real maximum may be higher; the data does not show it.
- **No amount found:** even the smallest amount checked does not reach P(approve) 0.8.

## Deferral (sending a case to a human)

- **Borderline:** P(approve) is very close to the 0.8 threshold.
- **Unusual input:** a value lies outside the range seen in the training data.
- **Missing data:** a required field is empty.
- **Disagreement:** the two models (trained on different datasets) give very different answers.
- **Weak group:** the applicant belongs to an age group the model cannot assess reliably.

## Models

- **Model A:** trained on dataset 1 (the course dataset).
- **Model B:** trained on dataset 2 (Financial Risk for Loan Approval).
- Both are Random Forests with a monotonic rule: asking for a larger loan can never raise
  P(approve). Their probabilities are calibrated.
