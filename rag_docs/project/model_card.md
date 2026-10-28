# Model card: loan maximum-amount assistant

Numbers filled in automatically on 2026-10-28 09:24. The team's text comes from `docs/model_card_notes.md`.

## Model details

- **Final models:** model A (dataset 1) and model B (dataset 2), each a Random Forest classifier.
- **Monotonic constraint:** loan_amount and loan_to_income may only lower the approval probability, so the maximum-amount sweep is valid (checked in step 20).
- **Calibration:** a correction learned on the validation part, so probabilities can be taken literally (step 19).
- **Output:** P(approve) at the requested amount, and the highest amount with P(approve) >= 0.8 (also 0.7 and 0.9).
- **Inputs:** age, dependents, annual income, credit score, loan amount, loan term, loan-to-income, marital status, education, employment status, loan purpose.

## Tuned settings

| dataset | n_estimators | max_depth | min_samples_leaf | max_features |
| --- | --- | --- | --- | --- |
| dataset_1 | 200 | 12 | 20 | 0.5 |
| dataset_2 | 200 | 12 | 10 | sqrt |

## Calibration

| dataset | method | brier_raw | brier_calibrated_out_of_fold | chosen |
| --- | --- | --- | --- | --- |
| dataset_1 | sigmoid | 0.1287 | 0.1281 | False |
| dataset_1 | isotonic | 0.1287 | 0.1278 | True |
| dataset_2 | sigmoid | 0.0823 | 0.08 | False |
| dataset_2 | isotonic | 0.0823 | 0.0789 | True |

## Results on the untouched test part

Every model scored once on the test part (step 21). The final model is random_forest_monotonic_calibrated.

| dataset | model | auc | accuracy | precision | recall | f1 | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | ebm_monotonic | 0.8168 | 0.8471 | 0.8463 | 0.9307 | 0.8865 | 0.1296 |
| dataset_1 | random_forest_monotonic | 0.8159 | 0.8471 | 0.8463 | 0.9307 | 0.8865 | 0.1307 |
| dataset_1 | random_forest_monotonic_calibrated | 0.8159 | 0.8471 | 0.8463 | 0.9307 | 0.8865 | 0.1299 |
| dataset_1 | random_forest_unconstrained | 0.8143 | 0.8471 | 0.8463 | 0.9307 | 0.8865 | 0.1302 |
| dataset_1 | logistic_regression | 0.8138 | 0.8467 | 0.8458 | 0.9307 | 0.8862 | 0.1353 |
| dataset_2 | ebm_monotonic | 0.9385 | 0.882 | 0.7826 | 0.7019 | 0.7401 | 0.0821 |
| dataset_2 | logistic_regression | 0.9369 | 0.884 | 0.7937 | 0.6964 | 0.7418 | 0.0832 |
| dataset_2 | random_forest_unconstrained | 0.9329 | 0.8823 | 0.7911 | 0.6908 | 0.7375 | 0.0856 |
| dataset_2 | random_forest_monotonic | 0.9278 | 0.87 | 0.7877 | 0.6253 | 0.6972 | 0.0909 |
| dataset_2 | random_forest_monotonic_calibrated | 0.9272 | 0.8757 | 0.7386 | 0.7437 | 0.7412 | 0.0893 |

## Confusion matrices (cut-off 0.5)

| dataset | model | correctly_rejected | wrongly_approved | wrongly_rejected | correctly_approved |
| --- | --- | --- | --- | --- | --- |
| dataset_1 | random_forest_monotonic_calibrated | 1949 | 846 | 347 | 4658 |
| dataset_2 | random_forest_monotonic_calibrated | 2093 | 189 | 184 | 534 |

## Monotonicity check

| dataset | model | applicants_checked | amounts_per_applicant | lowest_amount | highest_amount | curves_that_rise | largest_rise | passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | calibrated model A | 500 | 50 | 5000 | 44848 | 0 | 0.0 | True |
| dataset_2 | calibrated model B | 500 | 50 | 3674 | 184732 | 0 | 0.0 | True |

## Maximum-amount sweep on the test part

| dataset | threshold | share_with_no_amount | share_of_maximums_at_upper_limit | median_max_amount | median_requested_amount | consistency |
| --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 0.7 | 0.2944 | nan | 37299.0 | 21071.0 | 0.8431 |
| dataset_1 | 0.8 | 0.2944 | 0.3823 | 37291.0 | 21071.0 | 0.8431 |
| dataset_1 | 0.9 | 0.5932 | nan | 19903.0 | 21071.0 | 0.3591 |
| dataset_2 | 0.7 | 0.3853 | nan | 11996.0 | 22046.0 | 0.865 |
| dataset_2 | 0.8 | 0.6743 | 0.0 | 15522.0 | 22046.0 | 0.8363 |
| dataset_2 | 0.9 | 0.765 | nan | 15237.0 | 22046.0 | 0.8123 |

## Revocation demo (one applicant removed; ID kept out of this document)

| date | dataset | training_rows_before | training_rows_after | old_model_md5 | new_model_md5 | validation_auc_before | validation_auc_after | mean_prediction_change | largest_prediction_change | removed_applicant_p_before | removed_applicant_p_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-28 09:24 | dataset_1 | 36400 | 36399 | 22710bcb33fff269f83d4029ce1e89c5 | 19f597203e2c98c0987d1c7f3565860a | 0.817 | 0.8173 | 0.00644 | 0.04858 | 0.2229 | 0.2266 |

---

# Model card notes (written by the team)

This file is created once and never overwritten. Its text is copied into
docs/model_card.md every time step 23 runs.

## Intended use
TODO (academic proof of concept: estimate the largest loan amount an applicant
would likely be approved for, and explain it; supports a human, never decides alone)

## Out of scope
TODO (real lending decisions; applicants outside the training ranges;
any use without human review - credit scoring is high-risk under the EU AI Act, Annex III)

## Ethical considerations
TODO (age differences in approval; what the fairness checks in 1C found)

## Caveats
TODO (synthetic data; dataset 1's maximum often equals the upper sweep limit;
the EBM was as accurate as the Random Forest on the test part)
