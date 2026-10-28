# Fairness report: loan assessment models

Written automatically on 2026-10-28 09:36 from the results of the Topic 3 checks (training steps 24 to 28). Aggregated numbers only.

## 1. What this report covers

This fairness report checks whether the loan assessment models treat applicants of different ages differently. Age groups are 18-30, 31-45, 46-60 and 61+. Model A was trained on dataset 1 (the course dataset), model B on dataset 2 (Financial Risk for Loan Approval). All numbers come from the untouched test part and are aggregated per age group; no individual applicant is shown. Age is used for these checks only and is a model input in the original models.

## 2. Approval rates by age group

This section compares, per age group, how often applicants were really approved and how often each model would approve them.

**Key findings:**

- Model A (dataset 1) approves 0.0 % of applicants aged 61+ at cut-off 0.5; 15.4 % of them were really approved.
- Model B (dataset 2) approves 34.2 % of applicants aged 61+ at cut-off 0.5; 30.8 % of them were really approved.
- Model A (dataset 1): demographic parity ratio 0.0 (below 0.8, a common warning sign), equal opportunity difference 0.966.
- Model B (dataset 2): demographic parity ratio 0.429 (below 0.8, a common warning sign), equal opportunity difference 0.089.

*Per age group*

| dataset | cut_off | age_group | applicants | actual_approval_rate | predicted_approval_rate | true_positive_rate | precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 0.5 | 18-30 | 1888 | 0.633 | 0.693 | 0.922 | 0.843 |
| dataset_1 | 0.5 | 31-45 | 3989 | 0.731 | 0.834 | 0.966 | 0.847 |
| dataset_1 | 0.5 | 46-60 | 1533 | 0.543 | 0.568 | 0.887 | 0.848 |
| dataset_1 | 0.5 | 61+ | 390 | 0.154 | 0.0 | 0.0 | 0.0 |
| dataset_1 | 0.8 | 18-30 | 1888 | 0.633 | 0.692 | 0.92 | 0.842 |
| dataset_1 | 0.8 | 31-45 | 3989 | 0.731 | 0.832 | 0.964 | 0.847 |
| dataset_1 | 0.8 | 46-60 | 1533 | 0.543 | 0.568 | 0.887 | 0.848 |
| dataset_1 | 0.8 | 61+ | 390 | 0.154 | 0.0 | 0.0 | 0.0 |
| dataset_2 | 0.5 | 18-30 | 675 | 0.141 | 0.147 | 0.695 | 0.667 |
| dataset_2 | 0.5 | 31-45 | 1397 | 0.243 | 0.244 | 0.726 | 0.724 |
| dataset_2 | 0.5 | 46-60 | 808 | 0.304 | 0.3 | 0.78 | 0.793 |
| dataset_2 | 0.5 | 61+ | 120 | 0.308 | 0.342 | 0.784 | 0.707 |
| dataset_2 | 0.8 | 18-30 | 675 | 0.141 | 0.059 | 0.368 | 0.875 |
| dataset_2 | 0.8 | 31-45 | 1397 | 0.243 | 0.099 | 0.359 | 0.884 |
| dataset_2 | 0.8 | 46-60 | 808 | 0.304 | 0.149 | 0.435 | 0.892 |
| dataset_2 | 0.8 | 61+ | 120 | 0.308 | 0.142 | 0.432 | 0.941 |

*Gaps between age groups*

| dataset | cut_off | demographic_parity_difference | demographic_parity_ratio | equal_opportunity_difference | predictive_parity_difference | most_approved_group | least_approved_group |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 0.5 | 0.834 | 0.0 | 0.966 | 0.848 | 31-45 | 61+ |
| dataset_1 | 0.8 | 0.832 | 0.0 | 0.964 | 0.848 | 31-45 | 61+ |
| dataset_2 | 0.5 | 0.195 | 0.429 | 0.089 | 0.127 | 61+ | 18-30 |
| dataset_2 | 0.8 | 0.089 | 0.399 | 0.076 | 0.066 | 46-60 | 18-30 |

## 3. Maximum loan amount by age group

This section checks which age groups receive an estimated maximum loan amount (approval probability at least 0.8), and whether income or credit score explain the differences. 'Adjusted' rates assume every age group had the same mix of incomes and credit scores.

**Key findings:**

- Model A (dataset 1): the gap between age groups in getting an amount is 0.834; with the same income mix it is 0.761, with the same income and credit score mix 0.715. The gap is NOT explained by income or credit score (at most 14.3 % of it).
- Model B (dataset 2): the gap between age groups in getting an amount is 0.284; with the same income mix it is 0.117, with the same income and credit score mix 0.12. The gap is mostly explained by income (about 58.8 % of it).

*Per age group*

| dataset | age_group | applicants | share_getting_an_amount | median_maximum | median_income | median_credit_score |
| --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 18-30 | 1888 | 0.693 | 37281.0 | 78508.0 | 740.0 |
| dataset_1 | 31-45 | 3989 | 0.834 | 37278.0 | 90272.0 | 757.0 |
| dataset_1 | 46-60 | 1533 | 0.568 | 37330.0 | 67850.0 | 716.0 |
| dataset_1 | 61+ | 390 | 0.0 | nan | 43856.0 | 438.0 |
| dataset_2 | 18-30 | 675 | 0.182 | 15157.0 | 42093.0 | 560.0 |
| dataset_2 | 31-45 | 1397 | 0.319 | 15309.0 | 48642.0 | 575.0 |
| dataset_2 | 46-60 | 808 | 0.436 | 16280.0 | 54330.0 | 596.0 |
| dataset_2 | 61+ | 120 | 0.467 | 16944.0 | 60624.0 | 616.0 |

*Adjusted rates*

| dataset | adjustment | age_group | share_getting_an_amount | gap_between_groups |
| --- | --- | --- | --- | --- |
| dataset_1 | not adjusted | 18-30 | 0.693 | 0.834 |
| dataset_1 | not adjusted | 31-45 | 0.834 | 0.834 |
| dataset_1 | not adjusted | 46-60 | 0.568 | 0.834 |
| dataset_1 | not adjusted | 61+ | 0.0 | 0.834 |
| dataset_1 | same income mix | 18-30 | 0.703 | 0.761 |
| dataset_1 | same income mix | 31-45 | 0.761 | 0.761 |
| dataset_1 | same income mix | 46-60 | 0.654 | 0.761 |
| dataset_1 | same income mix | 61+ | 0.0 | 0.761 |
| dataset_1 | same credit score mix | 18-30 | 0.702 | 0.739 |
| dataset_1 | same credit score mix | 31-45 | 0.739 | 0.739 |
| dataset_1 | same credit score mix | 46-60 | 0.674 | 0.739 |
| dataset_1 | same credit score mix | 61+ | 0.0 | 0.739 |
| dataset_1 | same income and credit score mix | 18-30 | 0.709 | 0.715 |
| dataset_1 | same income and credit score mix | 31-45 | 0.715 | 0.715 |
| dataset_1 | same income and credit score mix | 46-60 | 0.695 | 0.715 |
| dataset_1 | same income and credit score mix | 61+ | 0.0 | 0.715 |
| dataset_2 | not adjusted | 18-30 | 0.182 | 0.284 |
| dataset_2 | not adjusted | 31-45 | 0.319 | 0.284 |
| dataset_2 | not adjusted | 46-60 | 0.436 | 0.284 |
| dataset_2 | not adjusted | 61+ | 0.467 | 0.284 |
| dataset_2 | same income mix | 18-30 | 0.254 | 0.117 |
| dataset_2 | same income mix | 31-45 | 0.326 | 0.117 |
| dataset_2 | same income mix | 46-60 | 0.371 | 0.117 |
| dataset_2 | same income mix | 61+ | 0.347 | 0.117 |
| dataset_2 | same credit score mix | 18-30 | 0.223 | 0.345 |
| dataset_2 | same credit score mix | 31-45 | 0.33 | 0.345 |
| dataset_2 | same credit score mix | 46-60 | 0.416 | 0.345 |
| dataset_2 | same credit score mix | 61+ | 0.568 | 0.345 |
| dataset_2 | same income and credit score mix | 18-30 | 0.269 | 0.12 |
| dataset_2 | same income and credit score mix | 31-45 | 0.334 | 0.12 |
| dataset_2 | same income and credit score mix | 46-60 | 0.359 | 0.12 |
| dataset_2 | same income and credit score mix | 61+ | 0.388 | 0.12 |

## 4. Calibration by age group

This section checks whether the approval probabilities are equally trustworthy for every age group. The calibration gap is the average predicted probability minus the share really approved; the 95 % interval comes from 1,000 bootstrap redraws.

**Key findings:**

- Both models are well calibrated in every age group: every 95 % interval includes 0.

*Per age group*

| dataset | age_group | applicants | mean_predicted | really_approved | calibration_gap | gap_95_lower | gap_95_upper | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 18-30 | 1888 | 0.638 | 0.633 | 0.005 | -0.012 | 0.021 | 0.1338 |
| dataset_1 | 31-45 | 3989 | 0.736 | 0.731 | 0.004 | -0.007 | 0.015 | 0.1294 |
| dataset_1 | 46-60 | 1533 | 0.551 | 0.543 | 0.008 | -0.01 | 0.025 | 0.1262 |
| dataset_1 | 61+ | 390 | 0.158 | 0.154 | 0.004 | -0.032 | 0.037 | 0.1305 |
| dataset_1 | all | 7800 | 0.647 | 0.642 | 0.005 | -0.002 | 0.013 | 0.1299 |
| dataset_2 | 18-30 | 675 | 0.162 | 0.141 | 0.021 | 0.0 | 0.04 | 0.0672 |
| dataset_2 | 31-45 | 1397 | 0.241 | 0.243 | -0.002 | -0.018 | 0.013 | 0.0946 |
| dataset_2 | 46-60 | 808 | 0.292 | 0.304 | -0.012 | -0.034 | 0.008 | 0.0962 |
| dataset_2 | 61+ | 120 | 0.339 | 0.308 | 0.031 | -0.027 | 0.089 | 0.1044 |
| dataset_2 | all | 3000 | 0.241 | 0.239 | 0.002 | -0.009 | 0.012 | 0.0893 |

## 5. Deferral to a human reviewer

This section describes when the system sends a case to a human instead of answering. Safety rules are always on: unusual input, missing data, and an age group the model cannot rank reliably. Evidence rules (borderline probability, disagreement between models A and B) are only switched on if the model is clearly less accurate on the cases they catch.

**Key findings:**

- Model A (dataset 1) sends 11.9 % of applicants to a human. Its accuracy is 84.9 % on the cases it decides and 83.2 % on the cases it defers.
- Model B (dataset 2) sends 10.9 % of applicants to a human. Its accuracy is 88.2 % on the cases it decides and 82.6 % on the cases it defers.
- Model A (dataset 1) defers 100.0 % of applicants aged 61+.
- Model B (dataset 2) defers 40.0 % of applicants aged 61+.

*Evidence rules, chosen on the validation part*

| dataset | rule | validation_cases_caught | validation_accuracy_overall | validation_accuracy_on_caught | switched_on |
| --- | --- | --- | --- | --- | --- |
| dataset_1 | borderline | 318 | 0.8501 | 0.8459 | False |
| dataset_1 | disagreement | 1649 | 0.8501 | 0.8551 | False |
| dataset_2 | borderline | 118 | 0.893 | 0.7966 | True |
| dataset_2 | disagreement | 94 | 0.893 | 0.8404 | True |

*Deferral rate per age group*

| dataset | age_group | applicants | deferral_rate |
| --- | --- | --- | --- |
| dataset_1 | 18-30 | 1888 | 0.0747 |
| dataset_1 | 31-45 | 3989 | 0.0637 |
| dataset_1 | 46-60 | 1533 | 0.0933 |
| dataset_1 | 61+ | 390 | 1.0 |
| dataset_2 | 18-30 | 675 | 0.0741 |
| dataset_2 | 31-45 | 1397 | 0.0859 |
| dataset_2 | 46-60 | 808 | 0.1361 |
| dataset_2 | 61+ | 120 | 0.4 |

*Accuracy on decided and deferred cases*

| dataset | applicants | deferred | deferral_rate | accuracy_all | accuracy_kept | accuracy_deferred | auc_all | auc_kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 7800 | 928 | 0.119 | 0.8471 | 0.8491 | 0.8319 | 0.8159 | 0.797 |
| dataset_2 | 3000 | 328 | 0.1093 | 0.8757 | 0.8817 | 0.8262 | 0.9272 | 0.923 |

## 6. Intervention: removing age from the model

This section tests whether removing age as a model input reduces the differences between age groups. Both models were retrained without age, with the same settings.

**Key findings:**

- Model A (dataset 1): AUC 0.8159 with age, 0.8162 without; demographic parity ratio 0.0 with age, 0.0 without; applicants aged 61+ getting an amount: 0.0 % with age, 0.0 % without.
- Model A (dataset 1): removing age did not change the outcome for applicants aged 61+, so the difference comes through other features linked to age, not from age itself.
- Model B (dataset 2): AUC 0.9272 with age, 0.925 without; demographic parity ratio 0.429 with age, 0.504 without; applicants aged 61+ getting an amount: 46.7 % with age, 45.8 % without.
- Model B (dataset 2): removing age did not change the outcome for applicants aged 61+, so the difference comes through other features linked to age, not from age itself.

*Accuracy and fairness, with and without age*

| dataset | version | auc | f1 | brier | demographic_parity_ratio_at_0.5 | equal_opportunity_difference_at_0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | with age | 0.8159 | 0.8865 | 0.1299 | 0.0 | 0.966 |
| dataset_1 | without age | 0.8162 | 0.8864 | 0.1298 | 0.0 | 0.966 |
| dataset_2 | with age | 0.9272 | 0.7412 | 0.0893 | 0.429 | 0.089 |
| dataset_2 | without age | 0.925 | 0.74 | 0.0898 | 0.504 | 0.087 |

*Per age group*

| dataset | version | age_group | model_approves_at_0.5 | within_group_auc | share_getting_an_amount_at_0.8 |
| --- | --- | --- | --- | --- | --- |
| dataset_1 | with age | 18-30 | 0.693 | 0.824 | 0.693 |
| dataset_1 | with age | 31-45 | 0.834 | 0.745 | 0.834 |
| dataset_1 | with age | 46-60 | 0.568 | 0.846 | 0.568 |
| dataset_1 | with age | 61+ | 0.0 | 0.522 | 0.0 |
| dataset_1 | without age | 18-30 | 0.693 | 0.82 | 0.693 |
| dataset_1 | without age | 31-45 | 0.834 | 0.748 | 0.834 |
| dataset_1 | without age | 46-60 | 0.568 | 0.846 | 0.568 |
| dataset_1 | without age | 61+ | 0.0 | 0.538 | 0.0 |
| dataset_2 | with age | 18-30 | 0.147 | 0.927 | 0.182 |
| dataset_2 | with age | 31-45 | 0.244 | 0.923 | 0.319 |
| dataset_2 | with age | 46-60 | 0.3 | 0.927 | 0.436 |
| dataset_2 | with age | 61+ | 0.342 | 0.931 | 0.467 |
| dataset_2 | without age | 18-30 | 0.176 | 0.931 | 0.213 |
| dataset_2 | without age | 31-45 | 0.243 | 0.921 | 0.293 |
| dataset_2 | without age | 46-60 | 0.292 | 0.924 | 0.394 |
| dataset_2 | without age | 61+ | 0.35 | 0.931 | 0.458 |

## 7. Limitations

This section lists what the fairness checks cannot show.

- Both datasets are synthetic, so the results describe these datasets, not real lending.
- Only age was checked. Gender was nearly balanced in dataset 1 and is not in dataset 2; combinations such as older applicants with low income were not checked separately.
- The 61+ group is small in dataset 2 (about 120 test applicants), so its numbers have wide intervals.
- In dataset 1 there are no applicants aged 61+ with a good credit score, so the data cannot show how the model would treat them.
- Deferral sends a case to a human; it does not guarantee a fair human decision, and older applicants are deferred more often.
