# Fairness report: loan assessment models

Written automatically on 2026-11-09 11:03 from the results of the Topic 3 checks (training steps 24 to 28). Aggregated numbers only.

## 1. What this report covers

This fairness report checks whether the loan assessment models treat applicants of different ages differently. Age groups are 18-30, 31-45, 46-60 and 61+. Model A was trained on dataset 1 (the course dataset), model B on dataset 2 (Financial Risk for Loan Approval). All numbers come from the untouched test part and are aggregated per age group; no individual applicant is shown. Age is used for these checks only and is a model input in the original models.

## 2. Approval rates by age group

This section compares, per age group, how often applicants were really approved and how often each model would approve them.

**Key findings:**

- Model A (dataset 1) approves 0.0 % of applicants aged 61+ at cut-off 0.5; 15.4 % of them were really approved.
- Model B (dataset 2) approves 35.0 % of applicants aged 61+ at cut-off 0.5; 30.8 % of them were really approved.
- Model A (dataset 1): demographic parity ratio 0.0 (below 0.8, a common warning sign), equal opportunity difference 0.966.
- Model B (dataset 2): demographic parity ratio 0.423 (below 0.8, a common warning sign), equal opportunity difference 0.116.

*Per age group*

| dataset | cut_off | age_group | applicants | actual_approval_rate | predicted_approval_rate | true_positive_rate | precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 0.5 | 18-30 | 1888 | 0.633 | 0.693 | 0.922 | 0.843 |
| dataset_1 | 0.5 | 31-45 | 3989 | 0.731 | 0.834 | 0.966 | 0.847 |
| dataset_1 | 0.5 | 46-60 | 1533 | 0.543 | 0.568 | 0.887 | 0.847 |
| dataset_1 | 0.5 | 61+ | 390 | 0.154 | 0.0 | 0.0 | 0.0 |
| dataset_1 | 0.8 | 18-30 | 1888 | 0.633 | 0.677 | 0.901 | 0.843 |
| dataset_1 | 0.8 | 31-45 | 3989 | 0.731 | 0.824 | 0.956 | 0.848 |
| dataset_1 | 0.8 | 46-60 | 1533 | 0.543 | 0.558 | 0.871 | 0.847 |
| dataset_1 | 0.8 | 61+ | 390 | 0.154 | 0.0 | 0.0 | 0.0 |
| dataset_2 | 0.5 | 18-30 | 675 | 0.141 | 0.148 | 0.695 | 0.66 |
| dataset_2 | 0.5 | 31-45 | 1397 | 0.243 | 0.243 | 0.721 | 0.721 |
| dataset_2 | 0.5 | 46-60 | 808 | 0.304 | 0.302 | 0.78 | 0.787 |
| dataset_2 | 0.5 | 61+ | 120 | 0.308 | 0.35 | 0.811 | 0.714 |
| dataset_2 | 0.8 | 18-30 | 675 | 0.141 | 0.053 | 0.326 | 0.861 |
| dataset_2 | 0.8 | 31-45 | 1397 | 0.243 | 0.082 | 0.306 | 0.912 |
| dataset_2 | 0.8 | 46-60 | 808 | 0.304 | 0.12 | 0.366 | 0.928 |
| dataset_2 | 0.8 | 61+ | 120 | 0.308 | 0.108 | 0.351 | 1.0 |

*Gaps between age groups*

| dataset | cut_off | demographic_parity_difference | demographic_parity_ratio | equal_opportunity_difference | predictive_parity_difference | most_approved_group | least_approved_group |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 0.5 | 0.834 | 0.0 | 0.966 | 0.847 | 31-45 | 61+ |
| dataset_1 | 0.8 | 0.824 | 0.0 | 0.956 | 0.848 | 31-45 | 61+ |
| dataset_2 | 0.5 | 0.202 | 0.423 | 0.116 | 0.127 | 61+ | 18-30 |
| dataset_2 | 0.8 | 0.067 | 0.444 | 0.06 | 0.139 | 46-60 | 18-30 |

## 3. Maximum loan amount by age group

This section checks which age groups receive an estimated maximum loan amount (approval probability at least 0.8), and whether income or credit score explain the differences. 'Adjusted' rates assume every age group had the same mix of incomes and credit scores.

**Key findings:**

- Model A (dataset 1): the gap between age groups in getting an amount is 0.834; with the same income mix it is 0.761, with the same income and credit score mix 0.715. The gap is NOT explained by income or credit score (at most 14.3 % of it).
- Model B (dataset 2): the gap between age groups in getting an amount is 0.257; with the same income mix it is 0.099, with the same income and credit score mix 0.129. The gap is mostly explained by income (about 61.5 % of it).

*Per age group*

| dataset | age_group | applicants | share_getting_an_amount | median_maximum | median_income | median_credit_score |
| --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 18-30 | 1888 | 0.693 | 31927.0 | 78508.0 | 740.0 |
| dataset_1 | 31-45 | 3989 | 0.834 | 31925.0 | 90272.0 | 757.0 |
| dataset_1 | 46-60 | 1533 | 0.568 | 32094.0 | 67850.0 | 716.0 |
| dataset_1 | 61+ | 390 | 0.0 | nan | 43856.0 | 438.0 |
| dataset_2 | 18-30 | 675 | 0.151 | 15210.0 | 42093.0 | 560.0 |
| dataset_2 | 31-45 | 1397 | 0.278 | 15182.0 | 48642.0 | 575.0 |
| dataset_2 | 46-60 | 808 | 0.379 | 16448.0 | 54330.0 | 596.0 |
| dataset_2 | 61+ | 120 | 0.408 | 16840.0 | 60624.0 | 616.0 |

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
| dataset_2 | not adjusted | 18-30 | 0.151 | 0.257 |
| dataset_2 | not adjusted | 31-45 | 0.278 | 0.257 |
| dataset_2 | not adjusted | 46-60 | 0.379 | 0.257 |
| dataset_2 | not adjusted | 61+ | 0.408 | 0.257 |
| dataset_2 | same income mix | 18-30 | 0.216 | 0.099 |
| dataset_2 | same income mix | 31-45 | 0.285 | 0.099 |
| dataset_2 | same income mix | 46-60 | 0.315 | 0.099 |
| dataset_2 | same income mix | 61+ | 0.299 | 0.099 |
| dataset_2 | same credit score mix | 18-30 | 0.182 | 0.338 |
| dataset_2 | same credit score mix | 31-45 | 0.287 | 0.338 |
| dataset_2 | same credit score mix | 46-60 | 0.363 | 0.338 |
| dataset_2 | same credit score mix | 61+ | 0.52 | 0.338 |
| dataset_2 | same income and credit score mix | 18-30 | 0.222 | 0.129 |
| dataset_2 | same income and credit score mix | 31-45 | 0.291 | 0.129 |
| dataset_2 | same income and credit score mix | 46-60 | 0.307 | 0.129 |
| dataset_2 | same income and credit score mix | 61+ | 0.351 | 0.129 |

## 4. Calibration by age group

This section checks whether the approval probabilities are equally trustworthy for every age group. The calibration gap is the average predicted probability minus the share really approved; the 95 % interval comes from 1,000 bootstrap redraws.

**Key findings:**

- Model B (dataset 2) is slightly too optimistic for age group 18-30 (gap 0.023, 95 % interval 0.002 to 0.043).
- All other age groups are well calibrated (their 95 % intervals include 0).

*Per age group*

| dataset | age_group | applicants | mean_predicted | really_approved | calibration_gap | gap_95_lower | gap_95_upper | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 18-30 | 1888 | 0.638 | 0.633 | 0.005 | -0.012 | 0.021 | 0.1333 |
| dataset_1 | 31-45 | 3989 | 0.736 | 0.731 | 0.005 | -0.006 | 0.016 | 0.1293 |
| dataset_1 | 46-60 | 1533 | 0.55 | 0.543 | 0.008 | -0.011 | 0.025 | 0.1267 |
| dataset_1 | 61+ | 390 | 0.155 | 0.154 | 0.001 | -0.035 | 0.035 | 0.1304 |
| dataset_1 | all | 7800 | 0.647 | 0.642 | 0.005 | -0.002 | 0.013 | 0.1298 |
| dataset_2 | 18-30 | 675 | 0.164 | 0.141 | 0.023 | 0.002 | 0.043 | 0.0677 |
| dataset_2 | 31-45 | 1397 | 0.242 | 0.243 | -0.002 | -0.018 | 0.014 | 0.0932 |
| dataset_2 | 46-60 | 808 | 0.292 | 0.304 | -0.012 | -0.034 | 0.008 | 0.097 |
| dataset_2 | 61+ | 120 | 0.341 | 0.308 | 0.032 | -0.024 | 0.088 | 0.1008 |
| dataset_2 | all | 3000 | 0.242 | 0.239 | 0.002 | -0.009 | 0.013 | 0.0888 |

## 5. Deferral to a human reviewer

This section describes when the system sends a case to a human instead of answering. Safety rules are always on: unusual input, missing data, and an age group the model cannot rank reliably. Evidence rules (borderline probability, disagreement between models A and B) are only switched on if the model is clearly less accurate on the cases they catch.

**Key findings:**

- Model A (dataset 1) sends 12.8 % of applicants to a human. Its accuracy is 85.0 % on the cases it decides and 82.8 % on the cases it defers.
- Model B (dataset 2) sends 11.0 % of applicants to a human. Its accuracy is 88.3 % on the cases it decides and 79.8 % on the cases it defers.
- Model A (dataset 1) defers 100.0 % of applicants aged 61+.
- Model B (dataset 2) defers 40.8 % of applicants aged 61+.

*Evidence rules, chosen on the validation part*

| dataset | rule | validation_cases_caught | validation_accuracy_overall | validation_accuracy_on_caught | switched_on |
| --- | --- | --- | --- | --- | --- |
| dataset_1 | borderline | 85 | 0.8501 | 0.7765 | True |
| dataset_1 | disagreement | 2257 | 0.8501 | 0.8551 | False |
| dataset_2 | borderline | 204 | 0.893 | 0.7794 | True |
| dataset_2 | disagreement | 92 | 0.893 | 0.8804 | False |

*Deferral rate per age group*

| dataset | age_group | applicants | deferral_rate |
| --- | --- | --- | --- |
| dataset_1 | 18-30 | 1888 | 0.089 |
| dataset_1 | 31-45 | 3989 | 0.0712 |
| dataset_1 | 46-60 | 1533 | 0.1005 |
| dataset_1 | 61+ | 390 | 1.0 |
| dataset_2 | 18-30 | 675 | 0.0696 |
| dataset_2 | 31-45 | 1397 | 0.0945 |
| dataset_2 | 46-60 | 808 | 0.1275 |
| dataset_2 | 61+ | 120 | 0.4083 |

*Accuracy on decided and deferred cases*

| dataset | applicants | deferred | deferral_rate | accuracy_all | accuracy_kept | accuracy_deferred | auc_all | auc_kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 7800 | 996 | 0.1277 | 0.8469 | 0.8496 | 0.8283 | 0.8175 | 0.7982 |
| dataset_2 | 3000 | 331 | 0.1103 | 0.874 | 0.8835 | 0.7976 | 0.9274 | 0.9232 |

## 6. Intervention: removing age from the model

This section tests whether removing age as a model input reduces the differences between age groups. Both models were retrained without age, with the same settings.

**Key findings:**

- Model A (dataset 1): AUC 0.8175 with age, 0.8143 without; demographic parity ratio 0.0 with age, 0.0 without; applicants aged 61+ getting an amount: 0.0 % with age, 0.0 % without.
- Model A (dataset 1): removing age did not change the outcome for applicants aged 61+, so the difference comes through other features linked to age, not from age itself.
- Model B (dataset 2): AUC 0.9274 with age, 0.9253 without; demographic parity ratio 0.423 with age, 0.511 without; applicants aged 61+ getting an amount: 40.8 % with age, 48.3 % without.

*Accuracy and fairness, with and without age*

| dataset | version | auc | f1 | brier | demographic_parity_ratio_at_0.5 | equal_opportunity_difference_at_0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | with age | 0.8175 | 0.8864 | 0.1298 | 0.0 | 0.966 |
| dataset_1 | without age | 0.8143 | 0.8864 | 0.1299 | 0.0 | 0.966 |
| dataset_2 | with age | 0.9274 | 0.7382 | 0.0888 | 0.423 | 0.116 |
| dataset_2 | without age | 0.9253 | 0.7375 | 0.0891 | 0.511 | 0.102 |

*Per age group*

| dataset | version | age_group | model_approves_at_0.5 | within_group_auc | share_getting_an_amount_at_0.8 |
| --- | --- | --- | --- | --- | --- |
| dataset_1 | with age | 18-30 | 0.693 | 0.823 | 0.693 |
| dataset_1 | with age | 31-45 | 0.834 | 0.746 | 0.834 |
| dataset_1 | with age | 46-60 | 0.568 | 0.85 | 0.568 |
| dataset_1 | with age | 61+ | 0.0 | 0.534 | 0.0 |
| dataset_1 | without age | 18-30 | 0.693 | 0.816 | 0.693 |
| dataset_1 | without age | 31-45 | 0.834 | 0.746 | 0.834 |
| dataset_1 | without age | 46-60 | 0.568 | 0.845 | 0.568 |
| dataset_1 | without age | 61+ | 0.0 | 0.517 | 0.0 |
| dataset_2 | with age | 18-30 | 0.148 | 0.928 | 0.151 |
| dataset_2 | with age | 31-45 | 0.243 | 0.924 | 0.278 |
| dataset_2 | with age | 46-60 | 0.302 | 0.925 | 0.379 |
| dataset_2 | with age | 61+ | 0.35 | 0.938 | 0.408 |
| dataset_2 | without age | 18-30 | 0.17 | 0.929 | 0.218 |
| dataset_2 | without age | 31-45 | 0.236 | 0.922 | 0.301 |
| dataset_2 | without age | 46-60 | 0.293 | 0.925 | 0.399 |
| dataset_2 | without age | 61+ | 0.333 | 0.932 | 0.483 |

## 7. Limitations

This section lists what the fairness checks cannot show.

- Both datasets are synthetic, so the results describe these datasets, not real lending.
- Only age was checked. Gender was nearly balanced in dataset 1 and is not in dataset 2; combinations such as older applicants with low income were not checked separately.
- The 61+ group is small in dataset 2 (about 120 test applicants), so its numbers have wide intervals.
- In dataset 1 there are no applicants aged 61+ with a good credit score, so the data cannot show how the model would treat them.
- Deferral sends a case to a human; it does not guarantee a fair human decision, and older applicants are deferred more often.
