# Data quality report

Written by `reports_governance.py` on 2026-10-28 09:23. Do not edit by hand; rerun instead.

## 1–2. Raw datasets, missing values and duplicates (steps 1 and 2)

### Summary

| dataset | rows | columns | missing_values_total | rows_with_missing_values | columns_with_missing_values | exact_duplicate_rows | duplicate_rows_ignoring_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 52000 | 27 | 0 | 0 | 0 | 0 | 0 |
| dataset_2 | 20000 | 36 | 0 | 0 | 0 | 0 | no ID column |

## 3. Leakage scan (step 3)

Single-feature AUC: how well each raw column alone predicts approval (0.5 = not at all, 1.0 = perfectly). Above 0.95 counts as leakage.

### dataset_1: columns that are not 'ok'

None.

### dataset_2: columns that are not 'ok'

| column | single_feature_auc | verdict |
| --- | --- | --- |
| RiskScore | 0.9873 | likely leakage |
| TotalDebtToIncomeRatio | 0.9064 | check before use |
| MonthlyIncome | 0.885 | check before use |
| AnnualIncome | 0.8837 | check before use |
| ApplicationDate | nan | skipped: date or free text |

## 4. Columns tied to the loan amount (step 4)

Columns that move with the loan amount. The amount sweep changes only the amount, so model inputs among these limit how far the sweep may go.

### dataset_1

| column | correlation_with_amount | r2_from_amount | link | model_input | action |
| --- | --- | --- | --- | --- | --- |
| Annual_Income | 0.849 | 0.797 | strongly tied | True | model input: keep the sweep within realistic amounts for the applicant |
| Credit_Score | 0.703 | 0.762 | strongly tied | True | model input: keep the sweep within realistic amounts for the applicant |

### dataset_2

| column | correlation_with_amount | r2_from_amount | link | model_input | action |
| --- | --- | --- | --- | --- | --- |
| MonthlyLoanPayment | 0.781 | 0.598 | strongly tied | False | not a model input: mention in the report |
| TotalDebtToIncomeRatio | 0.457 | 0.198 | partly tied | False | not a model input: mention in the report |
| BaseInterestRate | 0.386 | 0.138 | partly tied | False | not a model input: mention in the report |

## 5. Common features (step 5)

### Which raw column became which shared name

| shared_name | dataset_1_column | dataset_2_column | dataset_1_kind | dataset_2_kind | same_kind |
| --- | --- | --- | --- | --- | --- |
| age | Age | Age | number | number | True |
| marital_status | Marital_Status | MaritalStatus | text | text | True |
| dependents | Dependents | NumberOfDependents | number | number | True |
| education | Education | EducationLevel | text | text | True |
| employment_status | Employment_Status | EmploymentStatus | text | text | True |
| annual_income | Annual_Income | AnnualIncome | number | number | True |
| credit_score | Credit_Score | CreditScore | number | number | True |
| loan_amount | Loan_Amount_Requested | LoanAmount | number | number | True |
| loan_term_months | Loan_Term | LoanDuration | number | number | True |
| loan_purpose | Loan_Purpose | LoanPurpose | text | text | True |
| approved | Loan_Approval_Status | LoanApproved | number | number | True |

## 6. Labels and units (step 6)

### Category values not in the map (became 'Other')

None.

### Shared labels and their counts

| column | label | dataset_1 | dataset_2 |
| --- | --- | --- | --- |
| education | High School | 20769 | 5908 |
| education | Postgraduate | 10444 | 4004 |
| education | Undergraduate | 20787 | 10088 |
| employment_status | Employed | 25871 | 17036 |
| employment_status | Self-Employed | 15753 | 1573 |
| employment_status | Unemployed | 10376 | 1391 |
| loan_purpose | Education | 10344 | 3008 |
| loan_purpose | Home | 20919 | 5925 |
| loan_purpose | Personal or other | 10380 | 7033 |
| loan_purpose | Vehicle | 10357 | 4034 |
| marital_status | Divorced | 5256 | 2882 |
| marital_status | Married | 25957 | 10041 |
| marital_status | Other | 0 | 999 |
| marital_status | Single | 20787 | 6078 |

### Number columns side by side

| column | dataset_1_min | dataset_1_median | dataset_1_max | dataset_2_min | dataset_2_median | dataset_2_max |
| --- | --- | --- | --- | --- | --- | --- |
| age | 18 | 38.0 | 69 | 18 | 40.0 | 80 |
| dependents | 0 | 1.0 | 3 | 0 | 1.0 | 5 |
| annual_income | 20009 | 78839.0 | 149998 | 15000 | 48566.0 | 485341 |
| credit_score | 300 | 742.0 | 849 | 343 | 578.0 | 712 |
| loan_amount | 5000 | 21039.0 | 44848 | 3674 | 21914.5 | 184732 |
| loan_term_months | 12 | 126.0 | 239 | 12 | 48.0 | 120 |

## 7. Range checks and cleaning (step 7)

### Rows removed per reason

| dataset | rows_before | removed_missing | removed_duplicates | removed_out_of_range | rows_after |
| --- | --- | --- | --- | --- | --- |
| dataset_1 | 52000 | 0 | 0 | 0 | 52000 |
| dataset_2 | 20000 | 0 | 0 | 0 | 20000 |

### dataset_1: range rules

| column | allowed_min | allowed_max | actual_min | actual_max | rows_outside |
| --- | --- | --- | --- | --- | --- |
| age | 18 | 100 | 18 | 69 | 0 |
| dependents | 0 | 20 | 0 | 3 | 0 |
| annual_income | 1 | no limit | 20009 | 149998 | 0 |
| credit_score | 300 | 850 | 300 | 849 | 0 |
| loan_amount | 1 | no limit | 5000 | 44848 | 0 |
| loan_term_months | 1 | 480 | 12 | 239 | 0 |
| approved | 0 | 1 | 0 | 1 | 0 |

### dataset_2: range rules

| column | allowed_min | allowed_max | actual_min | actual_max | rows_outside |
| --- | --- | --- | --- | --- | --- |
| age | 18 | 100 | 18 | 80 | 0 |
| dependents | 0 | 20 | 0 | 5 | 0 |
| annual_income | 1 | no limit | 15000 | 485341 | 0 |
| credit_score | 300 | 850 | 343 | 712 | 0 |
| loan_amount | 1 | no limit | 3674 | 184732 | 0 |
| loan_term_months | 1 | 480 | 12 | 120 | 0 |
| approved | 0 | 1 | 0 | 1 | 0 |

## 8. Loan-to-income and age groups (step 8)

### Loan-to-income

| dataset | min | percentile_1 | median | percentile_99 | max |
| --- | --- | --- | --- | --- | --- |
| dataset_1 | 0.074 | 0.11 | 0.267 | 0.395 | 0.4 |
| dataset_2 | 0.027 | 0.07 | 0.451 | 2.623 | 7.544 |

### Rows and approval rate per age group

| dataset | age_group | rows | approval_rate |
| --- | --- | --- | --- |
| dataset_1 | 18-30 | 12583 | 0.633 |
| dataset_1 | 31-45 | 26589 | 0.731 |
| dataset_1 | 46-60 | 10227 | 0.543 |
| dataset_1 | 61+ | 2601 | 0.153 |
| dataset_2 | 18-30 | 4501 | 0.14 |
| dataset_2 | 31-45 | 9313 | 0.244 |
| dataset_2 | 46-60 | 5382 | 0.304 |
| dataset_2 | 61+ | 804 | 0.307 |

## 9. Split (step 9)

### Rows and approval rate per part

| dataset | split | rows | approval_rate |
| --- | --- | --- | --- |
| dataset_1 | train | 36400 | 0.642 |
| dataset_1 | validation | 7800 | 0.642 |
| dataset_1 | test | 7800 | 0.642 |
| dataset_2 | train | 14000 | 0.239 |
| dataset_2 | validation | 3000 | 0.239 |
| dataset_2 | test | 3000 | 0.239 |

### Age groups in the test part

| dataset | split | age_group | rows | small_test_group |
| --- | --- | --- | --- | --- |
| dataset_1 | test | 18-30 | 1888 | False |
| dataset_1 | test | 31-45 | 3989 | False |
| dataset_1 | test | 46-60 | 1533 | False |
| dataset_1 | test | 61+ | 390 | False |
| dataset_2 | test | 18-30 | 675 | False |
| dataset_2 | test | 31-45 | 1397 | False |
| dataset_2 | test | 46-60 | 808 | False |
| dataset_2 | test | 61+ | 120 | False |

## 10. Preprocessor (step 10)

Fitted on the training part only. Numbers pass through; categories are one-hot encoded.

### dataset_1

20 model features: age, dependents, annual_income, credit_score, loan_amount, loan_term_months, loan_to_income, marital_status_Divorced, marital_status_Married, marital_status_Single, education_High School, education_Postgraduate, education_Undergraduate, employment_status_Employed, employment_status_Self-Employed, employment_status_Unemployed, loan_purpose_Education, loan_purpose_Home, loan_purpose_Personal or other, loan_purpose_Vehicle

### dataset_2

21 model features: age, dependents, annual_income, credit_score, loan_amount, loan_term_months, loan_to_income, marital_status_Divorced, marital_status_Married, marital_status_Other, marital_status_Single, education_High School, education_Postgraduate, education_Undergraduate, employment_status_Employed, employment_status_Self-Employed, employment_status_Unemployed, loan_purpose_Education, loan_purpose_Home, loan_purpose_Personal or other, loan_purpose_Vehicle

## Governance log

Every data and model file with its md5 hash: `results/governance_log.csv`.
