# Datasheet

Facts are filled in automatically on 2026-10-28 09:23. The team's text comes from `docs/datasheet_notes.md`.

## Sources

- dataset_1: https://raw.githubusercontent.com/charlesanthony1996/human-centered-and-explainable-ai-datasets/refs/heads/main/Loan%20Dataset.csv
- dataset_2: https://raw.githubusercontent.com/charlesanthony1996/human-centered-and-explainable-ai-datasets/refs/heads/main/Loan.csv

### Size

| dataset | rows | columns | missing_values_total | rows_with_missing_values | columns_with_missing_values | exact_duplicate_rows | duplicate_rows_ignoring_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | 52000 | 27 | 0 | 0 | 0 | 0 | 0 |
| dataset_2 | 20000 | 36 | 0 | 0 | 0 | 0 | no ID column |

### Model features (plus loan_to_income = loan amount / income)

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

### Age groups (used for fairness checks only, not a model input)

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

---

# Datasheet notes (written by the team)

This file is created once and never overwritten. Its text is copied into
docs/datasheet.md every time the reports step runs.

## How the approval labels were made
TODO

## Evidence that the data is synthetic
TODO (see docs/data_quality.md: for example, approval rises with the requested amount
in dataset 1, and nobody in dataset 1 requested more than 40 % of their income)

## Leakage and derived columns
TODO (RiskScore in dataset 2 almost perfectly decides approval; MonthlyLoanPayment is
calculated from the amount; neither is a model input)

## Limitations
TODO (currency unknown, different credit score ranges, no true maximum amount,
small 61+ group in dataset 2)

## Privacy and consent
TODO (synthetic records, no real people; no individual records go into the RAG documents)

## Licence
TODO (licence of dataset 2 on Kaggle)
