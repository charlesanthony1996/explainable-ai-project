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
