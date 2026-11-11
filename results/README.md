# Results folder

Made by `uv run main.py evaluate results` on 2026-11-11 22:36. Run it again after every evaluation command.

## Headline numbers

| step | check | measure | value |
|---|---|---|---|
| 6 | LLM vs baseline | run A correctness (0-2) | nan |
| 6 | LLM vs baseline | run B correctness (0-2) | nan |
| 6 | LLM vs baseline | run C correctness (0-2) | nan |
| 7 | Zero-shot Qwen vs Random Forest | accuracy forest / Qwen | 0.9 / 0.8 |
| 8 | SHAP vs LIME | LIME stability / SHAP-LIME agreement (Jaccard) | 0.703 / 0.678 |
| 10 | Privacy probes | leaks in answers (v1) | 0 of 10 |
| 10 | Privacy probes | leaks in answers (v0) | 0 of 10 |
| 11 | Deferral compliance | system answer compliance (v1) | 1.0 |
| 11 | Deferral compliance | system answer compliance (v0) | 1.0 |
| 12 | Prompt intervention | before / after table | see table |

## Evaluation pathway outputs

| step | file | status | made by |
|---|---|---|---|
| 3A 1 | `tables/evaluation_applicant_facts.csv` | ok | `uv run main.py evaluate facts` |
| 3A 6 | `tables/llm_summary.csv` | ok | `uv run main.py evaluate summary` |
| 3A 6 | `tables/llm_summary_by_category.csv` | ok | `uv run main.py evaluate summary` |
| 3A 6 | `tables/llm_comparison.csv` | ok | `uv run main.py evaluate summary` |
| 3A 6 | `tables/llm_comparison_ci.csv` | ok | `uv run main.py evaluate results` |
| 3A 6 | `figures/evaluation_llm_runs.png` | **missing** | `uv run main.py evaluate results` |
| 3A 6 | `figures/evaluation_llm_by_category.png` | **missing** | `uv run main.py evaluate results` |
| 3B 7 | `tables/zero_shot_detail.csv` | ok | `uv run main.py evaluate zero-shot` |
| 3B 7 | `tables/zero_shot_comparison.csv` | ok | `uv run main.py evaluate zero-shot` |
| 3B 7 | `figures/evaluation_zero_shot.png` | ok | `uv run main.py evaluate results` |
| 3B 8 | `tables/shap_vs_lime_detail.csv` | ok | `uv run main.py evaluate lime` |
| 3B 8 | `tables/shap_vs_lime.csv` | ok | `uv run main.py evaluate lime` |
| 3B 8 | `figures/evaluation_shap_vs_lime.png` | ok | `uv run main.py evaluate results` |
| 3B 9 | `tables/age_pairs_v1.csv` | ok | `uv run main.py evaluate age-pairs` |
| 3B 9 | `tables/age_pairs_unjustified.csv` | ok | `uv run main.py evaluate results (after marking the review sheets)` |
| 3B 10 | `tables/privacy_v1.csv` | ok | `uv run main.py evaluate privacy` |
| 3B 11 | `tables/deferral_v1.csv` | ok | `uv run main.py evaluate deferral` |
| 3B 12 | `tables/age_pairs_v0.csv` | ok | `uv run main.py evaluate intervention` |
| 3B 12 | `tables/privacy_v0.csv` | ok | `uv run main.py evaluate intervention` |
| 3B 12 | `tables/deferral_v0.csv` | ok | `uv run main.py evaluate intervention` |
| 3B 12 | `tables/prompt_intervention.csv` | ok | `uv run main.py evaluate intervention` |
| 3B 12 | `figures/evaluation_prompt_intervention.png` | ok | `uv run main.py evaluate results` |
| 3C 13 | `tables/evaluation_overview.csv` | ok | `uv run main.py evaluate results` |

## Still to do

- Step 9 (v0): 10 of 10 age pairs not marked yet in evaluation/age_pairs_review_v0.csv (column unjustified_difference: yes / no).
- Step 9 (v1): 10 of 10 age pairs not marked yet in evaluation/age_pairs_review_v1.csv (column unjustified_difference: yes / no).
- Step 6: 0 of 135 answers scored in evaluation/llm_scores.csv. Means and intervals only use scored answers.
- Figure evaluation_llm_runs.png: needs scored answers (step 6).
- Figure evaluation_llm_by_category.png: needs scored answers (step 6).

Inputs filled in by people (in git): `evaluation/llm_scores.csv`, `evaluation/age_pairs_review_v0.csv`, `evaluation/age_pairs_review_v1.csv`.
