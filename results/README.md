# Results folder

Made by `uv run main.py evaluate results` on 2026-11-11 17:13. Run it again after every evaluation command.

## Evaluation pathway outputs

| step | file | status | made by |
|---|---|---|---|
| 3A 1 | `tables/evaluation_applicant_facts.csv` | **missing** | `uv run main.py evaluate facts` |
| 3A 6 | `tables/llm_summary.csv` | **missing** | `uv run main.py evaluate summary` |
| 3A 6 | `tables/llm_summary_by_category.csv` | **missing** | `uv run main.py evaluate summary` |
| 3A 6 | `tables/llm_comparison.csv` | **missing** | `uv run main.py evaluate summary` |
| 3A 6 | `tables/llm_comparison_ci.csv` | ok | `uv run main.py evaluate results` |
| 3A 6 | `figures/evaluation_llm_runs.png` | **missing** | `uv run main.py evaluate results` |
| 3A 6 | `figures/evaluation_llm_by_category.png` | **missing** | `uv run main.py evaluate results` |
| 3B 7 | `tables/zero_shot_detail.csv` | **missing** | `uv run main.py evaluate zero-shot` |
| 3B 7 | `tables/zero_shot_comparison.csv` | **missing** | `uv run main.py evaluate zero-shot` |
| 3B 7 | `figures/evaluation_zero_shot.png` | **missing** | `uv run main.py evaluate results` |
| 3B 8 | `tables/shap_vs_lime_detail.csv` | **missing** | `uv run main.py evaluate lime` |
| 3B 8 | `tables/shap_vs_lime.csv` | **missing** | `uv run main.py evaluate lime` |
| 3B 8 | `figures/evaluation_shap_vs_lime.png` | **missing** | `uv run main.py evaluate results` |
| 3B 9 | `tables/age_pairs_v1.csv` | **missing** | `uv run main.py evaluate age-pairs` |
| 3B 9 | `tables/age_pairs_unjustified.csv` | ok | `uv run main.py evaluate results (after marking the review sheets)` |
| 3B 10 | `tables/privacy_v1.csv` | **missing** | `uv run main.py evaluate privacy` |
| 3B 11 | `tables/deferral_v1.csv` | **missing** | `uv run main.py evaluate deferral` |
| 3B 12 | `tables/age_pairs_v0.csv` | **missing** | `uv run main.py evaluate intervention` |
| 3B 12 | `tables/privacy_v0.csv` | **missing** | `uv run main.py evaluate intervention` |
| 3B 12 | `tables/deferral_v0.csv` | **missing** | `uv run main.py evaluate intervention` |
| 3B 12 | `tables/prompt_intervention.csv` | **missing** | `uv run main.py evaluate intervention` |
| 3B 12 | `figures/evaluation_prompt_intervention.png` | **missing** | `uv run main.py evaluate results` |
| 3C 13 | `tables/evaluation_overview.csv` | **missing** | `uv run main.py evaluate results` |

## Still to do

- Step 9 (v0): 10 of 10 age pairs not marked yet in evaluation/age_pairs_review_v0.csv (column unjustified_difference: yes / no).
- Step 9 (v1): 10 of 10 age pairs not marked yet in evaluation/age_pairs_review_v1.csv (column unjustified_difference: yes / no).
- Step 6: 0 of 135 answers scored in evaluation/llm_scores.csv. Means and intervals only use scored answers.
- Figure evaluation_llm_runs.png: needs scored answers (step 6).
- Figure evaluation_llm_by_category.png: needs scored answers (step 6).
- Step 7: no zero-shot tables yet. Run: uv run main.py evaluate zero-shot
- Step 8: no SHAP vs LIME table yet. Run: uv run main.py evaluate lime
- Step 12: no prompt intervention table yet. Run: uv run main.py evaluate intervention

## Other tables and figures (Topics 1 and 2)

- `tables/control_zones.csv`
- `tables/control_zones_by_age_group.csv`
- `tables/fairness_definitions.json`
- `tables/parity_by_group_with_intervals.csv`
- `tables/parity_table_with_intervals.csv`
- `figures/control_zones.png`
- `figures/uebersicht_evaluation.png`
- `figures/uebersicht_gesamt.png`
- `figures/uebersicht_governance.png`

Inputs filled in by people (in git): `evaluation/llm_scores.csv`, `evaluation/age_pairs_review_v0.csv`, `evaluation/age_pairs_review_v1.csv`.
