"""
Evaluation pathway, 3C step 13: the results folder.

    uv run main.py evaluate results

Reads every table the evaluation commands saved (results/tables) and the sheets
people filled in (evaluation/), then:

    1. counts the unjustified age-pair differences people marked   (step 9's output)
    2. adds 95 % confidence intervals to the A vs C and B vs C comparison (step 6)
    3. draws one figure per check                                   -> results/figures/evaluation_*.png
    4. writes one overview table with the headline number of every check
    5. writes results/README.md: every table and figure, which step made it,
       and which ones are still missing (with the command that makes them)

It never calls Qwen and takes a few seconds, so run it again after every
evaluation command or after more answers are scored. A check whose tables do
not exist yet is skipped with a note, never an error.

Note: results/ is in .gitignore. To share the folder, zip it or add files with
    git add -f results/tables/<file> results/figures/<file>
"""

import datetime

import numpy as np
import pandas as pd

from explainable_ai_project.evaluation import ANSWER_KEY_FILE
from explainable_ai_project.evaluation import EVALUATION_FOLDER
from explainable_ai_project.evaluation import PROJECT_ROOT
from explainable_ai_project.evaluation import RUN_NAMES
from explainable_ai_project.evaluation import SCORES_FILE
from explainable_ai_project.evaluation import SEED
from explainable_ai_project.evaluation import TABLES_FOLDER
from explainable_ai_project.evaluation import read_sheet
from explainable_ai_project.evaluation import save_table
from explainable_ai_project.evaluation import score_value


FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"
RESULTS_README = PROJECT_ROOT / "results" / "README.md"

BOOTSTRAP_COUNT = 2000

# Colors (checked for colour blindness). One color per run / prompt version,
# always the same, so a run looks the same in every figure.
RUN_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
VERSION_COLORS = {"v0": "#eb6834", "v1": "#2a78d6"}
MODEL_COLORS = {"Random Forest": "#2a78d6", "Qwen zero-shot": "#eb6834"}
INK = "#1f1f1e"
INK_SOFT = "#5f5e58"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"

notes = []          # what was skipped and why, printed at the end and written to the README


# ==================================================================
# Small helpers
# ==================================================================

def read_table(file_name):
    """A table from results/tables, or None if that command has not been run yet."""
    file_path = TABLES_FOLDER / file_name
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


def as_bool(series):
    """True/False columns come back from CSV as text; make them real booleans."""
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def figure_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.size": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })
    return plt


def save_figure(figure, file_name):
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = FIGURES_FOLDER / file_name
    figure.savefig(output_file, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure)
    print("Saved", output_file.relative_to(PROJECT_ROOT))
    return output_file


def label_bars(axis, bars, values, text_format="{:.2f}", tops=None):
    """Write each bar's value above it (the numbers must never depend on color alone).
    tops: where to put each label (above the interval line, if there is one)."""
    tops = tops or [None] * len(values)
    for bar, value, top in zip(bars, values, tops):
        if pd.isna(value):
            axis.text(bar.get_x() + bar.get_width() / 2, 0.02, "n/a", ha="center", va="bottom",
                      fontsize=8, color=INK_SOFT, transform=axis.get_xaxis_transform())
            continue
        height = bar.get_height() if top is None or pd.isna(top) else max(bar.get_height(), top)
        axis.annotate(text_format.format(value), (bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                      fontsize=8, color=INK)


def grouped_bars(axis, groups, series_names, values, colors, text_format="{:.2f}", errors=None):
    """Bars for several series side by side in every group.

    values[series][group] -> number; errors[series][group] -> (low, high) or None.
    """
    count = len(series_names)
    width = 0.8 / count
    positions = np.arange(len(groups))
    for number, name in enumerate(series_names):
        heights = [values[name].get(group, np.nan) for group in groups]
        offsets = positions - 0.4 + width * (number + 0.5)
        bars = axis.bar(offsets, np.nan_to_num(heights), width, color=colors[name], label=name,
                        edgecolor=SURFACE, linewidth=2)
        tops = []
        for offset, group, height in zip(offsets, groups, heights):
            interval = errors[name].get(group) if errors is not None else None
            if interval is None or pd.isna(height) or any(pd.isna(value) for value in interval):
                tops.append(None)
                continue
            axis.plot([offset, offset], interval, color=INK, linewidth=1)
            tops.append(interval[1])
        label_bars(axis, bars, heights, text_format, tops)
    axis.set_xticks(positions)
    axis.set_xticklabels(groups)


# ==================================================================
# 1. Step 9: count the unjustified age-pair differences people marked
# ==================================================================

def count_unjustified_differences():
    rows = []
    for version in ["v0", "v1"]:
        review_file = EVALUATION_FOLDER / ("age_pairs_review_" + version + ".csv")
        if not review_file.exists():
            continue
        sheet = read_sheet(review_file, "applicant_id")
        marks = sheet["unjustified_difference"].str.strip().str.lower()
        marked = marks.isin(["yes", "no"])
        rows.append({
            "prompt_version": version,
            "pairs": len(sheet),
            "pairs_marked": int(marked.sum()),
            "unjustified_differences": int((marks == "yes").sum()),
            "unjustified_share_of_marked": round(float((marks[marked] == "yes").mean()), 3) if marked.any() else None,
            "same_numbers_share": round(float(as_bool(sheet["same_numbers"]).mean()), 3),
        })
        if not marked.all():
            notes.append("Step 9 (" + version + "): " + str(int((~marked).sum())) + " of " + str(len(sheet))
                         + " age pairs not marked yet in evaluation/age_pairs_review_" + version + ".csv"
                         + " (column unjustified_difference: yes / no).")
    if not rows:
        notes.append("Step 9: no age-pair review sheet yet. Run: uv run main.py evaluate age-pairs")
        return None
    table = pd.DataFrame(rows)
    save_table(table, "age_pairs_unjustified.csv")
    return table


# ==================================================================
# 2. Step 6: confidence intervals for the run comparison
# ==================================================================

def scored_answers():
    """One row per answer with its run and the 0 - 2 scores as numbers, or None."""
    if not SCORES_FILE.exists() or not ANSWER_KEY_FILE.exists():
        notes.append("Step 6: no scoring sheet yet. Run: uv run main.py evaluate sheet")
        return None
    scores = read_sheet(SCORES_FILE, "answer_id")
    key = read_sheet(ANSWER_KEY_FILE, "answer_id")
    table = scores.merge(key[["answer_id", "run"]], on="answer_id", how="left")
    for column in ["correctness", "faithfulness", "citations"]:
        table[column + "_value"] = table[column].map(score_value)
    table["refusal_value"] = table["correct_refusal"].str.strip().str.lower().map({"yes": 1.0, "no": 0.0})
    table["numbers_ok_value"] = (table["numbers_ok"] == "yes").astype(float)

    scored = int(table["correctness_value"].notna().sum())
    if scored < len(table):
        notes.append("Step 6: " + str(scored) + " of " + str(len(table)) + " answers scored in "
                     "evaluation/llm_scores.csv. Means and intervals only use scored answers.")
    return table


def bootstrap_mean(values, generator):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    means = generator.choice(values, size=(BOOTSTRAP_COUNT, len(values)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def comparison_with_intervals(answers):
    """Paired by question: the same 45 questions were asked in every run, so the
    difference is taken per question and the questions are resampled."""
    generator = np.random.default_rng(SEED)
    rows = []
    for first, second, meaning in [("A", "C", "whole improvement"), ("B", "C", "what RAG adds"),
                                   ("A", "B", "what the models and SHAP add")]:
        for column in ["correctness", "faithfulness", "numbers_ok", "refusal"]:
            wide = answers.pivot_table(index="question_id", columns="run", values=column + "_value", aggfunc="first")
            if first not in wide.columns or second not in wide.columns:
                continue
            paired = wide[[first, second]].dropna()
            if len(paired) < 2:
                continue
            difference = (paired[second] - paired[first]).to_numpy()
            low, high = bootstrap_mean(difference, generator)
            rows.append({
                "comparison": first + " vs " + second,
                "meaning": meaning,
                "measure": column,
                "questions_paired": len(paired),
                "mean_first_run": round(float(paired[first].mean()), 3) if len(paired) else None,
                "mean_second_run": round(float(paired[second].mean()), 3) if len(paired) else None,
                "difference": round(float(difference.mean()), 3) if len(paired) else None,
                "ci95_low": round(low, 3) if not np.isnan(low) else None,
                "ci95_high": round(high, 3) if not np.isnan(high) else None,
                "interval_excludes_0": bool(low > 0 or high < 0) if not np.isnan(low) else None,
                "second_better_on": int((difference > 0).sum()),
                "first_better_on": int((difference < 0).sum()),
                "ties": int((difference == 0).sum()),
            })
    if not rows:
        return None
    table = pd.DataFrame(rows).rename(columns={"mean_first": "mean_first_run", "mean_second_run": "mean_second_run"})
    save_table(table, "llm_comparison_ci.csv")
    return table


# ==================================================================
# 3. Figures
# ==================================================================

def figure_llm_runs(answers):
    """Step 6: the three runs side by side, with 95 % intervals."""
    plt = figure_style()
    generator = np.random.default_rng(SEED)
    runs = [run for run in ["A", "B", "C"] if run in set(answers["run"])]
    if answers["correctness_value"].notna().sum() == 0:
        notes.append("Figure evaluation_llm_runs.png: needs scored answers (step 6).")
        return None

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})

    score_groups = {"Correctness": "correctness_value", "Faithfulness": "faithfulness_value",
                    "Citations": "citations_value"}
    share_groups = {"Numbers supported": "numbers_ok_value", "Correct refusal": "refusal_value"}

    for axis, groups, top, label in [(left, score_groups, 2.25, "mean score (0 - 2)"),
                                     (right, share_groups, 1.12, "share of answers")]:
        values = {}
        errors = {}
        for run in runs:
            name = run + ": " + RUN_NAMES[run]
            rows = answers[answers["run"] == run]
            values[name] = {group: rows[column].mean() for group, column in groups.items()}
            errors[name] = {group: bootstrap_mean(rows[column], generator) for group, column in groups.items()}
        names = [run + ": " + RUN_NAMES[run] for run in runs]
        colors = {run + ": " + RUN_NAMES[run]: RUN_COLORS[run] for run in runs}
        grouped_bars(axis, list(groups), names, values, colors, errors=errors)
        axis.set_ylim(0, top)
        axis.set_ylabel(label)

    left.set_title("Answer quality by run (higher is better)")
    right.set_title("Automatic and refusal checks")
    left.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
    figure.text(0.01, -0.12, "Lines: 95 % bootstrap interval. Citations are only scored for run C questions that "
                "need a document; runs A and B are 'na'.", fontsize=8, color=INK_SOFT)
    return save_figure(figure, "evaluation_llm_runs.png")


def figure_llm_categories(answers):
    plt = figure_style()
    rows = answers[answers["correctness_value"].notna()]
    if rows.empty:
        notes.append("Figure evaluation_llm_by_category.png: needs scored answers (step 6).")
        return None
    runs = [run for run in ["A", "B", "C"] if run in set(rows["run"])]
    categories = sorted(rows["category"].unique())
    means = rows.groupby(["run", "category"])["correctness_value"].mean()
    values = {run + ": " + RUN_NAMES[run]: {category: means.get((run, category), np.nan) for category in categories}
              for run in runs}
    colors = {run + ": " + RUN_NAMES[run]: RUN_COLORS[run] for run in runs}

    figure, axis = plt.subplots(figsize=(11, 4))
    grouped_bars(axis, categories, list(values), values, colors, text_format="{:.1f}")
    axis.set_ylim(0, 2.25)
    axis.set_ylabel("mean correctness (0 - 2)")
    axis.set_title("Correctness by question category")
    axis.tick_params(axis="x", labelrotation=15)
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=3)
    return save_figure(figure, "evaluation_llm_by_category.png")


def figure_zero_shot():
    """Step 7: does Qwen alone predict as well as the Random Forest?"""
    comparison = read_table("zero_shot_comparison.csv")
    detail = read_table("zero_shot_detail.csv")
    if comparison is None or detail is None:
        notes.append("Step 7: no zero-shot tables yet. Run: uv run main.py evaluate zero-shot")
        return None
    plt = figure_style()
    row = comparison.iloc[0]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    groups = ["Accuracy", "AUC"]
    values = {"Random Forest": {"Accuracy": row["forest_accuracy"], "AUC": row["forest_auc"]},
              "Qwen zero-shot": {"Accuracy": row["qwen_accuracy"], "AUC": row["qwen_auc"]}}
    grouped_bars(left, groups, list(values), values, MODEL_COLORS, text_format="{:.2f}")
    left.set_ylim(0, 1.12)
    left.set_title("Approval prediction on " + str(int(row["applicants"])) + " test applicants")
    left.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2)
    left.text(0, -0.24, str(int(row["qwen_answers_usable"])) + " of " + str(int(row["applicants"]))
              + " Qwen answers could be read. AUC n/a if Qwen gave no usable probabilities.",
              transform=left.transAxes, fontsize=8, color=INK_SOFT)

    both = detail.dropna(subset=["forest_maximum", "qwen_maximum"])
    right.grid(True, axis="both")
    if both.empty:
        right.text(0.5, 0.5, "no applicant with a maximum from both", ha="center", va="center",
                   transform=right.transAxes, color=INK_SOFT)
    else:
        right.scatter(both["forest_maximum"], both["qwen_maximum"], s=40, color=MODEL_COLORS["Qwen zero-shot"],
                      edgecolor=SURFACE, linewidth=1.5, zorder=3)
        limit = float(max(both["forest_maximum"].max(), both["qwen_maximum"].max())) * 1.05
        right.plot([0, limit], [0, limit], color=INK_SOFT, linewidth=1, linestyle="--", zorder=2)
        right.text(limit * 0.97, limit * 0.9, "same amount", ha="right", fontsize=8, color=INK_SOFT)
        right.set_xlim(0, limit)
        right.set_ylim(0, limit)
    right.set_xlabel("maximum from the Random Forest sweep (P ≥ 0.8)")
    right.set_ylabel("maximum Qwen guessed")
    right.set_title("Estimated maximum loan amount")
    return save_figure(figure, "evaluation_zero_shot.png")


def figure_shap_lime():
    """Step 8: is LIME stable, and does it agree with SHAP?"""
    detail = read_table("shap_vs_lime_detail.csv")
    if detail is None:
        notes.append("Step 8: no SHAP vs LIME table yet. Run: uv run main.py evaluate lime")
        return None
    plt = figure_style()
    measures = {"LIME stability\n(top-5 overlap between runs)": "lime_stability_jaccard",
                "SHAP-LIME agreement\n(top-5 overlap)": "shap_lime_agreement_jaccard",
                "Same top feature\n(share of LIME runs)": "same_top_feature_share",
                "LIME fidelity\n(local R²)": "lime_fidelity_r2_mean"}

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.grid(False)
    axis.grid(True, axis="x")
    for position, (label, column) in enumerate(measures.items()):
        values = detail[column].to_numpy(dtype=float)
        jitter = np.random.default_rng(SEED + position).uniform(-0.12, 0.12, len(values))
        axis.scatter(values, position + jitter, s=40, color=RUN_COLORS["A"], edgecolor=SURFACE,
                     linewidth=1.5, zorder=3)
        mean = float(np.nanmean(values))
        axis.plot([mean, mean], [position - 0.3, position + 0.3], color=INK, linewidth=2, zorder=4)
        axis.annotate("mean " + format(mean, ".2f"), (mean, position - 0.3), xytext=(0, 2),
                      textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    axis.set_ylim(len(measures) - 0.5, -0.65)       # first measure on top, room for the labels
    axis.set_yticks(range(len(measures)))
    axis.set_yticklabels(list(measures))
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("0 = no overlap / no fit      1 = identical / perfect fit")
    axis.set_title("SHAP vs LIME on " + str(len(detail)) + " test applicants (one dot per applicant)")
    return save_figure(figure, "evaluation_shap_vs_lime.png")


def figure_intervention(unjustified):
    """Steps 9 - 12: the three Topic 3 checks with prompt v0 and v1."""
    table = read_table("prompt_intervention.csv")
    if table is None:
        notes.append("Step 12: no prompt intervention table yet. Run: uv run main.py evaluate intervention")
        return None
    plt = figure_style()
    table = table.set_index("prompt_version")
    versions = [version for version in ["v0", "v1"] if version in table.index]
    pairs = int(read_table("age_pairs_v1.csv").shape[0]) if read_table("age_pairs_v1.csv") is not None else None
    probes = int(read_table("privacy_v1.csv").shape[0]) if read_table("privacy_v1.csv") is not None else None

    # Each panel has its own scale (counts vs shares): small multiples, never two axes
    panels = [
        ("Age pairs: numbers differ", "age_pairs_different_numbers", "count", pairs, "lower is better"),
        ("Age pairs: answer mentions age", "age_pairs_mentioning_age", "count", pairs, "lower is better"),
        ("Privacy: leak in Qwen's draft", "privacy_leaks_first_draft", "count", probes, "lower is better"),
        ("Privacy: leak in final answer", "privacy_leaks_answer", "count", probes, "lower is better"),
        ("Deferral: Qwen's draft complies", "deferral_compliance_qwen_draft", "share", 1, "higher is better"),
        ("Deferral: system answer complies", "deferral_compliance_system", "share", 1, "higher is better"),
    ]
    if unjustified is not None:
        marked = unjustified.set_index("prompt_version")
        if marked["pairs_marked"].sum() > 0:
            for version in versions:
                if version in marked.index:
                    table.loc[version, "unjustified"] = marked.loc[version, "unjustified_differences"]
            panels.insert(0, ("Age pairs: unjustified difference\n(marked by people)", "unjustified", "count",
                              pairs, "lower is better"))

    columns = 4 if len(panels) > 6 else 3
    lines = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(lines, columns, figsize=(3.4 * columns, 2.9 * lines))
    axes = np.atleast_1d(axes).ravel()
    for axis, (title, column, kind, out_of, direction) in zip(axes, panels):
        heights = [table.loc[version, column] if column in table.columns else np.nan for version in versions]
        bars = axis.bar(versions, np.nan_to_num(np.asarray(heights, dtype=float)), 0.6,
                        color=[VERSION_COLORS[version] for version in versions], edgecolor=SURFACE, linewidth=2)
        label_bars(axis, bars, heights, "{:.0%}" if kind == "share" else "{:.0f}")
        top = out_of if out_of else max([value for value in heights if not pd.isna(value)] + [1])
        axis.set_ylim(0, top * 1.18)
        axis.set_title(title, fontsize=9.5)
        axis.set_xlabel(direction + (" · out of " + str(out_of) if kind == "count" and out_of else ""),
                        fontsize=7.5, color=INK_SOFT)
        if kind == "share":
            axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: format(value, ".0%")))
    for axis in axes[len(panels):]:
        axis.set_visible(False)

    figure.suptitle("Prompt intervention: v0 (no fairness, privacy, uncertainty rules) vs v1",
                    x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    figure.tight_layout()
    return save_figure(figure, "evaluation_prompt_intervention.png")


# ==================================================================
# 4. Overview: one headline number per check
# ==================================================================

def overview(comparison, unjustified):
    rows = []

    def add(step, check, measure, value, table):
        rows.append({"step": step, "check": check, "measure": measure, "value": value, "table": table})

    summary = read_table("llm_summary.csv")
    if summary is not None:
        for _, row in summary.iterrows():
            add("6", "LLM vs baseline", "run " + row["run"] + " correctness (0-2)",
                row["correctness_mean_0_2"], "llm_summary.csv")
    if comparison is not None:
        for _, row in comparison[comparison["measure"] == "correctness"].iterrows():
            interval = ("[" + str(row["ci95_low"]) + ", " + str(row["ci95_high"]) + "]") \
                if pd.notna(row["ci95_low"]) else ""
            add("6", "LLM vs baseline", row["comparison"] + " correctness difference (95 % CI)",
                str(row["difference"]) + " " + interval, "llm_comparison_ci.csv")

    zero = read_table("zero_shot_comparison.csv")
    if zero is not None:
        row = zero.iloc[0]
        add("7", "Zero-shot Qwen vs Random Forest", "accuracy forest / Qwen",
            str(row["forest_accuracy"]) + " / " + str(row["qwen_accuracy"]), "zero_shot_comparison.csv")
    lime = read_table("shap_vs_lime.csv")
    if lime is not None:
        row = lime.iloc[0]
        add("8", "SHAP vs LIME", "LIME stability / SHAP-LIME agreement (Jaccard)",
            str(row["lime_stability_jaccard_mean"]) + " / " + str(row["shap_lime_agreement_jaccard_mean"]),
            "shap_vs_lime.csv")
    if unjustified is not None:
        for _, row in unjustified[unjustified["pairs_marked"] > 0].iterrows():
            add("9", "Counterfactual age pairs", "unjustified differences (" + row["prompt_version"] + ")",
                str(row["unjustified_differences"]) + " of " + str(row["pairs_marked"]) + " marked",
                "age_pairs_unjustified.csv")
    for version in ["v1", "v0"]:
        leaks = read_table("privacy_" + version + ".csv")
        if leaks is not None:
            add("10", "Privacy probes", "leaks in answers (" + version + ")",
                str(int(as_bool(leaks["leak_in_answer"]).sum())) + " of " + str(len(leaks)),
                "privacy_" + version + ".csv")
        deferral = read_table("deferral_" + version + ".csv")
        if deferral is not None:
            add("11", "Deferral compliance", "system answer compliance (" + version + ")",
                round(float(as_bool(deferral["system_answer_compliant"]).mean()), 3), "deferral_" + version + ".csv")
    if read_table("prompt_intervention.csv") is not None:
        add("12", "Prompt intervention", "before / after table", "see table", "prompt_intervention.csv")

    if not rows:
        return None
    table = pd.DataFrame(rows)
    table = table.iloc[table["step"].astype(int).argsort(kind="stable")]
    save_table(table, "evaluation_overview.csv")
    return table


# ==================================================================
# 5. results/README.md: what is in the folder
# ==================================================================

# Every output of the evaluation pathway: file -> (step, command that makes it)
EXPECTED = {
    "tables/evaluation_applicant_facts.csv": ("3A 1", "evaluate facts"),
    "tables/llm_summary.csv": ("3A 6", "evaluate summary"),
    "tables/llm_summary_by_category.csv": ("3A 6", "evaluate summary"),
    "tables/llm_comparison.csv": ("3A 6", "evaluate summary"),
    "tables/llm_comparison_ci.csv": ("3A 6", "evaluate results"),
    "figures/evaluation_llm_runs.png": ("3A 6", "evaluate results"),
    "figures/evaluation_llm_by_category.png": ("3A 6", "evaluate results"),
    "tables/zero_shot_detail.csv": ("3B 7", "evaluate zero-shot"),
    "tables/zero_shot_comparison.csv": ("3B 7", "evaluate zero-shot"),
    "figures/evaluation_zero_shot.png": ("3B 7", "evaluate results"),
    "tables/shap_vs_lime_detail.csv": ("3B 8", "evaluate lime"),
    "tables/shap_vs_lime.csv": ("3B 8", "evaluate lime"),
    "figures/evaluation_shap_vs_lime.png": ("3B 8", "evaluate results"),
    "tables/age_pairs_v1.csv": ("3B 9", "evaluate age-pairs"),
    "tables/age_pairs_unjustified.csv": ("3B 9", "evaluate results (after marking the review sheets)"),
    "tables/privacy_v1.csv": ("3B 10", "evaluate privacy"),
    "tables/deferral_v1.csv": ("3B 11", "evaluate deferral"),
    "tables/age_pairs_v0.csv": ("3B 12", "evaluate intervention"),
    "tables/privacy_v0.csv": ("3B 12", "evaluate intervention"),
    "tables/deferral_v0.csv": ("3B 12", "evaluate intervention"),
    "tables/prompt_intervention.csv": ("3B 12", "evaluate intervention"),
    "figures/evaluation_prompt_intervention.png": ("3B 12", "evaluate results"),
    "tables/evaluation_overview.csv": ("3C 13", "evaluate results"),
}


def write_readme(overview_table):
    results_folder = PROJECT_ROOT / "results"
    lines = ["# Results folder", "",
             "Made by `uv run main.py evaluate results` on " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
             + ". Run it again after every evaluation command.", ""]

    if overview_table is not None:
        lines += ["## Headline numbers", "", "| step | check | measure | value |", "|---|---|---|---|"]
        for _, row in overview_table.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in ["step", "check", "measure", "value"]) + " |")
        lines.append("")

    lines += ["## Evaluation pathway outputs", "", "| step | file | status | made by |", "|---|---|---|---|"]
    missing = 0
    for file_name, (step, command) in EXPECTED.items():
        present = (results_folder / file_name).exists()
        missing += not present
        lines.append("| " + step + " | `" + file_name + "` | " + ("ok" if present else "**missing**")
                     + " | `uv run main.py " + command + "` |")
    lines.append("")

    if notes:
        lines += ["## Still to do", ""] + ["- " + note for note in notes] + [""]

    # Everything else in the folder (Topic 1 and 2 tables and figures from the other commands)
    others = []
    for folder in ["tables", "figures"]:
        for file_path in sorted((results_folder / folder).glob("*")) if (results_folder / folder).exists() else []:
            name = folder + "/" + file_path.name
            if name not in EXPECTED:
                others.append(name)
    if others:
        lines += ["## Other tables and figures (Topics 1 and 2)", ""] + ["- `" + name + "`" for name in others] + [""]

    lines += ["Inputs filled in by people (in git): `evaluation/llm_scores.csv`, "
              "`evaluation/age_pairs_review_v0.csv`, `evaluation/age_pairs_review_v1.csv`.", ""]
    RESULTS_README.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_README.write_text("\n".join(lines), encoding="utf-8")
    print("Saved", RESULTS_README.relative_to(PROJECT_ROOT), "-", len(EXPECTED) - missing, "of", len(EXPECTED),
          "evaluation outputs present")


# ==================================================================
# Command
# ==================================================================

def make_results():
    notes.clear()
    unjustified = count_unjustified_differences()

    answers = scored_answers()
    comparison = None
    if answers is not None:
        comparison = comparison_with_intervals(answers)
        figure_llm_runs(answers)
        figure_llm_categories(answers)

    figure_zero_shot()
    figure_shap_lime()
    figure_intervention(unjustified)

    table = overview(comparison, unjustified)
    write_readme(table)

    if notes:
        print("\nStill to do:")
        for note in notes:
            print("  -", note)


if __name__ == "__main__":
    make_results()
