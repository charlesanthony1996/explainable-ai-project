"""Four control zones instead of one deferral flag.

WHY THE FLAG IS NOT ENOUGH
--------------------------
The deferral rules answer one question - does a human look at this case -
and that single bit collapses situations that call for different handling.
A case the model is confident about and a case it cannot assess at all both
come out as "not flagged" or "flagged", and an automatic refusal that nobody
ever checks looks identical to an automatic approval.

Four zones, built from the rules that already exist, no retraining:

  auto-accept   confident approval, no rule fired. The system answers.
  sample audit  confident refusal, no rule fired. The system answers, and a
                fixed share is read by a person anyway. An unaudited stream
                of refusals is exactly where a bias of this kind survives:
                nobody complains about a loan they were never offered.
  human review  the case sits near the threshold, or the two models disagree
                beyond the limit. A person decides.
  safe-stop     the model has no basis for an answer - a field is missing, a
                value lies outside the range it was trained on, or the
                applicant belongs to a group where its within-group AUC is
                at chance. The system must not answer, not even tentatively.

The separation that matters is the last one. Human review says "a person
should check this"; safe-stop says "there is nothing here to check". Routing
both to the same queue hides which is which.

WHAT CANNOT BE MEASURED HERE
----------------------------
The course asks for disagreement, deferral, delay and override quality.
Disagreement and deferral are measured below. Delay and override quality
cannot be: there is no human in the loop to be slow or to overrule anything.
Reporting a number for them would be an invention, so the columns are absent
and this paragraph stands in their place.

Usage
-----
    uv run main.py zones
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explainable_ai_project.deferral import apply_rules
from explainable_ai_project.deferral import load_deferral_settings
from explainable_ai_project.load_split import load_split


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"
FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"

ZONES_FILE = TABLES_FOLDER / "control_zones.csv"
ZONES_BY_GROUP_FILE = TABLES_FOLDER / "control_zones_by_age_group.csv"
ZONES_FIGURE = FIGURES_FOLDER / "control_zones.png"

DATASET_NAMES = ["dataset_1", "dataset_2"]
AGE_GROUPS = ["18-30", "31-45", "46-60", "61+"]

# The share of automatic refusals read by a person. A declared number, not a
# derived one: it trades reviewer time against how quickly a drift in the
# refusal stream would be noticed. At 10 percent a systematic problem in a
# group of 390 shows up in about 39 cases.
AUDIT_SHARE = 0.10

ZONE_ORDER = ["auto-accept", "sample audit", "human review", "safe-stop"]
ZONE_COLOURS = {
    "auto-accept": "#1f7a4d",
    "sample audit": "#8a6d1a",
    "human review": "#c2601a",
    "safe-stop": "#b02828",
}

# Rules that mean "no basis for an answer" rather than "worth a second look".
NO_BASIS = ["missing_data", "unusual_input", "weak_group"]
SECOND_LOOK = ["borderline", "disagreement"]


def zone_of(flags, settings):
    """One zone per row, in priority order."""

    enabled = settings["enabled_rules"]
    no_basis = [rule for rule in NO_BASIS if rule in enabled]
    second_look = [rule for rule in SECOND_LOOK if rule in enabled]

    stop = (flags[no_basis].any(axis=1) if no_basis
            else pd.Series(False, index=flags.index))
    review = (flags[second_look].any(axis=1) if second_look
              else pd.Series(False, index=flags.index))
    approves = flags["p_approve"] >= settings["threshold"]

    zone = pd.Series("sample audit", index=flags.index)
    zone[approves] = "auto-accept"
    zone[review & ~stop] = "human review"
    zone[stop] = "safe-stop"
    return zone


def zones_for(dataset_name):
    settings = load_deferral_settings(dataset_name)
    test = load_split(dataset_name, "test")
    rows = test["rows"]

    flags = apply_rules(rows, dataset_name, settings)
    zone = zone_of(flags, settings)

    table = pd.DataFrame({
        "age_group": rows["age_group"].to_numpy(),
        "zone": zone.to_numpy(),
        "p_approve": flags["p_approve"].to_numpy(),
        "approved_in_data": test["labels"].to_numpy(),
    })
    return table, settings


def summarise(table, dataset_name, settings):
    total = len(table)
    rows = []
    for zone in ZONE_ORDER:
        part = table[table["zone"] == zone]
        share = len(part) / total
        # Only the audited slice of the refusal stream reaches a person.
        if zone == "sample audit":
            reaching_a_person = round(len(part) * AUDIT_SHARE)
        elif zone in ("human review", "safe-stop"):
            reaching_a_person = len(part)
        else:
            reaching_a_person = 0
        rows.append({
            "dataset": dataset_name,
            "zone": zone,
            "cases": len(part),
            "share": round(share, 4),
            "reaching_a_person": reaching_a_person,
            "mean_p_approve": round(float(part["p_approve"].mean()), 4)
                              if len(part) else None,
            "really_approved_share": round(float(part["approved_in_data"].mean()), 4)
                                     if len(part) else None,
        })
    summary = pd.DataFrame(rows)
    summary.attrs["threshold"] = settings["threshold"]
    return summary


def by_age_group(table, dataset_name):
    rows = []
    for group in AGE_GROUPS:
        part = table[table["age_group"] == group]
        if part.empty:
            continue
        row = {"dataset": dataset_name, "age_group": group,
               "applicants": len(part)}
        for zone in ZONE_ORDER:
            row[zone] = round(float((part["zone"] == zone).mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def draw(by_group):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

    for place, dataset_name in enumerate(DATASET_NAMES):
        part = by_group[by_group["dataset"] == dataset_name]
        if part.empty:
            continue
        bottom = np.zeros(len(part))
        positions = np.arange(len(part))
        for zone in ZONE_ORDER:
            values = part[zone].to_numpy()
            axes[place].bar(positions, values, bottom=bottom, width=0.62,
                            label=zone if place == 0 else None,
                            color=ZONE_COLOURS[zone], edgecolor="white",
                            linewidth=0.8)
            for index, value in enumerate(values):
                if value >= 0.08:
                    axes[place].text(positions[index],
                                     bottom[index] + value / 2,
                                     str(int(round(100 * value))) + "%",
                                     ha="center", va="center",
                                     fontsize=9, color="white",
                                     fontweight="bold")
            bottom += values

        axes[place].set_xticks(positions)
        axes[place].set_xticklabels(part["age_group"])
        axes[place].set_title(dataset_name + "  (model "
                              + ("A" if dataset_name == "dataset_1" else "B")
                              + ")", fontsize=11)
        axes[place].set_ylim(0, 1)
        axes[place].spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("share of applicants")
    figure.legend(loc="lower center", ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, -0.04))
    figure.suptitle("Control zones by age group, test split", fontsize=13,
                    fontweight="bold")
    figure.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure.savefig(ZONES_FIGURE, dpi=150, bbox_inches="tight",
                   facecolor="white")
    plt.close(figure)


def run_zones():
    print("Control zones - auto-accept, sample audit, human review, safe-stop")
    print("  audited share of automatic refusals:", AUDIT_SHARE)
    print()

    summaries = []
    group_tables = []

    for dataset_name in DATASET_NAMES:
        table, settings = zones_for(dataset_name)
        summary = summarise(table, dataset_name, settings)
        summaries.append(summary)
        group_tables.append(by_age_group(table, dataset_name))

        print(dataset_name, "- threshold", settings["threshold"],
              "- rules on:", ", ".join(settings["enabled_rules"]))
        print(summary[["zone", "cases", "share", "reaching_a_person",
                       "really_approved_share"]].to_string(index=False))
        print()

    all_summary = pd.concat(summaries, ignore_index=True)
    all_groups = pd.concat(group_tables, ignore_index=True)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    all_summary.to_csv(ZONES_FILE, index=False)
    all_groups.to_csv(ZONES_BY_GROUP_FILE, index=False)
    draw(all_groups)

    print("-" * 70)
    print("BY AGE GROUP (share of each group)")
    print("-" * 70)
    print(all_groups.to_string(index=False))

    print()
    print("  Not measured: delay and override quality. There is no human in "
          "the loop to be slow or to overrule anything.")
    print()
    for path in [ZONES_FILE, ZONES_BY_GROUP_FILE, ZONES_FIGURE]:
        print("  written:", path.relative_to(PROJECT_ROOT))
    return 0
