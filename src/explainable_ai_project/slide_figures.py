"""The figures for Michael's part of the deck, sized for a projector.

The four overview diagrams in `overview_figure.py` are reference material:
dense, every number on them, meant to be opened and studied. On a slide
they are unreadable. These carry the same content in plain words and large
type, and they follow the deck's own story:

    folie_drei_schritte        14  the whole project in three steps
    folie_plan                 15  the seven steps, control and evidence
    folie_fairness_metrics     16  step 3: the four measures, defined
    folie_equal_opportunity    17  step 4: the conflict, group by group
    folie_regeln_zonen         18  steps 5 and 6: rules and zones
    folie_contract             19  step 7a: the contract and the check
    folie_sprachmodell         20  step 7b: what the model does by itself
    folie_nicht_abgedeckt      21  what we did not cover

`folie_messung` and `folie_governance` are kept because they summarise the
same material on one page each, which is useful in the report and in the
vault even though the deck now walks through it step by step.

LINE SPACING IS COMPUTED, NOT GUESSED
The first version placed text at eyeballed offsets and the lines collided.
The canvas is 100 units tall over FIGURE_HEIGHT inches, so one unit is
FIGURE_HEIGHT/100 inches and a line of n points needs
n/72 * 1.3 / (FIGURE_HEIGHT/100) units. LINE() returns that, and every
block is laid out from it. Changing a font size changes the spacing with
it, instead of silently overlapping.

    uv run main.py slides
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent
FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"
VAULT = Path(r"C:\Users\micha\OneDrive\Dokumente\Obsidian Vault\IMC_S3\HC-EAI\Project")

# The deck's own colours, read out of its theme.
INK = "#14213D"
PAPER = "#F7F5F0"
MUTED = "#4D566A"
RUST = "#B5541A"
BLUE = "#2F6FB3"
GREEN = "#1F7A4D"

FIGURE_WIDTH = 13.3
FIGURE_HEIGHT = 6.4


def line_units(points):
    """How many canvas units one line of this font size occupies."""
    return (points / 72.0) * 1.32 / (FIGURE_HEIGHT / 100.0)


def canvas():
    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)
    axes.axis("off")
    figure.patch.set_facecolor(PAPER)
    return figure, axes


def write(axes, x, y, text, size, colour=INK, weight="normal", align="left",
          style="normal"):
    axes.text(x, y, text, ha=align, va="top", fontsize=size, color=colour,
              fontweight=weight, style=style, zorder=5)
    return y - line_units(size) * (text.count("\n") + 1)


def header_card(axes, x, y, width, height, colour, title, subtitle=None,
                fill="#FFFFFF"):
    """A card with a solid coloured header strip. Returns the first free y."""

    strip = 9.5 if subtitle else 7.0
    axes.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.6,rounding_size=1.5",
        linewidth=2.2, edgecolor=colour, facecolor=fill, zorder=2))
    axes.add_patch(FancyBboxPatch(
        (x, y + height - strip), width, strip,
        boxstyle="round,pad=0.6,rounding_size=1.5",
        linewidth=0, facecolor=colour, zorder=3))

    top = y + height - 1.6
    write(axes, x + width / 2, top, title, 13.5, "#FFFFFF", "bold", "center")
    if subtitle:
        write(axes, x + width / 2, top - line_units(13.5) - 0.6, subtitle,
              11, "#FFFFFF", "normal", "center")
    return y + height - strip - 2.6


def item_height(detail, title_size=11.5, detail_size=10.2, gap=3.0):
    """How tall one bullet is, counting the lines its detail actually has."""
    detail_lines = detail.count("\n") + 1
    return (line_units(title_size) + 0.6
            + detail_lines * line_units(detail_size) + gap)


def bullets(axes, x, y, items, colour, title_size=11.5, detail_size=10.2):
    """Bold line plus a detail of one or more lines.

    Each item advances by its own height. A fixed pitch was what made the
    first version overlap: three of the details are two lines long and the
    pitch allowed for one.
    """

    top = y
    for headline, detail in items:
        write(axes, x, top, headline, title_size, INK, "bold")
        write(axes, x, top - line_units(title_size) - 0.6, detail,
              detail_size, colour)
        top -= item_height(detail, title_size, detail_size)
    return top


def needed_height(items, subtitle=True, title_size=11.5, detail_size=10.2):
    """Card height that fits the header and every bullet, with padding."""
    strip = 9.5 if subtitle else 7.0
    body = sum(item_height(detail, title_size, detail_size)
               for _, detail in items)
    return strip + 2.6 + body + 1.5


CARD_TOP = 88.0
CARD_WIDTH = 30.6
CARD_X = [1.2, 34.7, 68.2]
# The gaps between the cards, measured from their edges plus the box pad,
# so the arrows sit in open paper instead of disappearing under a card.
GUTTER = [(32.6, 34.0), (66.1, 67.5)]


def three_columns(axes, columns):
    """Three cards of equal height, the height taken from the fullest one.

    Every card gets the same height so the row looks like a row, and that
    height is whatever the longest column needs. Nothing is placed by eye,
    so adding a bullet or a second detail line cannot push text out of its
    box: the box grows downwards instead.
    """

    height = max(needed_height(items, subtitle=bool(subtitle))
                 for _, _, subtitle, items, _ in columns)
    bottom = CARD_TOP - height

    for (colour, title, subtitle, items, fill), x in zip(columns, CARD_X):
        free = header_card(axes, x, bottom, CARD_WIDTH, height, colour,
                           title, subtitle, fill or "#FFFFFF")
        bullets(axes, x + 2.4, free, items, colour)

    middle = bottom + height / 2
    flow(axes, *GUTTER[0], middle)
    flow(axes, *GUTTER[1], middle, columns[2][0])
    return bottom


def table(axes, left, right, top, columns, rows, row_height=8.0,
          head_size=10.0, body_size=11.0, highlight=None):
    """A header strip and a stack of rows, laid out from column shares.

    `columns` is a list of (heading, share, align) where share is the
    position of the column across the table, 0 at the left edge and 1 at
    the right. `rows` is a list of value lists. `highlight` is the index of
    the row that gets the rust frame, or None.

    Returns the y of the lowest thing drawn.
    """

    head_lines = max(head.count("\n") + 1 for head, _, _ in columns)
    strip = line_units(head_size) * head_lines + 3.0
    axes.add_patch(FancyBboxPatch(
        (left, top - strip + 1.4), right - left, strip,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        linewidth=0, facecolor=INK, zorder=2))
    for head, share, align in columns:
        axes.text(left + (right - left) * share, top - 0.4, head,
                  ha=align, va="top", fontsize=head_size, color="#FFFFFF",
                  fontweight="bold", zorder=5)

    first = top - strip - 1.2
    for place, values in enumerate(rows):
        row_top = first - place * row_height
        if place == highlight:
            axes.add_patch(FancyBboxPatch(
                (left, row_top - row_height + 2.4), right - left,
                row_height - 0.9,
                boxstyle="round,pad=0.3,rounding_size=1.0",
                linewidth=1.8, edgecolor=RUST, facecolor="#FBEFE7",
                zorder=1))
        colour = RUST if place == highlight else INK
        for value, (_, share, align) in zip(values, columns):
            bold = value.isupper() and len(value) < 14
            axes.text(left + (right - left) * share, row_top, value,
                      ha=align, va="top", fontsize=body_size, color=colour,
                      fontweight="bold" if bold or place == highlight
                      else "normal", zorder=5)

    return first - len(rows) * row_height


def note_box(axes, left, right, top, height, colour, fill, strong, soft,
             strong_size=11.5, soft_size=10.3):
    """The closing box under a table: one bold line, then the detail."""

    axes.add_patch(FancyBboxPatch(
        (left, top - height), right - left, height,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=2.0, edgecolor=colour, facecolor=fill, zorder=1))
    write(axes, left + 2.0, top - 1.4, strong, strong_size, INK, "bold")
    write(axes, left + 2.0, top - 1.4 - line_units(strong_size) - 0.9, soft,
          soft_size, MUTED)
    return top - height


def flow(axes, x_from, x_to, y, colour=MUTED):
    axes.add_patch(FancyArrowPatch(
        (x_from, y), (x_to, y), arrowstyle="-|>", mutation_scale=24,
        linewidth=3.0, color=colour, zorder=1))


def footline(axes, bottom, strong, soft):
    """The closing two lines, hung under the cards rather than at a fixed y.

    The cards are as tall as their content, so their lower edge moves. A
    fixed footline left a hole on the short figures.
    """

    top = bottom - 5.0
    write(axes, 50, top, strong, 12.5, INK, "bold", "center")
    last = write(axes, 50, top - line_units(12.5) - 1.2, soft,
                 10.8, MUTED, "normal", "center", "italic")

    # Nothing is drawn below this, so pull the view up to it. Otherwise the
    # figure carries an empty band a third of its height, which on a slide
    # either wastes the space or shrinks the readable part.
    axes.set_ylim(last - 2.0, 100)


# ------------------------------------------------------------------
# 1 · The three steps
# ------------------------------------------------------------------

def three_steps():
    figure, axes = canvas()
    write(axes, 50, 97.0, "The whole project in three steps",
          15, INK, "bold", "center")

    bottom = three_columns(axes, [
        (BLUE, "1 \u00b7 TRAINING", "Learn from past decisions", [
            ("72,000 past applications",
             "two datasets, both synthetic"),
            ("Two Random Forests",
             "one per dataset, 400 and 200 trees"),
            ("Asking for more never helps",
             "built-in rule, tested 25,000 times"),
            ("Honest probabilities",
             "after calibration, 80% means 80%"),
        ], None),
        (RUST, "2 \u00b7 PREDICTION", "One applicant, one answer", [
            ("The largest amount",
             "that still reaches an 80% chance"),
            ("The reasons behind it",
             "SHAP: which detail helped, which hurt"),
            ("Machine or human?",
             "four rules decide, before any answer"),
            ("Qwen puts it in words",
             "it may quote our figures,\nnever compute its own"),
        ], None),
        (GREEN, "3 \u00b7 CHECK & GOVERN", "Is it fair? Is it honest?", [
            ("Fairness, with error bars",
             "is the gap real or just a small sample?"),
            ("Who may decide what",
             "four zones, from automatic to stop"),
            ("Every number re-checked",
             "against the figures we computed"),
            ("The decision, written down",
             "what we chose, who is accountable"),
        ], "#F3F8F5"),
    ])

    footline(axes, bottom,
             "Step 3 is what makes steps 1 and 2 defensible.",
             "Without it the system is only plausible. With it, every claim "
             "can be checked.")
    return figure


# ------------------------------------------------------------------
# 2 · What gets measured
# ------------------------------------------------------------------

def measurement():
    figure, axes = canvas()
    write(axes, 50, 97.0,
          "Two things can go wrong, so two things get measured",
          15, INK, "bold", "center")

    bottom = three_columns(axes, [
        (RUST, "THE MODEL", "Is it fair to everyone?", [
            ("Approval by age group",
             "over-61: model says yes to 0%,\nreal data says 15.4%"),
            ("Is that gap real?",
             "2,000 resamples: 0.959 to 0.973.\nNot a small-sample accident"),
            ("Machine or person?",
             "four zones. Every over-61 case\ngoes to a person"),
        ], None),
        (BLUE, "THE ANSWER", "Is it honest about what it knows?", [
            ("Does it name its source?",
             "full system 15 of 15.\nWithout sources: 0 of 15"),
            ("Does it invent sources?",
             "24 of 24 answers made up a\nreference when given none"),
            ("Same case, only age changed",
             "age brought up 5 times to 1.\nToo few cases to be sure"),
        ], None),
        (GREEN, "THE DECISION", "Written down, not implied", [
            ("Two fairness rules collide",
             "one holds, one fails.\nBoth cannot hold at once"),
            ("We chose equal opportunity",
             "and report the breach\ninstead of hiding it"),
            ("Why we did not patch it",
             "removing age changes nothing:\nthe credit score carries it"),
        ], "#F3F8F5"),
    ])

    footline(axes, bottom,
             "A fair model with a lying explanation is no better than an "
             "unfair one.",
             "That is why both halves are measured, and why the decision is "
             "written rather than assumed.")
    return figure


# ------------------------------------------------------------------
# 3 · The governance decision
# ------------------------------------------------------------------

def governance():
    figure, axes = canvas()
    write(axes, 50, 97.0,
          "The fairness rules conflict, so we had to choose",
          15, INK, "bold", "center")

    bottom = three_columns(axes, [
        (RUST, "THE CONFLICT", "Both cannot be true at once", [
            ("Are the approvals reliable?",
             "Yes. Equally reliable in every\nage group. Gap 0.005"),
            ("Does everyone get a chance?",
             "No. Of the over-61 who really\ngot a loan, the model finds 0%"),
            ("This is a proven fact",
             "when groups differ this much,\nno model can satisfy both"),
        ], None),
        (BLUE, "OUR CHOICE", "And why we made it", [
            ("We chose equal opportunity",
             "the harm that counts is to people\nwho would have repaid"),
            ("We report, we do not remove",
             "deleting the age field changes\nnothing \u2014 we tested it"),
            ("Instead: every case to a human",
             "and the model card says it is\nnot fit for the over-61"),
        ], None),
        (GREEN, "AND THE LIMITS", "Said out loud, not buried", [
            ("Who is accountable",
             "both of us, by name"),
            ("When is it reviewed",
             "never. The project ends at\nsubmission, and we say so"),
            ("What would shut it off",
             "four triggers. Any use of real\napplicant data voids all of this"),
        ], "#F3F8F5"),
    ])

    footline(axes, bottom,
             "One safety rule we rejected on evidence: deferring when the two "
             "models disagree.",
             "The model was no worse on those cases. It would have sent 2,257 "
             "of 7,800 to a person for nothing.")
    return figure


# ------------------------------------------------------------------
# The plan: the seven steps, what controls each, what proves it
# ------------------------------------------------------------------

# (control lines, step number and name, detail, evidence file, slide)
PLAN_STEPS = [
    (["Leakage scan: RiskScore removed",
      "Preprocessor fitted on train only"],
     "1 · Data", "52,000 + 20,000 rows, synthetic",
     "datasheet.md", None),
    (["A larger amount may only lower P",
      "Test set used exactly once"],
     "2 · Models A + B", "monotonic forest, isotonic calibration",
     "model_card.md", None),
    (["Tolerance declared before the numbers",
      "The interval decides, not the point"],
     "3 · Fairness measured", "per age group, with intervals",
     "parity_table_with_intervals.csv", "16"),
    (["Equal opportunity chosen",
      "The loss stated, not hidden"],
     "4 · Conflict decided", "predictive parity holds, the others fail",
     "governance_decision.md", "17"),
    (["An evidence rule needs a measured",
      "accuracy drop of at least 0.03"],
     "5 · Rules chosen", "3 safety + 2 evidence rules → 4 active",
     "deferral_rule_choice.csv", "18"),
    (["10% of the rejections are audited",
      "safe-stop: the system answers nothing"],
     "6 · Four zones", "auto-accept / audit / review / safe-stop",
     "control_zones.csv", "18"),
    (["The model never computes a figure",
      "The check runs in code, not in the prompt"],
     "7 · Answer checked", "every number, every citation",
     "evaluation_*.csv", "19 20"),
]

TOLERANCES = [
    ("Demographic parity ratio", "at least 0.80", "four-fifths rule (EEOC)"),
    ("Demographic parity difference", "at most 0.10", "set by the project"),
    ("Equal opportunity difference", "at most 0.10", "set by the project"),
    ("Predictive parity difference", "at most 0.10", "set by the project"),
]


def plan():
    figure, axes = canvas()
    write(axes, 50, 98.0, "The plan: seven steps, each one controlled "
                          "and each one evidenced",
          14.5, INK, "bold", "center")
    write(axes, 50, 93.0,
          "Accountable: Michael Aichinger and Charles Anthony jointly  |  "
          "No review planned: the project ends at submission",
          10.0, MUTED, "normal", "center")

    # The tolerance strip. It belongs at the top because it is what makes
    # the next three slides judgeable: the limits were written down before
    # the numbers were computed.
    axes.add_patch(FancyBboxPatch(
        (2.0, 77.0), 96.0, 10.0,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.8, edgecolor=MUTED, facecolor="#EFEDE7", zorder=1))
    write(axes, 50, 86.0,
          "DECLARED IN ADVANCE  —  a measure counts as breached only "
          "when the WHOLE interval lies outside the tolerance",
          10.3, INK, "bold", "center")
    for place, (name, limit, source) in enumerate(TOLERANCES):
        x = 5.0 + place * 23.5
        write(axes, x, 82.3, name, 9.6, INK, "bold")
        write(axes, x, 79.3, limit + "   " + source, 9.0, MUTED)

    write(axes, 20.0, 73.0, "CONTROL", 10.5, RUST, "bold", "right")
    write(axes, 50.0, 73.0, "THE PATH", 10.5, INK, "bold", "center")
    write(axes, 69.0, 73.0, "EVIDENCE", 10.5, GREEN, "bold")

    row = 9.3
    first = 68.0
    for place, step in enumerate(PLAN_STEPS):
        controls, name, detail, evidence, slide = step
        top = first - place * row
        axes.add_patch(FancyBboxPatch(
            (31.0, top - row + 1.6), 36.0, row - 1.0,
            boxstyle="round,pad=0.3,rounding_size=1.0",
            linewidth=1.8, edgecolor=BLUE, facecolor="#FFFFFF", zorder=2))
        write(axes, 49.0, top - 0.5, name, 11.0, BLUE, "bold", "center")
        write(axes, 49.0, top - 0.5 - line_units(11.0), detail,
              9.4, MUTED, "normal", "center")

        for line, text in enumerate(controls):
            write(axes, 29.5, top - 0.3 - line * line_units(9.0), text,
                  9.0, RUST, "normal", "right")

        write(axes, 68.5, top - 0.3, evidence, 9.0, GREEN)
        if slide:
            write(axes, 68.5, top - 0.3 - line_units(9.0),
                  "proved on slide " + slide.replace(" ", " and "),
                  9.0, INK, "bold")
        else:
            write(axes, 68.5, top - 0.3 - line_units(9.0),
                  "covered by the team", 9.0, MUTED, "normal", "left",
                  "italic")

    axes.set_ylim(first - len(PLAN_STEPS) * row - 1.0, 100)
    return figure


# ------------------------------------------------------------------
# Step 3 · the four fairness measures
# ------------------------------------------------------------------

def fairness_metrics():
    figure, axes = canvas()
    write(axes, 50, 97.0,
          "Step 3: four measures of fairness, three of them breached",
          15, INK, "bold", "center")
    write(axes, 50, 91.5,
          "Model A, dataset 1, cut-off 0.5. Reference group 31-45, the one "
          "the model treats best.",
          11.0, MUTED, "normal", "center")

    columns = [
        ("Measure", 0.00, "left"),
        ("The question it asks", 0.24, "left"),
        ("Value", 0.645, "center"),
        ("95% interval", 0.75, "center"),
        ("Allowed", 0.86, "center"),
        ("Verdict", 1.00, "right"),
    ]
    rows = [
        ["Demographic parity (diff.)",
         "Is every group approved equally often?",
         "0.834", "0.822 – 0.845", "≤ 0.10", "BREACHED"],
        ["Demographic parity (ratio)",
         "The same as a ratio — four-fifths rule",
         "0.000", "0.000 – 0.000", "≥ 0.80", "BREACHED"],
        ["Equal opportunity (diff.)",
         "Of those who would repay, who gets through?",
         "0.966", "0.959 – 0.973", "≤ 0.10", "BREACHED"],
        ["Predictive parity (diff.)",
         "When it says yes, is it right as often?",
         "0.005", "0.004 – 0.038", "≤ 0.10", "HOLDS"],
    ]
    bottom = table(axes, 2.0, 97.0, 85.0, columns, rows,
                   row_height=9.2, head_size=10.0, body_size=10.0,
                   highlight=2)

    write(axes, 2.0, bottom - 1.0,
          r"$\mathrm{EOD} = \max_g TPR_g - \min_g TPR_g$, "
          r"where $TPR_g$ is the share of the really-approved in group "
          r"$g$ that the model approves too",
          10.0, INK)
    write(axes, 2.0, bottom - 5.0,
          r"$\mathrm{DPR} = \min_g \hat{p}_g \,/\, \max_g \hat{p}_g$, "
          r"where $\hat{p}_g$ is the share of group $g$ the model approves",
          10.0, INK)

    note_box(axes, 2.0, 97.0, bottom - 11.0, 13.5, GREEN, "#F3F8F5",
             "One measure holds and three fail — so a choice is "
             "unavoidable, and it has to be written down.",
             "2,000 bootstrap resamples, seed 42. On dataset 2 the groups "
             "are smaller and 4 of the 8 measures stay undecided: their "
             "interval\nstraddles the tolerance, so the honest answer is "
             "\"not enough data\", not \"fair\".")
    axes.set_ylim(bottom - 26.5, 100)
    return figure


# ------------------------------------------------------------------
# Steps 5 and 6 · which rules are on, and where every case goes
# ------------------------------------------------------------------

def rules_and_zones():
    figure, axes = canvas()
    write(axes, 50, 98.0,
          "Steps 5 and 6: which rules send a case to a person, "
          "and where it lands",
          14.5, INK, "bold", "center")

    write(axes, 2.0, 92.0, "THE FIVE CANDIDATE RULES", 10.5, RUST, "bold")
    columns = [
        ("Rule", 0.00, "left"),
        ("Type", 0.19, "left"),
        ("What it catches", 0.30, "left"),
        ("Catches", 0.685, "center"),
        ("Accuracy there", 0.795, "center"),
        ("Overall", 0.90, "center"),
        ("", 1.00, "right"),
    ]
    rows = [
        ["missing data", "safety", "a required field is empty",
         "0.0%", "—", "—", "ON"],
        ["unusual input", "safety", "a value outside the training range",
         "7.9%", "—", "—", "ON"],
        ["weak group", "safety", "a group the model is poor on",
         "5.0%", "—", "—", "ON"],
        ["borderline", "evidence", "probability sits near the cut-off",
         "1.1%", "77.7%", "85.0%", "ON"],
        ["disagreement", "evidence", "models A and B do not agree",
         "29.2%", "85.5%", "85.0%", "OFF"],
    ]
    bottom = table(axes, 2.0, 97.0, 86.0, columns, rows,
                   row_height=6.6, head_size=9.4, body_size=10.0,
                   highlight=4)

    write(axes, 2.0, bottom - 0.6,
          "An evidence rule goes on only where the model is measurably "
          "worse — at least 0.03 below overall accuracy:",
          10.2, INK, "bold")
    write(axes, 2.0, bottom - 4.4,
          "borderline  85.0 − 77.7 = 7.3 points → ON          "
          "disagreement  85.0 − 85.5 = −0.5 points → OFF",
          10.0, MUTED)
    write(axes, 2.0, bottom - 7.8,
          "Switching disagreement on would have sent 2,257 of 7,738 cases "
          "to a person, where the model was no worse.",
          10.0, MUTED)

    write(axes, 2.0, bottom - 13.5, "WHERE EVERY CASE GOES  ·  "
          "7,800 TEST APPLICANTS", 10.5, BLUE, "bold")
    zone_columns = [
        ("Zone", 0.00, "left"),
        ("What it means", 0.155, "left"),
        ("Cases", 0.585, "center"),
        ("Share", 0.685, "center"),
        ("Sees a person", 0.795, "center"),
        ("Really approved", 1.00, "right"),
    ]
    zone_rows = [
        ["auto-accept", "answered without anyone looking",
         "5,156", "66.1%", "0", "84.7%"],
        ["sample audit", "a 10% sample is read afterwards",
         "1,648", "21.1%", "165", "14.1%"],
        ["human review", "a person decides together with the model",
         "68", "0.9%", "68", "79.4%"],
        ["safe-stop", "no basis at all — the system gives no answer",
         "928", "11.9%", "928", "37.9%"],
    ]
    lowest = table(axes, 2.0, 97.0, bottom - 19.0, zone_columns, zone_rows,
                   row_height=6.6, head_size=9.4, body_size=10.0,
                   highlight=3)

    write(axes, 2.0, lowest - 0.6,
          "1,161 of 7,800 cases reach a person — 14.9%. "
          "Of the over-61: 100% safe-stop, not one automatic answer.",
          10.6, INK, "bold")
    axes.set_ylim(lowest - 6.5, 100)
    return figure


# ------------------------------------------------------------------
# Step 7a · the contract and the check
# ------------------------------------------------------------------

def contract_figure():
    figure, axes = canvas()
    write(axes, 50, 97.0,
          "Step 7a: the numbers are fixed before the model is asked",
          15, INK, "bold", "center")

    bottom = three_columns(axes, [
        (BLUE, "THE CONTRACT", "Written before the answer", [
            ("Every figure it may say",
             "a list built from the models,\nnot from the model's memory"),
            ("What goes in",
             "probabilities, the maximum amount,\nthe SHAP shares, the "
             "applicant's data"),
            ("What never goes in",
             "anything the assistant would\nhave to work out itself"),
        ], None),
        (RUST, "THE CHECK", "Afterwards, in code", [
            ("Every number matched",
             "against the list, 1% relative or\n0.005 absolute tolerance"),
            ("Every citation matched",
             "against the passages that were\nactually retrieved"),
            ("Two tries, then a safe answer",
             "the fallback answer was never\nneeded in 72 runs"),
        ], None),
        (GREEN, "WHAT IT PROVED", "24 questions, 4 applicants", [
            ("Citations: 15 of 15",
             "run C only. A and B: 0 of 15.\nExact McNemar p = 0.000061"),
            ("Invented markers: 24 of 24",
             "in both runs without sources.\np = 0.00000012"),
            ("Grounded figures: 97.7%",
             "run C, 131 figures checked.\nRun A: 87.2% of 39"),
        ], "#F3F8F5"),
    ])

    footline(axes, bottom,
             "The contract is what makes the measurement possible at all.",
             "A number is then right or wrong against a file, not against "
             "an opinion — so no human judgement is needed to mark it.")
    return figure


# ------------------------------------------------------------------
# Step 7b · what the language model does on its own
# ------------------------------------------------------------------

def language_model():
    figure, axes = canvas()
    write(axes, 50, 97.0,
          "Step 7b: three things only the language model can be blamed for",
          15, INK, "bold", "center")

    bottom = three_columns(axes, [
        (RUST, "THE PROMPT", "The same rule, written twice", [
            ("Version 1 · the principle",
             "\"Cite the source it came from.\"\nResult: 0 of 15"),
            ("Version 2 · the place",
             "\"End every paragraph with the\nmarker, like [2].\" "
             "Result: 15 of 15"),
            ("And the cost",
             "the runs with no sources now\ninvent markers: 24 of 24"),
        ], None),
        (BLUE, "THE AGE TEST", "Same case, two ages", [
            ("Every model figure frozen",
             "sent as 30 and as 67; a difference\ncan only come from the "
             "words"),
            ("Age brought up",
             "5 times for the 67-year-old,\n1 time for the 30. p = 0.219"),
            ("Discouraging language",
             "no difference at all: 2 against 2.\n24 pairs is too few to "
             "be sure"),
        ], None),
        (GREEN, "THE HUMAN RATING", "What no code can judge", [
            ("Two readers, blind",
             "24 answers shuffled, the run label\nhidden, scored 0, 1 or 2"),
            ("Correctness, out of 2",
             "A 1.06   B 1.50   C 1.44\nthe contract makes it right"),
            ("Clarity, out of 2",
             "A 1.81   B 1.19   C 1.38\nand harder to read"),
        ], "#F3F8F5"),
    ])

    footline(axes, bottom,
             "Grounding buys correctness and costs readability. "
             "That is a design problem, not a bug.",
             "Cohen's κ = −0.04: one reader simply stricter. 75% "
             "agreed within one point, and both ranked the runs alike.")
    return figure


# ------------------------------------------------------------------
# What we did not cover
# ------------------------------------------------------------------

NOT_COVERED = [
    ("No operating owner after submission",
     "The project ends with the hand-in. Nobody is on call for it, and we "
     "say so rather than imply otherwise."),
    ("No review date",
     "A real system would be re-measured on a schedule. Ours will not be, "
     "because there will be no ours."),
    ("Delay and override quality not measurable",
     "No person ever worked a case, so we cannot say how long review takes "
     "or how often a human would overrule the model."),
    ("Learning-to-defer not implemented",
     "The rules decide who defers. A model that learns when to defer would "
     "be the next step, and we did not take it."),
    ("Both datasets are synthetic",
     "Every number here describes invented people. On real applicant data "
     "the whole record would have to be redone."),
    ("The 0.10 tolerance was set, not derived",
     "We declared it before measuring, which keeps it honest, but no "
     "regulation or study fixes it at that value."),
]


def not_covered():
    figure, axes = canvas()
    write(axes, 50, 97.0,
          "What this record does NOT cover — said out loud, "
          "not left quiet",
          15, RUST, "bold", "center")
    write(axes, 50, 91.0,
          "A governance record that only lists what went well is not a "
          "governance record.",
          11.5, MUTED, "normal", "center", "italic")

    axes.add_patch(FancyBboxPatch(
        (2.0, 29.0), 96.0, 55.0,
        boxstyle="round,pad=0.5,rounding_size=1.5",
        linewidth=2.4, edgecolor=RUST, facecolor="#FBEFE7", zorder=1))

    top = 80.0
    for place, (headline, detail) in enumerate(NOT_COVERED):
        column, row = place % 2, place // 2
        x = 5.0 + column * 47.0
        y = top - row * 19.0
        write(axes, x, y, str(place + 1) + "  " + headline,
              11.8, INK, "bold")
        write(axes, x + 2.5, y - line_units(11.8) - 0.8,
              wrap(detail, 46), 10.2, MUTED)

    write(axes, 50, 25.0,
          "Source: provenance.json, governance_decision.md",
          10.0, MUTED, "normal", "center")
    axes.set_ylim(20.0, 100)
    return figure


def wrap(text, width):
    """Break a sentence into lines of at most `width` characters."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


# ------------------------------------------------------------------
# 4 · Equal opportunity, group by group
# ------------------------------------------------------------------

# results/tables/parity_by_group_with_intervals.csv, dataset_1, cut-off 0.5.
# Hard-coded rather than read at draw time so the figure can be rebuilt
# without the models on disk; the test below re-reads the file and fails
# if the two ever drift apart.
EQUAL_OPPORTUNITY_ROWS = [
    ("18-30", "1,888", "63.3%", "69.3%", "92.2%", False),
    ("31-45", "3,989", "73.1%", "83.4%", "96.6%", False),
    ("46-60", "1,533", "54.3%", "56.8%", "88.7%", False),
    ("61+", "390", "15.4%", "0.0%", "0.0%", True),
]

EQUAL_OPPORTUNITY_HEADS = [
    ("Age group", 0.12),
    ("People\nin the test", 0.30),
    ("Really\napproved", 0.47),
    ("Model\nsays yes", 0.64),
    ("Of those who would repay,\nthe model approves", 0.845),
]


def equal_opportunity():
    figure, axes = canvas()
    write(axes, 50, 97.0,
          "Equal opportunity: who does the model let through?",
          15, INK, "bold", "center")
    write(axes, 50, 91.0,
          "Of the people who really did get a loan, what share does the "
          "model also approve?",
          11.5, MUTED, "normal", "center")

    left, right = 3.0, 97.0
    head_y = 83.0
    row_height = 9.4
    first_row = 72.5

    # Header strip
    axes.add_patch(FancyBboxPatch(
        (left, head_y - 6.6), right - left, 8.2,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        linewidth=0, facecolor=INK, zorder=2))
    for label, share in EQUAL_OPPORTUNITY_HEADS:
        write(axes, left + (right - left) * share, head_y, label,
              10.5, "#FFFFFF", "bold", "center")

    for place, row in enumerate(EQUAL_OPPORTUNITY_ROWS):
        group, people, actual, predicted, caught, worst = row
        top = first_row - place * row_height
        if worst:
            axes.add_patch(FancyBboxPatch(
                (left, top - row_height + 2.2), right - left, row_height - 0.8,
                boxstyle="round,pad=0.3,rounding_size=1.0",
                linewidth=1.8, edgecolor=RUST, facecolor="#FBEFE7",
                zorder=1))
        colour = RUST if worst else INK
        for value, (_, share) in zip(row[:5], EQUAL_OPPORTUNITY_HEADS):
            weight = "bold" if worst or share > 0.8 else "normal"
            write(axes, left + (right - left) * share, top, value,
                  13.0 if share > 0.8 else 11.5, colour, weight, "center")

    bar = first_row - 4 * row_height - 3.0
    axes.add_patch(FancyBboxPatch(
        (left, bar - 12.0), right - left, 13.0,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=2.0, edgecolor=GREEN, facecolor="#F3F8F5", zorder=1))
    write(axes, left + 2.0, bar - 0.8,
          "The gap: 96.6% for the best group, 0.0% for the over-61. "
          "Difference 0.966, where we allow 0.100.",
          12.0, INK, "bold")
    write(axes, left + 2.0, bar - 5.6,
          "Not a small-sample accident: resampling the test rows 2,000 times "
          "puts the gap between 0.959 and 0.973.\n"
          "Same model, same cut-off — the over-61 are simply never "
          "approved, although 15.4% of them were in the data.",
          10.8, MUTED)

    axes.set_ylim(bar - 14.0, 100)
    return figure


def check_equal_opportunity_rows():
    """Re-read the parity table and compare it with the rows above.

    The figure carries its numbers as text so it can be drawn without the
    models on disk. That is only safe if something notices when the table
    is recomputed and the figure is not.
    """

    table = (PROJECT_ROOT / "results" / "tables"
             / "parity_by_group_with_intervals.csv")
    if not table.is_file():
        print("  (parity table not on disk, figures not cross-checked)")
        return

    import csv

    on_disk = {}
    with open(table, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "dataset_1" and row["cut_off"] == "0.5":
                on_disk[row["age_group"]] = row

    wrong = []
    for group, people, actual, predicted, caught, _ in EQUAL_OPPORTUNITY_ROWS:
        row = on_disk.get(group)
        if row is None:
            wrong.append(f"{group}: not in the table any more")
            continue
        for shown, column in [(people, "applicants"),
                              (actual, "actual_approval_rate"),
                              (predicted, "predicted_approval_rate"),
                              (caught, "true_positive_rate")]:
            value = float(row[column])
            expected = (f"{value:,.0f}" if column == "applicants"
                        else f"{value * 100:.1f}%")
            if shown != expected:
                wrong.append(f"{group} {column}: figure says {shown}, "
                             f"table says {expected}")

    if wrong:
        raise RuntimeError(
            "folie_equal_opportunity.png is out of date:\n  "
            + "\n  ".join(wrong))
    print("  equal-opportunity numbers match the parity table")


# ------------------------------------------------------------------

def make_slide_figures():
    check_equal_opportunity_rows()
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

    for drawer, name in [
            (three_steps, "folie_drei_schritte.png"),
            (plan, "folie_plan.png"),
            (fairness_metrics, "folie_fairness_metrics.png"),
            (equal_opportunity, "folie_equal_opportunity.png"),
            (rules_and_zones, "folie_regeln_zonen.png"),
            (contract_figure, "folie_contract.png"),
            (language_model, "folie_sprachmodell.png"),
            (not_covered, "folie_nicht_abgedeckt.png"),
            (measurement, "folie_messung.png"),
            (governance, "folie_governance.png")]:
        figure = drawer()
        path = FIGURES_FOLDER / name
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAPER)
        plt.close(figure)
        print("  written:", path.relative_to(PROJECT_ROOT))
        if VAULT.is_dir():
            import shutil
            shutil.copy2(path, VAULT / name)
    return 0
