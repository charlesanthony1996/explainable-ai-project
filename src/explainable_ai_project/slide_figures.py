"""Three figures sized for a projector, not for reading at a desk.

The four overview diagrams are reference material: dense, every number on
them, meant to be opened and studied. On a slide they are unreadable.
These carry the same content in plain words and large type.

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

    for drawer, name in [(three_steps, "folie_drei_schritte.png"),
                         (measurement, "folie_messung.png"),
                         (governance, "folie_governance.png"),
                         (equal_opportunity, "folie_equal_opportunity.png")]:
        figure = drawer()
        path = FIGURES_FOLDER / name
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAPER)
        plt.close(figure)
        print("  written:", path.relative_to(PROJECT_ROOT))
        if VAULT.is_dir():
            import shutil
            shutil.copy2(path, VAULT / name)
    return 0
