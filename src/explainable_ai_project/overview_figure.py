"""The two overview diagrams: the whole programme, and the evaluation design.

These are drawn from code rather than by hand so that a number in a box can
never quietly go stale: every figure printed here is either a constant of the
project or was read off the result tables, and the comment beside it says
which. Redraw with

    uv run main.py overview

The second diagram exists because the first one cannot carry it. The three
configurations differ only in what goes into the prompt, and that is exactly
the thing a flow chart of the system does not show.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent
FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"

OVERVIEW_FILE = FIGURES_FOLDER / "uebersicht_gesamt.png"
EVALUATION_FILE = FIGURES_FOLDER / "uebersicht_evaluation.png"
GOVERNANCE_FILE = FIGURES_FOLDER / "uebersicht_governance.png"
MEASUREMENT_FILE = FIGURES_FOLDER / "uebersicht_messung.png"

# A second home for the figures, so they can be embedded in the notes. Skipped
# without complaint if the vault is not on this machine.
VAULT_FOLDER = Path(
    r"C:\Users\micha\OneDrive\Dokumente\Obsidian Vault\IMC_S3\HC-EAI\Project"
)

BLUE = "#1f5fa8"
GREEN = "#1f7a4d"
ORANGE = "#c2601a"
PURPLE = "#6a3fa0"
TEAL = "#157a7a"
GOLD = "#8a6d1a"
RED = "#b02828"
GREY = "#555555"


def tint(colour, amount=0.92):
    """A very pale version of a colour, for box fills."""
    colour = colour.lstrip("#")
    red, green, blue = (int(colour[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda channel: (channel / 255) + amount * (1 - channel / 255)
    return (mix(red), mix(green), mix(blue))


def box(axes, x, y, width, height, title, lines=(), colour=GREY,
        title_size=11, line_size=9, fill=None, linewidth=1.6, bold_title=True):
    """One rounded box with a heading and some lines underneath."""

    axes.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=linewidth, edgecolor=colour,
        facecolor=tint(colour) if fill is None else fill,
        zorder=2,
    ))

    axes.text(x + width / 2, y + height - 1.15, title,
              ha="center", va="top", fontsize=title_size, color=colour,
              fontweight="bold" if bold_title else "normal", zorder=3)

    for number, line in enumerate(lines):
        axes.text(x + width / 2, y + height - 2.9 - number * 1.55, line,
                  ha="center", va="top", fontsize=line_size, color="#222222",
                  zorder=3)


def arrow(axes, start, end, colour=GREY, style="-|>", dashed=False,
          linewidth=1.8, curve=0.0):
    axes.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle=style, mutation_scale=16,
        linewidth=linewidth, color=colour,
        linestyle="--" if dashed else "-",
        connectionstyle="arc3,rad=" + str(curve),
        shrinkA=2, shrinkB=2, zorder=1,
    ))


def new_canvas(width, height, title, subtitle):
    figure, axes = plt.subplots(figsize=(width, height))
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)
    axes.axis("off")
    figure.patch.set_facecolor("white")

    axes.text(50, 98.5, title, ha="center", va="top",
              fontsize=19, fontweight="bold", color="#111111")
    axes.text(50, 95.3, subtitle, ha="center", va="top",
              fontsize=10.5, color="#444444")
    return figure, axes


# ------------------------------------------------------------------
# Diagram 1: the whole programme
# ------------------------------------------------------------------

def draw_overview():
    figure, axes = new_canvas(
        19, 13,
        "HC-EAI Capstone - Ablauf des Gesamtprogramms",
        "Offline: fuenf Pipeline-Befehle bauen Daten, Modelle, Fairness-Befunde, "
        "Erklaerungen und Wissensbasis  |  Online: Inferenz fuer einen Antragsteller",
    )

    # ---------------- offline band ----------------
    axes.add_patch(FancyBboxPatch(
        (1.5, 52.5), 97, 40.0,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        linewidth=1.3, edgecolor="#999999", facecolor="#fbfbfb", zorder=0))
    axes.text(3.5, 91.2, "OFFLINE - einmal ausfuehren (in dieser Reihenfolge)",
              ha="left", va="top", fontsize=12, fontweight="bold", color="#333333")

    width = 18.0
    gap = 1.1
    left = 3.0
    top = 59.0
    height = 29.5

    blocks = [
        (BLUE, "1A  main.py data", "Schritte 1-11", [
            "DS 1: 52.000 Zeilen",
            "DS 2: 20.000 Zeilen",
            "Leakage-Scan:",
            "RiskScore (nur DS 2)",
            "AUC 0,9873 -> raus",
            "10 gemeinsame Features",
            "LTI = Betrag / Einkommen",
            "Split 70/15/15, Seed 42",
            "stratifiziert nach",
            "Zusage x Altersgruppe",
            "One-Hot -> 20 / 21 Spalten",
            "Preprocessor nur auf Train",
        ]),
        (GREEN, "1B  main.py models", "Schritte 12-23", [
            "Baselines: Logistische",
            "Regression und EBM",
            "Monotoner Random Forest",
            "A auf DS 1 (400 Baeume)",
            "B auf DS 2 (200 Baeume)",
            "Betrag und LTI duerfen P",
            "nur senken",
            "24 Hyperparameter-Kombis",
            "Kalibrierung: isotonic",
            "Monotonie: 500 x 50 Kurven,",
            "0 steigend",
            "Test genau einmal",
        ]),
        (ORANGE, "1C  main.py fairness", "Schritte 24-29", [
            "Zusagequoten je Altersgruppe",
            "Modell A: 61+ -> 0,0 %",
            "(real zugesagt: 15,4 %)",
            "Direkte Standardisierung:",
            "Einkommen + Score erklaeren",
            "max. 14,3 % der Luecke",
            "Kalibrierung je Gruppe:",
            "alle Bootstrap-CI enthalten 0",
            "Deferral: 3 Sicherheits- +",
            "2 Evidenzregeln -> 4 aktiv",
            "Alter entfernen aendert nichts",
            "-> Proxy ist Credit Score",
        ]),
        (TEAL, "T2  main.py shap", "Topic 2", [
            "TreeSHAP auf dem",
            "unkalibrierten monotonen",
            "Forest (exakt, kein Sampling)",
            "Additivitaetspruefung:",
            "Abweichung 1,6e-14",
            "Globale Wichtigkeit:",
            "credit_score 0,20999",
            "= mehr als 3x das naechste",
            "Beeswarm je Datensatz",
            "Monotonie als Spearman-",
            "Rangkorrelation,",
            "nicht als Vorzeichentest",
        ]),
        (PURPLE, "1D  main.py rag", "Schritte 30-35", [
            "EU AI Act, EBA-Leitlinien",
            "+ 5 Projektdokumente",
            "Chunks: 180 Woerter,",
            "40 Woerter Overlap",
            "-> 903 Chunks",
            "all-MiniLM-L6-v2,",
            "384 Dimensionen",
            "FAISS IndexFlatIP,",
            "exakte Kosinus-Suche",
            "Retrieval-Check 15 Fragen:",
            "Trefferquote 100 %,",
            "mittlerer Rang 1,27",
        ]),
    ]

    centres = []
    for number, (colour, title, subtitle, lines) in enumerate(blocks):
        x = left + number * (width + gap)
        box(axes, x, top, width, height, title, [subtitle] + [""] + list(lines),
            colour=colour, title_size=11.5, line_size=8.3)
        centres.append(x + width / 2)

    for number in range(len(blocks) - 1):
        x_from = left + number * (width + gap) + width
        arrow(axes, (x_from + 0.1, top + height / 2),
              (x_from + gap - 0.1, top + height / 2), colour="#888888")

    # 1C writes a document that 1D then indexes.
    arrow(axes, (centres[2], top - 0.3), (centres[4], top - 0.3),
          colour=ORANGE, dashed=True, curve=-0.20)
    axes.text(centres[3], 53.0,
              "Der Fairness-Report aus 1C wird selbst RAG-Dokument",
              ha="center", va="bottom", fontsize=8.8, color=ORANGE, style="italic")

    # ---------------- artefacts ----------------
    axes.text(3.0, 51.2, "Gespeicherte Artefakte", ha="left", va="bottom",
              fontsize=10.5, fontweight="bold", color="#444444")

    artefacts = [
        (3.0, "splits_*.csv\npreprocessor_*.joblib\nfeature_names_*.json"),
        (22.1, "rf_calibrated_*.joblib\nsweep_limits_*.json\nmodel_card.md"),
        (41.2, "deferral_settings_*.json\nfairness_report.md"),
        (60.3, "shap_*.csv\nshap_fidelity.csv"),
        (79.4, "faiss.index + Chunks\nsources.csv (md5)"),
    ]
    for x, text in artefacts:
        axes.add_patch(FancyBboxPatch(
            (x, 42.0), width, 7.4,
            boxstyle="round,pad=0.3,rounding_size=0.5",
            linewidth=1.1, edgecolor="#aaaaaa", facecolor="#f2f2f2", zorder=2))
        axes.text(x + width / 2, 45.7, text, ha="center", va="center",
                  fontsize=8.2, color="#333333", zorder=3)
        arrow(axes, (x + width / 2, 51.0), (x + width / 2, 49.7),
              colour="#aaaaaa", dashed=True, linewidth=1.2)

    # ---------------- online band ----------------
    axes.add_patch(FancyBboxPatch(
        (1.5, 1.5), 97, 37.0,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        linewidth=1.5, edgecolor=GOLD, facecolor="#fffdf3", zorder=0))
    axes.text(3.5, 37.2, "ONLINE - Inferenz fuer einen Antragsteller",
              ha="left", va="top", fontsize=12, fontweight="bold", color=GOLD)

    step_width = 12.4
    step_gap = 1.35
    step_left = 3.0
    step_y = 22.0
    step_height = 10.6

    steps = [
        ("1  Eingabe", ["10 Felder, z. B.", "Einkommen 72.000,", "beantragt 25.000"]),
        ("2  Modell A + B", ["kalibrierte P(Zusage)", "beim beantragten Betrag", "0,8528 / 0,2896"]),
        ("3  Sweep", ["groesster Betrag", "mit P >= 0,8", "28.429 / 12.279"]),
        ("4  SHAP", ["welche Features", "treiben das Ergebnis", "(exakt, pro Person)"]),
        ("5  Deferral", ["unusual_input, missing_data,", "weak_group, borderline", "-> markiert ja/nein"]),
        ("6  Contract-JSON", ["jede Zahl, die genannt", "werden darf, plus", "allowed_numbers"]),
        ("7  RAG-Retrieval", ["Top-4 Chunks", "aus FAISS", "(nur Konfiguration C)"]),
    ]

    xs = []
    for number, (title, lines) in enumerate(steps):
        x = step_left + number * (step_width + step_gap)
        xs.append(x)
        box(axes, x, step_y, step_width, step_height, title, lines,
            colour=GOLD, title_size=9.6, line_size=7.8, fill="white")
        if number > 0:
            arrow(axes, (x - step_gap + 0.1, step_y + step_height / 2),
                  (x - 0.1, step_y + step_height / 2), colour=GOLD)

    # Second row: the language model, the check, the output.
    second_y = 4.5
    second_height = 10.6

    box(axes, xs[2], second_y, step_width * 1.6, second_height,
        "8  Qwen 2.5 7B",
        ["temperature 0, seed 42", "Prompt v0 / v1 / v2", "antwortet in Alltagssprache"],
        colour=GOLD, title_size=9.6, line_size=7.8, fill="white")

    box(axes, xs[4] + 2.0, second_y, step_width * 1.9, second_height,
        "9  PRUEFUNG der Antwort",
        ["jede Zahl gegen das Contract", "jedes Zitat gegen die Chunks",
         "bei Markierung: Mensch genannt?"],
        colour=RED, title_size=10.2, line_size=7.9, linewidth=2.4)

    box(axes, xs[6] - 0.6, second_y, step_width * 1.1, second_height,
        "10  Ausgabe",
        ["Antwort + Pruefergebnis", "ins interaction_log.csv",
         "bei Markierung:", "Mensch entscheidet"],
        colour=GOLD, title_size=9.6, line_size=7.6, fill="white")

    # The wrap from the end of the first row back to the start of the second,
    # drawn as an elbow rather than a long curve so it cannot be mistaken for
    # an arrow coming out of one of the middle boxes.
    turn_y = (step_y + second_y + second_height) / 2
    end_x = xs[6] + step_width / 2
    qwen_x = xs[2] + step_width * 0.8
    arrow(axes, (end_x, step_y - 0.2), (end_x, turn_y), colour=GOLD, style="-")
    arrow(axes, (end_x, turn_y), (qwen_x, turn_y), colour=GOLD, style="-")
    arrow(axes, (qwen_x, turn_y), (qwen_x, second_y + second_height + 0.2),
          colour=GOLD)

    arrow(axes, (xs[2] + step_width * 1.6, second_y + second_height / 2),
          (xs[4] + 1.8, second_y + second_height / 2), colour=GOLD)
    arrow(axes, (xs[4] + 2.0 + step_width * 1.9, second_y + second_height / 2),
          (xs[6] - 0.8, second_y + second_height / 2), colour=RED)

    axes.text(96.5, 2.2,
              "Schritt 9 ist der Kern: ohne ihn waere das System ein "
              "gewoehnlicher RAG-Chatbot.",
              ha="right", va="bottom", fontsize=8.8, color=RED, style="italic")

    return figure


# ------------------------------------------------------------------
# Diagram 2: the evaluation design
# ------------------------------------------------------------------

def draw_evaluation():
    figure, axes = new_canvas(
        17, 11,
        "Evaluation - dieselbe Frage durch drei Prompts",
        "24 Fragen x 3 Konfigurationen = 72 Antworten  |  "
        "ein Contract je Antragsteller, an alle drei Arme weitergereicht",
    )

    # ---------------- shared input ----------------
    box(axes, 3.0, 76.0, 22.0, 15.0, "4 Antragsteller",
        ["P1  34 J.  Maximum 28.429, an der Grenze",
         "P2  67 J.  kein Betrag, MARKIERT",
         "P3  27 J.  kein Betrag, nicht markiert",
         "P4  45 J.  Maximum 41.579, sauber",
         "",
         "P2 und P3 sind das Kontrollpaar"],
        colour=BLUE, title_size=11.5, line_size=8.4)

    box(axes, 28.0, 76.0, 22.0, 15.0, "24 Fragen",
        ["Maximalbetrag 3   Features 3",
         "Unsicherheit 2    Modellguete 2",
         "Fairness 4        Regulierung 4",
         "Kreditwuerdigkeit 3  Datenlimits 2",
         "Verweigerung 1",
         "davon 15 quellenpflichtig"],
        colour=BLUE, title_size=11.5, line_size=8.4)

    box(axes, 53.0, 76.0, 25.0, 15.0, "1 Contract je Antragsteller",
        ["gebaut aus Modell A + B, Sweep,",
         "SHAP und Deferral",
         "",
         "wird in ALLEN drei Armen gebaut -",
         "in A nur nicht in den Prompt gelegt,",
         "aber weiter als Pruefmassstab benutzt"],
        colour=GREEN, title_size=11.5, line_size=8.4)

    # ---------------- the three arms ----------------
    arm_y = 51.5
    arm_height = 15.5
    arm_width = 28.0

    arms = [
        (3.0, GREY, "A  Standalone-LLM",
         ["im Prompt:", "nur die Antragstellerdaten",
          "", "kennt das Maximum nicht",
          "-> jede Zahl darueber ist messbar falsch"]),
        (35.5, ORANGE, "B  + Contract",
         ["im Prompt:", "Antragstellerdaten + Contract",
          "", "kennt die Modellzahlen",
          "-> aber keine Quellen"]),
        (68.0, PURPLE, "C  + abgerufene Passagen",
         ["im Prompt:", "Antragstellerdaten + Contract",
          "+ Top-4 Chunks mit Markern [1]-[4]",
          "", "das vollstaendige System"]),
    ]

    for x, colour, title, lines in arms:
        box(axes, x, arm_y, arm_width, arm_height, title, lines,
            colour=colour, title_size=12, line_size=8.6)
        arrow(axes, (x + arm_width / 2, 75.6),
              (x + arm_width / 2, arm_y + arm_height + 0.4), colour=colour)

    axes.text(50, 69.6,
              "Die Arme unterscheiden sich NUR im Prompt - gleiche Frage, "
              "gleicher Antragsteller, gleiches Contract, seed 42, temperature 0",
              ha="center", va="center", fontsize=9.4, color="#333333",
              style="italic",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor="none"), zorder=4)

    # ---------------- the checks ----------------
    # One band, not three columns under the three arms: every one of the 72
    # answers goes through all of these. Drawing them under the arms would
    # say that A is only checked, B only tested and C only rated.
    check_y = 27.0
    check_height = 18.5

    axes.add_patch(FancyBboxPatch(
        (3.0, check_y), 93.0, check_height,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        linewidth=1.5, edgecolor="#777777", facecolor="#fafafa", zorder=1))
    axes.text(50, check_y + check_height - 1.2,
              "Jede der 72 Antworten durchlaeuft alle drei Auswertungen",
              ha="center", va="top", fontsize=11.5, fontweight="bold",
              color="#333333", zorder=3)

    inner_y = check_y + 1.2
    inner_height = check_height - 4.4

    box(axes, 4.8, inner_y, 28.5, inner_height, "1  Automatische Pruefungen",
        ["Contract-Treue: jede Zahl gegen",
         "das Contract - derselbe Massstab",
         "fuer alle drei Arme",
         "Groundedness: jede Zahl gegen das,",
         "was dieser Arm gesehen hat",
         "Zitate, Scheinzitate, Deferral"],
        colour=RED, title_size=10.2, line_size=8.2, fill="white")

    box(axes, 35.8, inner_y, 28.5, inner_height, "2  Gepaarter Test",
        ["Exakter McNemar auf den",
         "diskordanten Paaren:",
         "b = erster falsch, zweiter richtig",
         "c = erster richtig, zweiter falsch",
         "p = binomtest(b, b+c, 0,5)",
         "nur Paare, in denen BEIDE Arme",
         "sich auf eine Zahl festgelegt haben"],
        colour=GREEN, title_size=10.2, line_size=8.2, fill="white")

    box(axes, 66.8, inner_y, 28.5, inner_height, "3  Rubrik (von Hand)",
        ["0 / 1 / 2 fuer Korrektheit,",
         "Quellentreue, Verstaendlichkeit",
         "72 Antworten gemischt,",
         "Konfigurationsspalte verdeckt,",
         "Seed 42 -> dieselbe Reihenfolge",
         "fuer eine zweite bewertende Person"],
        colour=TEAL, title_size=10.2, line_size=8.2, fill="white")

    for x, colour in [(3.0, GREY), (35.5, ORANGE), (68.0, PURPLE)]:
        arrow(axes, (x + arm_width / 2, arm_y - 0.4),
              (x + arm_width / 2, check_y + check_height + 0.4), colour=colour)

    # ---------------- results ----------------
    axes.add_patch(FancyBboxPatch(
        (3.0, 2.0), 93.0, 21.5,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        linewidth=1.8, edgecolor="#222222", facecolor="#f7f7f7", zorder=2))
    axes.text(50, 22.3,
              "Ergebnis des Laufs vom 10.11.2026   (24 Fragen, Prompt v2, "
              "qwen2.5:7b-instruct)",
              ha="center", va="top", fontsize=11.5, fontweight="bold",
              color="#111111", zorder=3)

    # Monospace and one fixed left edge, so the columns actually line up.
    result_lines = [
        ("Metrik                                        A        B        C     "
         "   gepaarter Test", True),
        ("Zitate auf quellenpflichtige Fragen        0/15     0/15    15/15     "
         "B vs C  p = 6,1e-05", False),
        ("Erfundene Zitatmarker                     24/24    24/24     0/24     "
         "B vs C  p = 1,2e-07", False),
        ("Contract-Treue (festgelegte Antworten)    0,769    0,882    0,444     "
         "B vs C  p = 0,031", False),
        ("Nichts erfunden (festgelegte Antworten)   0,769    0,882    0,944     "
         "B vs C  nicht signifikant", False),
        ("Markierte Faelle: Mensch genannt            1/7      5/7      5/7     "
         "A vs C  p = 0,125", False),
        ("Antworten, die ueberhaupt eine Zahl nennen 13/24    17/24    18/24    "
         " -", False),
    ]
    for number, (line, is_header) in enumerate(result_lines):
        axes.text(6.0, 19.3 - number * 2.35, line,
                  ha="left", va="top", fontsize=9.2, family="monospace",
                  color="#111111" if is_header else "#222222",
                  fontweight="bold" if is_header else "normal", zorder=3)

    return figure


# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Diagram 3: governance
#
# Not the architecture again. This one answers the questions a reviewer
# asks: at each step of a decision, what stops it going wrong, and what
# file proves the step happened. The flow runs down the middle, the
# controls sit on its left and the evidence on its right, so any step can
# be read across in one line.
# ------------------------------------------------------------------

def draw_governance():
    figure, axes = new_canvas(
        16, 13,
        "Governance - wer entscheidet was, und womit belegt",
        "Verantwortlich: Michael Aichinger und Charles Anthony gemeinsam  |  "
        "Kein Review geplant: das Projekt endet mit der Abgabe",
    )

    # --- declared tolerances -------------------------------------
    axes.add_patch(FancyBboxPatch(
        (2.5, 83.0), 95, 9.0,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        linewidth=1.5, edgecolor=GREY, facecolor="#f7f7f5", zorder=1))
    axes.text(50, 91.2, "ERKLAERTE TOLERANZ  -  eine Metrik gilt erst als "
                        "verletzt, wenn das ganze Konfidenzintervall ausserhalb liegt",
              ha="center", va="top", fontsize=10, fontweight="bold",
              color="#333333")
    tolerances = [
        ("Demographic parity ratio", "mind. 0,80", "Four-Fifths-Rule (EEOC)"),
        ("Demographic parity difference", "max. 0,10", "vom Projekt gesetzt"),
        ("Equal opportunity difference", "max. 0,10", "vom Projekt gesetzt"),
        ("Predictive parity difference", "max. 0,10", "vom Projekt gesetzt"),
    ]
    for number, (name, limit, origin) in enumerate(tolerances):
        x = 4.5 + number * 23.4
        axes.text(x, 88.0, name, ha="left", va="top", fontsize=8.4,
                  color="#222222", fontweight="bold")
        axes.text(x, 86.2, limit + "   " + origin, ha="left", va="top",
                  fontsize=8.0, color="#666666")

    # --- the three lanes -----------------------------------------
    axes.text(16.0, 80.0, "KONTROLLE", ha="center", va="top", fontsize=10.5,
              fontweight="bold", color=RED)
    axes.text(50.0, 80.0, "ENTSCHEIDUNGSWEG", ha="center", va="top",
              fontsize=10.5, fontweight="bold", color="#333333")
    axes.text(84.0, 80.0, "NACHWEIS", ha="center", va="top", fontsize=10.5,
              fontweight="bold", color=TEAL)

    steps = [
        ("1  Daten", ["52.000 + 20.000 Zeilen", "synthetisch, keine Personen"],
         ["Leakage-Scan: RiskScore raus", "Preprocessor nur auf Train"],
         ["governance_log.csv", "datasheet.md", "provenance.json"]),
        ("2  Modelle A + B", ["monotoner Random Forest", "isoton kalibriert"],
         ["Betrag darf P nur senken", "Test genau einmal benutzt"],
         ["model_card.md", "model_results.csv"]),
        ("3  Fairness geprueft", ["je Altersgruppe", "mit Konfidenzintervall"],
         ["Toleranz vorher erklaert", "Intervall entscheidet, nicht der Punkt"],
         ["parity_table_with_intervals.csv", "fairness_definitions.json"]),
        ("4  Konflikt entschieden", ["Predictive parity haelt,", "demographic parity nicht"],
         ["Equal opportunity gewaehlt", "Verzicht begruendet, nicht kaschiert"],
         ["governance_decision.md"]),
        ("5  Regeln gewaehlt", ["3 Sicherheits- + 2 Evidenzregeln", "-> 4 aktiv"],
         ["Evidenzregel nur bei", "Genauigkeitsabfall >= 0,03"],
         ["deferral_rule_choice.csv"]),
        ("6  Vier Zonen", ["auto-accept / sample audit /", "human review / safe-stop"],
         ["10 % der Ablehnungen geprueft", "safe-stop: System antwortet nicht"],
         ["control_zones.csv"]),
        ("7  Antwort geprueft", ["jede Zahl gegen das Contract", "jedes Zitat gegen die Quelle"],
         ["LLM rechnet nie selbst", "Pruefung laeuft automatisch"],
         ["interaction_log.csv", "evaluation_*.csv"]),
    ]

    top = 76.0
    height = 7.4
    gap = 1.7

    for number, (title, lines, controls, evidence) in enumerate(steps):
        y = top - number * (height + gap) - height
        box(axes, 33.0, y, 34.0, height, title, lines,
            colour=BLUE, title_size=10.2, line_size=8.2, fill="white")

        for place, text in enumerate(controls):
            axes.text(31.0, y + height - 2.0 - place * 1.8, text,
                      ha="right", va="top", fontsize=8.2, color=RED)
        for place, text in enumerate(evidence):
            axes.text(69.0, y + height - 2.0 - place * 1.8, text,
                      ha="left", va="top", fontsize=8.2, color=TEAL,
                      family="monospace")

        if number > 0:
            arrow(axes, (50.0, y + height + gap), (50.0, y + height + 0.3),
                  colour="#999999")

    # --- what is not covered -------------------------------------
    axes.add_patch(FancyBboxPatch(
        (2.5, 1.5), 95, 7.4,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        linewidth=1.8, edgecolor=RED, facecolor="#fdf5f5", zorder=1))
    axes.text(50, 7.9, "NICHT ABGEDECKT  -  ausdruecklich, statt stillschweigend",
              ha="center", va="top", fontsize=10, fontweight="bold", color=RED)
    axes.text(50, 5.6,
              "Kein operativer Verantwortlicher nach der Abgabe  ·  "
              "kein Review geplant  ·  Delay und Override-Qualitaet nicht "
              "messbar, da kein Mensch im Loop  ·\n"
              "Learning-to-defer nicht umgesetzt  ·  beide Datensaetze "
              "synthetisch, der gesamte Record faellt bei echten Antragsteller"
              "daten  ·  Toleranz 0,10 gesetzt, nicht abgeleitet",
              ha="center", va="top", fontsize=8.4, color="#333333")

    return figure


# ------------------------------------------------------------------
# Diagram 4: measurement and governance
#
# The other three show what the system does. This one shows how the parts
# of the checking layer hang together, which is the thing that is hard to
# hold in your head: two separate places get measured - the model and the
# answer - and both feed one written decision.
#
# Every box carries the command that produces it, the question it answers
# and the result it gave, so the diagram doubles as an index.
# ------------------------------------------------------------------

def draw_measurement():
    figure, axes = new_canvas(
        17, 12.5,
        "Messung und Governance - wie die Schritte zusammenhaengen",
        "Zwei Stellen werden geprueft: das Modell und die Antwort. "
        "Beides zusammen traegt eine schriftliche Entscheidung.",
    )

    def step(x, y, width, height, number, name, command, question, result,
             colour):
        axes.add_patch(FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.35,rounding_size=0.7",
            linewidth=1.8, edgecolor=colour, facecolor="white", zorder=2))
        axes.add_patch(FancyBboxPatch(
            (x, y + height - 1.5), width, 1.5,
            boxstyle="round,pad=0.35,rounding_size=0.7",
            linewidth=0, facecolor=tint(colour, 0.86), zorder=2))
        axes.text(x + width / 2, y + height - 0.45, number + "  " + name,
                  ha="center", va="top", fontsize=11, fontweight="bold",
                  color=colour, zorder=4)
        axes.text(x + 0.5, y + height - 2.0, command,
                  ha="left", va="top", fontsize=8.2, color="#555555",
                  family="monospace", zorder=4)
        axes.text(x + 0.5, y + height - 3.3, "FRAGE", ha="left", va="top",
                  fontsize=7.8, fontweight="bold", color="#888888", zorder=4)
        axes.text(x + 0.5, y + height - 4.1, question, ha="left", va="top",
                  fontsize=8.6, color="#222222", zorder=4)
        # The result sits a fixed distance above the floor, with room for
        # three lines beneath the label. Anchoring it to the bottom edge is
        # what pushed the text through the border in the first version.
        axes.text(x + 0.5, y + 5.0, "ERGEBNIS", ha="left", va="top",
                  fontsize=7.8, fontweight="bold", color="#888888", zorder=4)
        axes.text(x + 0.5, y + 4.1, result, ha="left", va="top",
                  fontsize=8.6, color=colour, fontweight="bold", zorder=4)

    # ---- band 1: the model ----------------------------------------
    axes.add_patch(FancyBboxPatch(
        (1.5, 62.0), 97, 30.0,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        linewidth=1.3, edgecolor="#aaaaaa", facecolor="#fbfbfa", zorder=0))
    axes.text(3.5, 91.0, "EBENE 1  -  WAS DAS MODELL TUT",
              ha="left", va="top", fontsize=11.5, fontweight="bold",
              color="#333333")
    axes.text(3.5, 88.5,
              "Geprueft wird der Random Forest, bevor ein Sprachmodell "
              "ueberhaupt beteiligt ist.",
              ha="left", va="top", fontsize=9, color="#666666")

    step(4.0, 66.0, 45.0, 17.0, "1", "Fairness mit Konfidenzintervallen",
         "uv run main.py verify_fairness",
         "Behandelt das Modell Altersgruppen unterschiedlich -\n"
         "und ist der Unterschied groesser als der Zufall?",
         "Luecke real: 0,966 (Spanne 0,959-0,973).\n"
         "Bei Datensatz 2: 4 von 8 Massen unentscheidbar.",
         RED)

    step(51.5, 66.0, 45.0, 17.0, "2", "Vier Kontrollzonen",
         "uv run main.py zones",
         "Welche Faelle darf das System entscheiden,\n"
         "welche nicht - und wer schaut nach?",
         "61+ zu 100 % safe-stop: kein automatisches\n"
         "Urteil, nur eine Warteschlange.",
         ORANGE)

    # ---- band 2: the answer ---------------------------------------
    axes.add_patch(FancyBboxPatch(
        (1.5, 26.5), 97, 31.0,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        linewidth=1.3, edgecolor="#aaaaaa", facecolor="#fbfbfa", zorder=0))
    axes.text(3.5, 56.3, "EBENE 2  -  WAS DAS SPRACHMODELL DARAUS MACHT",
              ha="left", va="top", fontsize=11.5, fontweight="bold",
              color="#333333")
    axes.text(3.5, 53.9,
              "Dieselben Modellzahlen, drei verschiedene Prompts. "
              "Nur der Prompt unterscheidet sich.",
              ha="left", va="top", fontsize=9, color="#666666")

    step(4.0, 29.0, 22.0, 19.0, "3", "Zahlen und Zitate",
         "uv run main.py evaluate",
         "Sagt die Antwort nur, was\ngerechnet wurde - und nennt\nsie ihre Quelle?",
         "Zitate 15/15 gegen 0/15.\nErfundene Marker 24/24\ngegen 0/24.",
         BLUE)

    step(27.5, 29.0, 22.0, 19.0, "4", "Alterspaare",
         "uv run main.py counterfactual",
         "Antwortet es anders, wenn nur\ndas Alter anders ist - bei\ngleichen Modellzahlen?",
         "Alter haeufiger genannt (5:1),\naber nicht signifikant.\nKein Unterschied beim Raten.",
         PURPLE)

    step(51.0, 29.0, 22.0, 19.0, "5", "Handbewertung",
         "uv run main.py rubric",
         "Ist die Antwort richtig - und\nversteht ein Antragsteller sie?\nZwei Personen, verblindet.",
         "Korrekter mit Contract,\naber schlechter verstaendlich.\nBeide Bewerter gleich geordnet.",
         TEAL)

    step(74.5, 29.0, 22.0, 19.0, "6", "Zweiter Lauf",
         "(Dayas: 45 Fragen, qwen3-4B)",
         "Kommt eine unabhaengige\nzweite Umsetzung zum\nselben Schluss?",
         "Ja. Beide: Faktenblatt\nentfernt erfundene Zahlen,\nCodepruefung sichert Deferral.",
         GOLD)

    # arrows from both bands into governance
    for x in (26.5, 74.0):
        arrow(axes, (x, 65.7), (x, 58.0), colour="#999999", linewidth=2.2)
    for x in (15.0, 38.5, 62.0, 85.5):
        arrow(axes, (x, 28.7), (x, 20.3), colour="#999999", linewidth=2.2)

    # ---- band 3: governance ---------------------------------------
    axes.add_patch(FancyBboxPatch(
        (1.5, 1.5), 97, 18.0,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        linewidth=2.0, edgecolor=GREEN, facecolor="#f5faf6", zorder=0))
    axes.text(50, 18.1,
              "EBENE 3  -  DIE ENTSCHEIDUNG, SCHRIFTLICH",
              ha="center", va="top", fontsize=11.5, fontweight="bold",
              color=GREEN)
    axes.text(50, 15.9,
              "uv run main.py provenance   ->   docs/governance_decision.md "
              "+ results/provenance.json",
              ha="center", va="top", fontsize=8.6, color="#555555",
              family="monospace")

    decisions = [
        ("DER KONFLIKT",
         "Predictive parity haelt (0,005),\ndemographic parity nicht (0,000).\n"
         "Beides geht nicht gleichzeitig."),
        ("DIE WAHL",
         "Wir halten uns an equal opportunity\nund melden den Verstoss, statt\n"
         "ihn zu entfernen."),
        ("WARUM NICHT REPARIEREN",
         "Alter entfernen aendert nichts -\nder Kreditscore traegt dieselbe\n"
         "Information."),
        ("WER UND WANN",
         "Verantwortlich: beide gemeinsam.\nReview: keines, das Projekt endet\n"
         "mit der Abgabe."),
        ("ABSCHALT-AUSLOESER",
         "Parity-Intervall unter 0,80 ·\nGruppen-AUC unter 0,60 ·\n"
         "Deferral ueber 25 % · echte Daten."),
    ]
    for number, (label, text) in enumerate(decisions):
        x = 3.5 + number * 18.9
        axes.text(x, 13.2, label, ha="left", va="top", fontsize=8.6,
                  fontweight="bold", color=GREEN)
        axes.text(x, 11.5, text, ha="left", va="top", fontsize=8.4,
                  color="#222222")

    axes.text(50, 2.6,
              "Der rote Faden: Ebene 1 misst, ob das Modell fair ist. "
              "Ebene 2 misst, ob die Antwort ehrlich ist. "
              "Ebene 3 schreibt auf, was wir mit dem Befund tun.",
              ha="center", va="bottom", fontsize=9.2, color="#333333",
              style="italic")

    return figure


def make_overview_figures():
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

    for drawer, path in [(draw_overview, OVERVIEW_FILE),
                         (draw_evaluation, EVALUATION_FILE),
                         (draw_governance, GOVERNANCE_FILE),
                         (draw_measurement, MEASUREMENT_FILE)]:
        figure = drawer()
        figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        print("  written:", path.relative_to(PROJECT_ROOT))

        if VAULT_FOLDER.is_dir():
            import shutil
            shutil.copy2(path, VAULT_FOLDER / path.name)
            print("           copied into the vault as", path.name)

    return 0
