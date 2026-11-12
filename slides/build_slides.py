"""Build Michael's part of the team deck: slides 14 to 19.

EVERYTHING ON THESE SLIDES IS REAL POWERPOINT TEXT.
No picture carries a word of it. Titles, statements, table cells and the
closing message are all text boxes and native PowerPoint tables, so every
word can be changed in PowerPoint without coming back here.

Six slides, about fifty seconds each, for a five-minute slot:

    14  the path          two things had to be proved, steps 1 to 7
    15  step 3            fairness measured four ways
    16  step 4            why we follow equal opportunity
    17  steps 5 and 6     which rules fire, and where cases land
    18  step 7            the contract, and what the check proved
    19  the limits        the numbers we chose ourselves

The wording is deliberately plain. Short sentences, no clause stacking,
no words that need a second language to parse.

Slides 1 to 13 are the team's and are not touched.
"""

import copy
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SKILL = Path(r"C:\Users\micha\.claude\skills\synced"
             r"\c686bb79-314b-47fc-8cc8-15198ef780ba_5531554f-ae05-4653-aabd-8a71abc2b30c"
             r"\pptx")
UV = r"C:\Users\micha\.local\bin\uv.exe"

# The deck's own colours.
INK = "14213D"
PAPER = "F7F5F0"
MUTED = "4D566A"
RUST = "B5541A"
BLUE = "2F6FB3"
GREEN = "1F7A4D"
WHITE = "FFFFFF"
TINT_RUST = "FBEFE7"
TINT_GREEN = "F3F8F5"
TINT_GREY = "EFEDE7"

SLIDE_WIDTH = 13.333
MARGIN = 0.89
BODY_WIDTH = SLIDE_WIDTH - 2 * MARGIN


# ------------------------------------------------------------------
# The content. Everything below is just words and numbers.
# ------------------------------------------------------------------

SLIDES = [
    {
        "key": "path",
        "eyebrow": "MY PART \u00b7 STEP BY STEP",
        "title": "Two things to prove",
        "statement": "Is it fair?      \u00b7      Does it hand over when it "
                     "does not know?",
        "table": {
            "columns": [1.5, 4.2, 7.35],
            "header": ["Step", "What", "Key point"],
            "rows": [
                ["1 \u2013 2", "Data and models", "team, slides 3 and 5"],
                ["3", "Measure fairness", "limits set first"],
                ["4", "Choose the fairness rule", "three fail, one holds"],
                ["5", "Rules: who goes to a person", "evidence, not taste"],
                ["6", "Four zones", "from automatic to stop"],
                ["7", "Check the model's answer", "numbers and citations"],
            ],
            "highlight": None,
        },
        "message": "Limits written down BEFORE measuring \u00b7 broken only "
                   "if the WHOLE range falls outside",
        "message_colour": GREEN,
        "source": "Source: governance_decision.md \u00b7 14",
        "notes":
            "My part starts at step 3. Steps 1 and 2 are the team's slides "
            "3 and 5, so I will not repeat them. One thing to stress here: "
            "we wrote the limits down before we measured anything. And a "
            "measure only counts as broken when the whole confidence "
            "range sits outside the limit, not just the middle value. That "
            "is what stops us from moving the goalposts after seeing the "
            "numbers.",
    },
    {
        "key": "measures",
        "eyebrow": "STEP 3 \u00b7 MEASURING FAIRNESS",
        "title": "Fairness is not one number. We measured four.",
        "statement": None,
        "table": {
            "columns": [3.3, 5.05, 1.3, 1.9, 1.5],
            "row_height": 0.50,
            "header": ["Measure", "Asks", "Result", "Limit", ""],
            "rows": [
                ["Demographic parity\n(difference)",
                 "Same approval rate in every age group?",
                 "0.834", "0.10 or less", "FAILS"],
                ["Demographic parity\n(ratio)",
                 "Four-fifths rule (EEOC)",
                 "0.000", "0.80 or more", "FAILS"],
                ["Equal opportunity",
                 "Of those who would repay \u2014 who gets through?",
                 "0.966", "0.10 or less", "FAILS"],
                ["Predictive parity",
                 "A \"yes\" equally often right?",
                 "0.005", "0.10 or less", "HOLDS"],
            ],
            "highlight": 2,
        },
        "message": "Three fail, one holds \u00b7 all four at once is "
                   "impossible when groups truly differ \u00b7 so we chose",
        "message_colour": GREEN,
        "source": "Source: parity_table_with_intervals.csv \u00b7 15",
        "notes":
            "Fairness is not one thing, so we measured four. Demographic "
            "parity asks whether every group is approved as often. Equal "
            "opportunity asks only about the people who would really "
            "repay. Predictive parity asks whether a yes means the same in "
            "every group. Three fail and one holds. Every number here has "
            "a confidence range behind it from 2,000 resamples, and we "
            "only call a measure broken when the whole range is outside "
            "the limit. If anyone asks about the second dataset: the "
            "groups there are much smaller, 120 older applicants instead "
            "of 390, and five of the eight measures stay undecided. "
            "Undecided is not the same as fair, and we report it that way.",
    },
    {
        "key": "choice",
        "eyebrow": "STEP 4 \u00b7 THE CHOICE",
        "title": "We follow equal opportunity",
        "reasons": [
            ("Predictive parity",
             "already holds — nothing to choose"),
            ("Demographic parity",
             "same rate for all. But groups truly differ: 73.1% approved "
             "at 31-45, only 15.4% at 61+ → would mean saying yes to "
             "people who cannot repay"),
            ("Equal opportunity",
             "only those who would have repaid — the harm that counts"),
        ],
        "table": {
            "columns": [2.6, 2.3, 3.3, 3.35],
            "header": ["Age group", "People",
                       "Really approved",
                       "Of those, model finds"],
            "rows": [
                ["18-30", "1,888", "63.3%", "92.2%"],
                ["31-45", "3,989", "73.1%", "96.6%"],
                ["46-60", "1,533", "54.3%", "88.7%"],
                ["61+", "390", "15.4%", "0.0%"],
            ],
            "highlight": 3,
        },
        "message": "Younger groups 0.079 apart, inside the limit \u00b7 "
                   "problem is 61+ alone \u00b7 removing age: 0.966 before, "
                   "0.966 after",
        "message_colour": RUST,
        "source": "Source: parity_by_group_with_intervals.csv \u00b7 16",
        "notes":
            "Why this measure and not another. Predictive parity already "
            "holds, so there is nothing to decide. Demographic parity "
            "would force the same approval rate in every group, but the "
            "groups really are different - 73 percent of the middle group "
            "were approved and only 15 percent of the oldest. Forcing "
            "equal rates would mean approving people who cannot repay, "
            "which helps nobody. Equal opportunity only looks at the "
            "people who would have repaid, and asks whether each group "
            "gets through equally. That is the harm that counts. The table "
            "shows what happens: 92, 97 and 89 percent in the three "
            "younger groups, and zero in the oldest. Of 390 older "
            "applicants, 60 really did get a loan, and the model approves "
            "none of them.",
    },
    {
        "key": "rules",
        "eyebrow": "STEPS 5 AND 6 \u00b7 WHO DECIDES",
        "title": "The model decides only where it is good enough",
        "statement": "Rule switched on only if the model is 0.03 worse on "
                     "the cases it catches",
        "table": {
            "columns": [2.5, 4.75, 1.6, 1.6, 1.1],
            "row_height": 0.29,
            "head_height": 0.34,
            "header": ["Rule", "Fires when",
                       "Right there", "Overall", "Used?"],
            "rows": [
                ["missing data", "a required field is empty",
                 "\u2014", "\u2014", "YES"],
                ["unusual input", "value outside the normal range",
                 "\u2014", "\u2014", "YES"],
                ["weak group", "cannot rank this group \u2014 AUC under 0.60",
                 "\u2014", "\u2014", "YES"],
                ["borderline", "chance between 0.75 and 0.85",
                 "77.7%", "85.0%", "YES"],
                ["disagreement", "the two models differ by over 0.50",
                 "85.5%", "85.0%", "NO"],
            ],
            "highlight": 4,
        },
        "table2": {
            "columns": [2.5, 1.8, 7.25],
            "row_height": 0.29,
            "head_height": 0.34,
            "header": ["Zone", "Share", "Who looks"],
            "rows": [
                ["auto-accept", "66.1%", "nobody"],
                ["sample audit", "21.1%", "1 in 10 read afterwards"],
                ["human review", "0.9%", "person decides with the model"],
                ["safe-stop", "11.9%", "no number is given at all"],
            ],
            "highlight": 3,
        },
        "message": "14.9% reach a person \u00b7 61+: 100%, as safe-stop "
                   "\u00b7 at AUC 0.52 no number beats a bad number",
        "message_colour": RUST,
        "source": "Source: deferral_rule_choice.csv, control_zones.csv \u00b7 17",
        "notes":
            "Five rules were candidates. Three are safety rules and are "
            "always on. Two had to earn their place: a rule is only used "
            "if the model is clearly worse on the cases it catches. "
            "Borderline qualifies, 77.7 against 85.0 overall. Disagreement "
            "does not - 85.5 against 85.0, the model is actually slightly "
            "better there - so it is off. Switching it on would have sent "
            "2,257 of 7,738 cases to a person for nothing. That is a "
            "control we rejected on evidence, not on taste. The four zones "
            "show where everything lands. Two points people ask about. "
            "Sample audit is every automatic refusal, not a narrow band, "
            "and the 10 percent is a rule we set - nobody actually read "
            "them, there was no person in the loop. And safe-stop is not "
            "the same as human review: review means the model gives a "
            "number and a person checks it, safe-stop means the model has "
            "no basis at all and gives no number. The over-61 are in "
            "safe-stop because a number they could anchor on would be "
            "worse than none. One more thing worth saying: the fairness "
            "choice and the weak-group rule are separate. Equal "
            "opportunity is what we report against; the AUC rule is what "
            "actually stops the case. They happen to point at the same "
            "group.",
    },
    {
        "key": "contract",
        "eyebrow": "STEP 7 \u00b7 CHECKING THE ANSWER",
        "title": "Our numbers may be quoted. Never invented.",
        "steps": [
            ("1 \u00b7 CONTRACT  \u2014  written first",
             "applicant data \u00b7 computed chances \u00b7 largest amount "
             "\u00b7 SHAP shares\n\nNothing else allowed in the answer."),
            ("2 \u00b7 QWEN  \u2014  writes the answer",
             "turns the list into sentences\n\nRun C also gets 4 passages: "
             "EU AI Act \u00b7 EBA \u00b7 our reports"),
            ("3 \u00b7 CHECK  \u2014  in code, afterwards",
             "every number against the list\nevery citation against the "
             "passages\n\nFails \u2192 1 retry \u2192 fixed safe answer. "
             "Never needed in 72 runs."),
        ],
        "table": {
            "columns": [5.35, 2.0, 2.0, 2.2],
            "header": ["24 questions \u00b7 4 applicants",
                       "A: data only", "B: + contract", "C: + sources"],
            "rows": [
                ["Source questions with a real citation",
                 "0 of 15", "0 of 15", "15 of 15"],
                ["Answers with a made-up citation mark",
                 "24 of 24", "24 of 24", "0"],
            ],
            "highlight": None,
        },
        "message": "Citations only where real sources exist \u00b7 without "
                   "them the model invents the look of a source \u2014 in "
                   "every single answer",
        "message_colour": BLUE,
        "source": "Source: contract.py, evaluation_mcnemar.csv \u00b7 18",
        "notes":
            "This is the design choice that makes the language part "
            "measurable at all. The model never computes anything. Before "
            "it is asked, the code writes down every figure the answer is "
            "allowed to contain, and afterwards the code checks the answer "
            "against that list. Three runs, same questions, same model: A "
            "gets the applicant's data only, B adds our contract, C adds "
            "four retrieved passages. Only the support changes, which is "
            "what makes the test fair. Two results. Citations appear only "
            "in run C, 15 of 15 against 0 of 15. And the uncomfortable "
            "one: when there are no sources, the model invents a citation "
            "mark anyway, in all 24 answers. A prompt rule produced the "
            "shape of a citation without the substance. Both differences "
            "are far beyond chance - the p values are 0.000061 and "
            "0.00000012, which means a split like this would almost never "
            "happen if the set-ups were equally good.",
    },
    {
        "key": "limits",
        "eyebrow": "THE LIMITS",
        "title": "Four numbers in this talk we chose ourselves",
        "statement": None,
        "table": {
            "columns": [3.5, 2.2, 7.85],
            "header": ["What", "Value", "Where it comes from"],
            "rows": [
                ["Fairness limit", "0.10",
                 "we set it \u2014 no rule or study fixes this value"],
                ["Weak-group limit", "AUC 0.60",
                 "we set it \u2014 0.50 is guessing, so under 0.60 is near "
                 "guessing"],
                ["Audit share", "10%",
                 "we set it \u2014 and nobody ever did it: no person in "
                 "the loop"],
                ["The data", "synthetic",
                 "invented people \u2014 real data means redoing all of "
                 "this"],
            ],
            "highlight": None,
        },
        "message": "Also missing: no owner after hand-in \u00b7 no review "
                   "planned \u00b7 learning-to-defer not built",
        "message_colour": RUST,
        "source": "Source: provenance.json, governance_decision.md \u00b7 19",
        "notes":
            "I want to end here rather than on the results. Four numbers "
            "in this talk are ours, not derived from anything: the 0.10 "
            "fairness limit, the 0.60 AUC limit, the 10 percent audit "
            "share, and the fact that both datasets are synthetic. "
            "Declaring them in advance keeps them honest, but they are "
            "still our choices. And three things are simply missing: "
            "nobody owns this after the hand-in, no review is planned, and "
            "we did not build a model that learns when to defer. A "
            "governance record that lists only what went well is not a "
            "governance record.",
    },
]


# ------------------------------------------------------------------
# Building
# ------------------------------------------------------------------

def duplicate_slides(count):
    shutil.copy("base.pptx", "work.pptx")
    after = "slide13.xml"
    for _ in range(count):
        out = subprocess.run(
            [UV, "run", "--with", "python-pptx", "--with", "lxml",
             "--with", "defusedxml", "--with", "Pillow", "python",
             str(SKILL / "scripts" / "add_slide.py"),
             "work.pptx", "slide13.xml",
             "-o", "work2.pptx", "--after", after],
            capture_output=True, text=True, cwd=HERE)
        found = re.search(r"Created ppt/slides/(slide\d+\.xml)",
                          out.stdout or "")
        if not found:
            print("  failed:", ((out.stdout or "") + (out.stderr or ""))[-250:])
            return False
        after = found.group(1)
        shutil.move(HERE / "work2.pptx", HERE / "work.pptx")
    return True


def strip_slide(slide):
    """Keep the eyebrow, the title and the source line. Drop the rest.

    The template slide carries a layout we do not want; what we do want is
    its three framing text boxes, which already have the deck's fonts and
    positions.
    """

    from pptx.util import Emu

    boxes = [s for s in slide.shapes
             if s.has_text_frame and s.text_frame.text.strip()]
    boxes.sort(key=lambda s: s.top)
    eyebrow, title, source = boxes[0], boxes[1], boxes[-1]

    for shape in list(slide.shapes):
        if shape not in (eyebrow, title, source):
            shape._element.getparent().remove(shape._element)

    # The title box in this template is two lines tall; one is enough.
    title.height = Emu(int(0.62 * 914400))
    return eyebrow, title, source


def set_text(shape, text):
    """Replace the text, keeping the first run's formatting."""
    paragraph = shape.text_frame.paragraphs[0]
    if not paragraph.runs:
        return
    paragraph.runs[0].text = text
    for extra in paragraph.runs[1:]:
        extra.text = ""
    for extra in shape.text_frame.paragraphs[1:]:
        for run in extra.runs:
            run.text = ""


def add_text(slide, x, y, width, height, pieces, size=13, colour=INK,
             bold=False, align="left", space_after=0):
    """A plain text box. `pieces` is a string or a list of (text, bold)."""

    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt

    box = slide.shapes.add_textbox(Inches(x), Inches(y),
                                   Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = Inches(0)
    frame.margin_right = Inches(0)
    frame.margin_top = Inches(0)
    frame.margin_bottom = Inches(0)

    if isinstance(pieces, str):
        pieces = [(pieces, bold)]

    paragraph = frame.paragraphs[0]
    paragraph.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER,
                           "right": PP_ALIGN.RIGHT}[align]
    if space_after:
        paragraph.space_after = Pt(space_after)
    for text, is_bold in pieces:
        run = paragraph.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = is_bold
        run.font.color.rgb = RGBColor.from_string(colour)
    return box


def add_table(slide, x, y, spec, body_size=12, head_size=11):
    """A native PowerPoint table. Every cell stays editable.

    Row height comes from the spec where a table needs it: PowerPoint
    grows a row whose text wraps, and a grown row pushes everything below
    it off the slide, so a two-line cell has to be paid for here.
    """

    row_height = spec.get("row_height", 0.34)
    head_height = spec.get("head_height", 0.36)

    from pptx.dml.color import RGBColor
    from pptx.enum.text import MSO_ANCHOR
    from pptx.util import Inches, Pt

    header = spec["header"]
    rows = spec["rows"]
    widths = spec["columns"]
    highlight = spec.get("highlight")

    total_width = sum(widths)
    total_height = head_height + len(rows) * row_height

    graphic = slide.shapes.add_table(len(rows) + 1, len(header),
                                     Inches(x), Inches(y),
                                     Inches(total_width),
                                     Inches(total_height))
    table = graphic.table
    table.first_row = True
    table.horz_banding = False

    for index, width in enumerate(widths):
        table.columns[index].width = Inches(width)
    table.rows[0].height = Inches(head_height)
    for index in range(len(rows)):
        table.rows[index + 1].height = Inches(row_height)

    def write(cell, text, size, colour, bold, fill):
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor.from_string(fill)
        cell.margin_left = Inches(0.10)
        cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.02)
        cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        frame = cell.text_frame
        frame.word_wrap = True
        paragraph = frame.paragraphs[0]
        for old in list(paragraph.runs):
            old._r.getparent().remove(old._r)
        run = paragraph.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = RGBColor.from_string(colour)

    for column, text in enumerate(header):
        write(table.cell(0, column), text, head_size, WHITE, True, INK)

    for index, values in enumerate(rows):
        marked = index == highlight
        fill = TINT_RUST if marked else (PAPER if index % 2 else WHITE)
        colour = RUST if marked else INK
        for column, text in enumerate(values):
            bold = marked or column == 0 or (
                text.isupper() and len(text) <= 6)
            write(table.cell(index + 1, column), text, body_size,
                  colour, bold, fill)

    return y + total_height


def add_message(slide, x, y, width, text, colour):
    """The closing line, in a tinted box. Still editable text."""

    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR
    from pptx.util import Inches, Pt

    tint = {RUST: TINT_RUST, GREEN: TINT_GREEN, BLUE: "EEF3F9"}.get(
        colour, TINT_GREY)
    height = 0.62 if len(text) < 115 else 0.88

    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 Inches(x), Inches(y),
                                 Inches(width), Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor.from_string(tint)
    box.line.color.rgb = RGBColor.from_string(colour)
    box.line.width = Pt(1.5)
    box.shadow.inherit = False

    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = Inches(0.18)
    frame.margin_right = Inches(0.18)
    frame.margin_top = Inches(0.04)
    frame.margin_bottom = Inches(0.04)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor.from_string(INK)
    return y + height


def set_notes(slide, donor, text):
    """Speaker notes. The notes master here declares no placeholder, so
    the shape is borrowed from a slide that has one."""

    notes = slide.notes_slide
    if notes.notes_text_frame is None:
        body = donor.notes_slide.notes_placeholder
        notes.shapes._spTree.append(copy.deepcopy(body._element))
    notes.notes_text_frame.text = text


def build_slide(slide, content, donor):
    eyebrow, title, source = strip_slide(slide)
    set_text(eyebrow, content["eyebrow"])
    set_text(title, content["title"])
    set_text(source, content["source"])

    y = 2.05

    if content.get("statement"):
        add_text(slide, MARGIN, y, BODY_WIDTH, 0.34,
                 content["statement"], size=14, colour=MUTED)
        y += 0.42

    if content.get("reasons"):
        for headline, detail in content["reasons"]:
            add_text(slide, MARGIN, y, BODY_WIDTH, 0.24,
                     headline, size=13, colour=INK, bold=True)
            y += 0.26
            add_text(slide, MARGIN + 0.22, y, BODY_WIDTH - 0.22, 0.24,
                     detail, size=12, colour=MUTED)
            y += 0.28 if len(detail) < 95 else 0.50
        y += 0.14

    if content.get("steps"):
        column = BODY_WIDTH / 3 - 0.2
        for place, (headline, detail) in enumerate(content["steps"]):
            x = MARGIN + place * (column + 0.3)
            add_text(slide, x, y, column, 0.26, headline,
                     size=13, colour=BLUE, bold=True)
            add_text(slide, x, y + 0.30, column, 1.85, detail,
                     size=11.5, colour=MUTED)
        y += 2.28

    if content.get("table"):
        y = add_table(slide, MARGIN, y, content["table"]) + 0.14

    if content.get("table2"):
        y = add_table(slide, MARGIN, y, content["table2"]) + 0.14

    add_message(slide, MARGIN, y, BODY_WIDTH,
                content["message"], content["message_colour"])

    if content.get("notes"):
        set_notes(slide, donor, content["notes"])


def main():
    print("Duplicating templates")
    if not duplicate_slides(len(SLIDES)):
        return 1

    from pptx import Presentation
    pres = Presentation("work.pptx")
    print()
    for place, content in enumerate(SLIDES):
        build_slide(pres.slides[13 + place], content, pres.slides[12])
        print("  slide", 14 + place, "-", content["title"])

    out = "Trustworthy Loan Assessment Assistant - simple.pptx"
    pres.save(out)
    print()
    print("saved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

