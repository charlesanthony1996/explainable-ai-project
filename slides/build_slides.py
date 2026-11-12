"""Build Michael's six slides into the team deck.

Slides 1 to 13 are the team's and are not touched. Slides 14 to 19 are his
part, and they tell one story:

    14  what was built            the three steps, end to end
    15  how it explains           SHAP, the per-applicant shares
    16  does it work              three levels of support, measured
    17  is it fair                equal opportunity, group by group
    18  how it was checked        both halves of the system, measured
    19  what was decided          the conflict, the choice, the limits

Four of the six carry a full-width figure drawn by
`uv run main.py slides`. Those slides keep the small eyebrow line and the
source line but drop the title box, because each figure already carries its
own heading; two headings stacked would only eat the height the figure
needs.

Templates are borrowed from the deck itself. Character budgets measured from
them: block label <= 43, block text <= 95, closing line <= 78, stat value
<= 11, stat caption <= 73, note <= 142, column label <= 19, column text
<= 76, wide block <= 292. Every string below stays inside, which is the only
defence against overflow without a renderer here.
"""

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
FIGURES = Path(r"C:\explainable-ai-project\results\figures")

PLAN = [
    ("13", "story"),
    ("13", "shap"),
    ("7", "evaluation"),
    ("13", "equal_opportunity"),
    ("13", "measurement"),
    ("13", "governance"),
]

# A figure slide: no title box, the picture fills the space between the
# eyebrow and the source line. Given as a box rather than a width, because
# the four figures do not all have the same shape - the picture is scaled
# to fit and centred inside it. Eyebrow ends at 1.15, source starts at 6.82.
WIDE = "fit:1.22,11.90,5.35"

CONTENT = {
    "story": {
        "eyebrow": "MY PART \u00b7 WHAT I BUILT AND HOW I CHECKED IT",
        "source": "Source: slide_figures.py, run_all.sh \u00b7 14",
        "figure": ("folie_drei_schritte.png", WIDE),
        "notes":
            "This is my half of the project in one picture. Step 1 and step "
            "2 are the system: train two forests on 72,000 past "
            "applications, then for one applicant compute the largest "
            "amount that still reaches an 80% chance, the SHAP reasons, and "
            "whether a human has to decide. Step 3 is what I spent most of "
            "the time on: checking that the result is fair, that the "
            "explanation is honest, and writing down the decisions we made. "
            "The point of the last line: without step 3 the system is only "
            "plausible. With it, every claim on the next five slides can be "
            "checked against a file in the repository.",
    },
    "shap": {
        "eyebrow": "EXPLANATIONS \u00b7 TOPIC 2",
        "title": "SHAP: how much each detail mattered",
        "blocks": [
            ("WHAT IT IS",
             "A share of the decision given to every input, for one applicant, not in general."),
            ("HOW IT IS DEFINED",
             "Start from the average applicant. Add each detail in turn. The change it causes is its share."),
        ],
        "line": "RESULT: credit score is the strongest by far \u2014 three times the next input. "
                "Error 0.000000000000016.",
        "source": "Source: shap_beeswarm_dataset_1.png \u00b7 15",
        "figure": ("shap_beeswarm_dataset_1.png", 7.9, 2.32, 4.0),
        "notes":
            "Every dot is one applicant. Left of the middle line the detail "
            "pushed the approval chance down, right of it up, and the colour "
            "is whether that applicant's value was high or low. Credit score "
            "is the widest band by far. SHAP is exact for tree models, not "
            "an approximation: the shares add up to the prediction, and the "
            "error of that sum is 1.6 times ten to the minus fourteen. The "
            "top five shares are what the assistant is allowed to put into "
            "words.",
    },
    "evaluation": {
        "eyebrow": "EVALUATION \u00b7 RESEARCH QUESTION 2",
        "title": "Three levels of support, one measurable difference",
        "cards": [
            ("15 of 15", "source questions cited by the full system \u2014 none by A or B"),
            ("24 of 24", "answers inventing a citation marker when given no sources"),
            ("1 of 7", "flagged cases where the model alone named human review"),
        ],
        "note": "A: applicant data only. B: plus our computed figures. C: plus four cited "
                "passages. 24 questions, 4 applicants, 72 answers. qwen2.5:7b, seed 42, local. "
                "Exact McNemar: p = 0.000061 and p = 0.00000012.",
        "source": "Source: evaluation.py, evaluation_mcnemar.csv \u00b7 16",
        "notes":
            "Only one thing changes between the three runs: how much support "
            "Qwen gets. That is what makes the comparison fair. The contract "
            "is built for all three, but in run A it is simply not shown - "
            "it still decides whether an answer is right, so no human "
            "judgement is needed to mark A wrong. Two findings. First, "
            "citations only appear when the sources are actually there: 15 "
            "of 15 against 0 of 15, p = 0.000061. Second, the uncomfortable "
            "one: when the runs have no sources, they invent the markers "
            "anyway, in every single answer. A prompt rule produced the "
            "shape of a citation without the substance.",
    },
    "equal_opportunity": {
        "eyebrow": "FAIRNESS \u00b7 EQUAL OPPORTUNITY, WITH CONFIDENCE INTERVALS",
        "source": "Source: parity_by_group_with_intervals.csv \u00b7 17",
        "figure": ("folie_equal_opportunity.png", WIDE),
        "notes":
            "Equal opportunity asks one question: of the people who really "
            "did get a loan, what share does the model also approve? For "
            "three age groups that is between 88 and 97 percent. For the "
            "over-61 it is zero. Not low - zero. Of the 390 older applicants "
            "in the test set, 60 really were approved, and the model "
            "approves none of them. The gap of 0.966 is almost ten times "
            "what we allow. The confidence interval matters because 390 "
            "people is not many: resampling 2,000 times puts the gap between "
            "0.959 and 0.973, so it is a property of the model, not an "
            "accident of which rows landed in the test split.",
    },
    "measurement": {
        "eyebrow": "VERIFICATION \u00b7 TWO HALVES, BOTH MEASURED",
        "source": "Source: fairness_verification.py, evaluation.py, counterfactual_age.py \u00b7 18",
        "figure": ("folie_messung.png", WIDE),
        "notes":
            "A fair model with a lying explanation is no better than an "
            "unfair one, so both halves get measured. Left: is the model "
            "fair, and is the gap real rather than a small-sample accident. "
            "Middle: is the answer honest - does it name its source, does it "
            "invent one, and does it treat the same case differently when "
            "only the age changes. That last test is the strongest design "
            "here: the applicant is sent twice, as 30 and as 67, with every "
            "model output held exactly fixed, so any difference can only "
            "come from the language model. Age came up 5 times against 1, "
            "which is too few pairs to call significant. Right: the decision "
            "itself is written down rather than implied.",
    },
    "governance": {
        "eyebrow": "GOVERNANCE \u00b7 THE DECISION WE HAD TO MAKE",
        "source": "Source: governance_decision.md, provenance.json, control_zones.csv \u00b7 19",
        "figure": ("folie_governance.png", WIDE),
        "notes":
            "This is the slide I would most like a question on. Two fairness "
            "rules cannot both hold here, and that is a proven result, not "
            "our failure: when the groups differ as much as these do, no "
            "model satisfies both. Predictive parity holds - an approval "
            "means the same thing in every age group, gap 0.005. Equal "
            "opportunity fails completely. We chose equal opportunity as the "
            "rule that matters, because the harm that counts is to someone "
            "who would have repaid. We did not patch it: deleting the age "
            "field changes nothing, we tested that, because the credit score "
            "carries the same signal. Instead every over-61 case goes to a "
            "person and the model card says the model is not fit for that "
            "group. And one control we rejected on evidence: deferring when "
            "the two models disagree would have sent 2,257 of 7,800 cases to "
            "a human, on cases where the model was no worse.",
    },
}


def duplicate_slides():
    shutil.copy("base.pptx", "work.pptx")
    after = "slide13.xml"
    created = []
    for template, key in PLAN:
        out = subprocess.run(
            [UV, "run", "--with", "python-pptx", "--with", "lxml",
             "--with", "defusedxml", "--with", "Pillow", "python",
             str(SKILL / "scripts" / "add_slide.py"),
             "work.pptx", "slide" + template + ".xml",
             "-o", "work2.pptx", "--after", after],
            capture_output=True, text=True, cwd=HERE)
        found = re.search(r"Created ppt/slides/(slide\d+\.xml)", out.stdout or "")
        if not found:
            print("  failed:", key, ((out.stdout or "") + (out.stderr or ""))[-250:])
            return None
        after = found.group(1)
        created.append(key)
        shutil.move(HERE / "work2.pptx", HERE / "work.pptx")
    return created


def set_text(shape, text):
    paragraph = shape.text_frame.paragraphs[0]
    if not paragraph.runs:
        return
    paragraph.runs[0].text = text
    for extra in paragraph.runs[1:]:
        extra.text = ""
    for extra in shape.text_frame.paragraphs[1:]:
        for run in extra.runs:
            run.text = ""


def drop(shape):
    shape._element.getparent().remove(shape._element)


def set_notes(slide, donor, text):
    """Write speaker notes, borrowing a body placeholder if there is none.

    This deck's notes master declares no placeholders, so python-pptx
    creates an empty notes slide and `notes_text_frame` comes back None.
    The team's own slides do have one, so the shape is copied from there
    and emptied before the new text goes in.
    """

    import copy

    notes = slide.notes_slide
    if notes.notes_text_frame is None:
        body = donor.notes_slide.notes_placeholder
        notes.shapes._spTree.append(copy.deepcopy(body._element))
    notes.notes_text_frame.text = text


def fill(slide, content):
    """Map content by what each box is, not by reading order.

    Reading order is not enough: in a two-column layout the right column's
    label sorts before the left column's text, which once put an
    85-character sentence into a 0.26-inch label box.
    """

    from pptx.util import Emu, Inches

    boxes = [s for s in slide.shapes
             if s.has_text_frame and s.text_frame.text.strip()]
    boxes.sort(key=lambda s: s.top)

    set_text(boxes[0], content["eyebrow"])
    set_text(boxes[-1], content["source"])

    if "title" in content:
        set_text(boxes[1], content["title"])
        middle = boxes[2:-1]
    else:
        # A full-width figure carries its own heading; the title box would
        # repeat it and steal the height the picture needs.
        drop(boxes[1])
        middle = boxes[2:-1]

    if "cards" in content:
        values = sorted([s for s in middle if Emu(s.height).inches > 0.65
                         and Emu(s.top).inches < 3.0], key=lambda s: s.left)
        captions = sorted([s for s in middle if Emu(s.top).inches >= 3.0
                           and Emu(s.top).inches < 4.4], key=lambda s: s.left)
        note = [s for s in middle if Emu(s.top).inches >= 4.4]
        for box, (value, caption) in zip(values, content["cards"]):
            set_text(box, value)
        for box, (value, caption) in zip(captions, content["cards"]):
            set_text(box, caption)
        if note:
            set_text(note[0], content["note"])
        return

    if "columns" in content:
        labels = sorted([s for s in middle if Emu(s.height).inches <= 0.40],
                        key=lambda s: s.left)
        texts = sorted([s for s in middle
                        if 0.40 < Emu(s.height).inches < 1.1],
                       key=lambda s: s.left)
        wide = [s for s in middle if Emu(s.height).inches >= 1.1]
        for box, (label, text) in zip(labels, content["columns"]):
            set_text(box, label)
        for box, (label, text) in zip(texts, content["columns"]):
            set_text(box, text)
        if wide:
            set_text(wide[0], content["block"])
        return

    line = [s for s in middle if Emu(s.width).inches > 10]
    blocks = [s for s in middle if Emu(s.width).inches <= 10]
    wanted = content.get("blocks", [])

    # With a figure the text has to stack down the LEFT column, because the
    # right half is where the picture goes. Without one, the four blocks
    # read across the page as the template intends.
    if "figure" in content:
        order = lambda s: (s.left, s.top)
    else:
        order = lambda s: (round(Emu(s.top).inches * 2), s.left)

    labels = sorted([s for s in blocks if Emu(s.height).inches <= 0.40],
                    key=order)
    texts = sorted([s for s in blocks if Emu(s.height).inches > 0.40],
                   key=order)

    for place in range(min(len(labels), len(texts))):
        if place < len(wanted):
            set_text(labels[place], wanted[place][0])
            set_text(texts[place], wanted[place][1])
        else:
            # fewer blocks than the template offers: clear the rest so the
            # figure sits on empty ground instead of leftover text
            drop(labels[place])
            drop(texts[place])

    if line:
        if "line" in content:
            set_text(line[0], content["line"])
        else:
            drop(line[0])

    # Anything the template drew behind the text blocks - the rounded cards -
    # has no text and would show as empty boxes under a full-width figure.
    if "figure" in content and not wanted:
        for shape in list(slide.shapes):
            if shape.has_text_frame and not shape.text_frame.text.strip():
                drop(shape)

    if "figure" in content:
        place_figure(slide, content["figure"])


def place_figure(slide, figure):
    """Put the picture on the slide, either at a fixed spot or fitted.

    A fitted picture is scaled to the largest size that fits the given box
    and centred in it, so a figure whose shape changes cannot grow into the
    source line underneath.
    """

    from PIL import Image
    from pptx.util import Inches

    if isinstance(figure[1], str) and figure[1].startswith("fit:"):
        name = figure[0]
        top, box_width, box_height = [float(v)
                                      for v in figure[1][4:].split(",")]
        path = FIGURES / name
        if not path.exists():
            print("    figure missing:", name)
            return
        with Image.open(path) as picture:
            shape = picture.width / picture.height
        width = min(box_width, box_height * shape)
        height = width / shape
        slide.shapes.add_picture(
            str(path),
            Inches((13.333 - width) / 2),
            Inches(top + (box_height - height) / 2),
            width=Inches(width))
        return

    name, x, y, width = figure
    path = FIGURES / name
    if path.exists():
        slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                 width=Inches(width))
    else:
        print("    figure missing:", name)


def main():
    print("Duplicating templates")
    created = duplicate_slides()
    if created is None:
        return 1

    from pptx import Presentation
    pres = Presentation("work.pptx")
    print()
    for place, key in enumerate(created):
        content = CONTENT[key]
        slide = pres.slides[13 + place]
        fill(slide, content)
        if content.get("notes"):
            set_notes(slide, pres.slides[12], content["notes"])
        mark = "  + figure" if "figure" in content else ""
        name = content.get("title") or content["eyebrow"]
        print("  slide", 14 + place, "-", name, mark)

    out = "Trustworthy Loan Assessment Assistant - 19 slides.pptx"
    pres.save(out)
    print()
    print("saved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

