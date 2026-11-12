"""Build Michael's part of the team deck: slides 14 to 22.

Slides 1 to 13 are the team's and are not touched.

The story is one path with its proofs. Slide 15 is the guiding slide: the
seven steps, what controls each one, and which file proves it. Every slide
after it redeems one of those promises, in the same order, and each of them
answers the same four questions:

    WHAT IT IS        the definition, in words
    WHAT WE DID       the method and the sample
    THE RESULT        the numbers, in a table
    THE DECISION      what follows from them

    14  the whole project in three steps
    15  THE PLAN - seven steps, control and evidence
    16  step 3: the four fairness measures
    17  step 4: the conflict, group by group
    18  steps 5 and 6: rules and zones
    19  step 7a: the contract and the check
    20  step 7b: what the language model does by itself
    21  what this record does not cover
    22  SHAP - hidden, kept for the question round

Most slides carry a full-width figure drawn by `uv run main.py slides`.
Those keep the small eyebrow line and the source line but drop the title
box, because each figure already carries its own heading; two headings
stacked would only eat the height the figure needs.

Templates are borrowed from the deck itself. Character budgets measured
from them: block label <= 43, block text <= 95, closing line <= 78, stat
value <= 11, stat caption <= 73, note <= 200, column label <= 19, column
text <= 76, wide block <= 292. check_fit.py is what enforces this, since
there is no renderer on this machine.
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
FIGURES = Path(r"C:\explainable-ai-project\results\figures")

PLAN = [
    ("13", "story"),
    ("13", "theplan"),
    ("13", "fairness"),
    ("13", "conflict"),
    ("13", "rules"),
    ("13", "contract"),
    ("13", "language"),
    ("13", "notcovered"),
    ("13", "shap"),
]

# Slides that are in the file but not shown when presenting.
HIDDEN = {"shap"}

# A figure slide: no title box, the picture fills the space between the
# eyebrow and the source line. Given as a box rather than a width, because
# the figures do not all have the same shape - the picture is scaled to fit
# and centred inside it. Eyebrow ends at 1.15, source starts at 6.82.
WIDE = "fit:1.22,11.90,5.35"

CONTENT = {
    "story": {
        "eyebrow": "THE SYSTEM AS A WHOLE \u00b7 MY PART IS MARKED",
        "source": "Source: slide_figures.py, run_all.sh \u00b7 14",
        "figure": ("folie_drei_schritte.png", WIDE),
        "notes":
            "The whole system in one picture, so you can see where my part "
            "sits. Steps 1 and 2 are the team's model work: two forests on "
            "72,000 past applications, and for one applicant the largest "
            "amount that still reaches an 80% chance, with the SHAP "
            "reasons. The marked items are mine - inside step 2, the rules "
            "that decide whether a machine may answer at all, and then the "
            "whole of step 3. That is where most of my time went: checking "
            "that the result is fair, that the explanation is honest, and "
            "writing the decisions down. Without step 3 the system is only "
            "plausible. With it, every claim in the next seven slides can "
            "be checked against a file.",
    },
    "theplan": {
        "eyebrow": "CHECK AND GOVERNANCE \u00b7 EVERY STEP HAS A CONTROL AND A PROOF",
        "source": "Source: governance_decision.md, provenance.json \u00b7 15",
        "figure": ("folie_plan.png", WIDE),
        "notes":
            "This is the slide everything else hangs off. Read it in three "
            "columns. The middle is what was done, in order. The left is "
            "the control on that step - the rule we gave ourselves before "
            "doing it. The right is the file that proves it happened, and "
            "which slide shows it. Two things to point out. First, the "
            "tolerances at the top were written down BEFORE the numbers "
            "were computed, and a measure only counts as breached when the "
            "whole confidence interval lies outside - that is what stops "
            "us from moving the goalposts afterwards. Second, steps 1 and "
            "2 are the team's slides 3 and 5, so I will not repeat them. "
            "I start at step 3. If anyone asks who is accountable: Charles "
            "and I jointly, there is no review planned, and the project "
            "ends at submission - that is on the last slide rather than "
            "this one.",
    },
    "fairness": {
        "eyebrow": "STEP 3 \u00b7 FAIRNESS MEASURED, WITH CONFIDENCE INTERVALS",
        "source": "Source: fairness_verification.py, parity_table_with_intervals.csv \u00b7 16",
        "figure": ("folie_fairness_metrics.png", WIDE),
        "notes":
            "Four measures, because fairness is not one thing. Demographic "
            "parity asks whether every group is approved equally often - "
            "it is the crudest, and it ignores whether the groups differ. "
            "Equal opportunity asks only about the people who really would "
            "repay: of those, who gets through. Predictive parity asks "
            "whether a yes means the same thing in every group. Three are "
            "breached, one holds. The intervals matter: with 2,000 "
            "resamples the equal-opportunity gap sits between 0.959 and "
            "0.973, so it is a property of the model, not of which rows "
            "landed in the test split. The eight measures are these four "
            "at two cut-offs, 0.5 and 0.8. On dataset 1 all eight are "
            "decided: six breached, two held, both of them predictive "
            "parity. On dataset 2 the groups are far smaller - 120 over-61 "
            "instead of 390 - and five of the eight stay undecided, "
            "because the interval straddles the tolerance. Undecided is "
            "not the same as fair, and we report it as undecided.",
    },
    "conflict": {
        "eyebrow": "STEP 4 \u00b7 THE CONFLICT, GROUP BY GROUP",
        "source": "Source: parity_by_group_with_intervals.csv, governance_decision.md \u00b7 17",
        "figure": ("folie_equal_opportunity.png", WIDE),
        "notes":
            "Here is the measure that fails, in detail. Of the people who "
            "really did get a loan, the model approves 92 percent in the "
            "youngest group, 97 in the next, 89 in the next - and zero "
            "among the over-61. Of 390 older applicants, 60 really were "
            "approved, and the model approves none of them. Predictive "
            "parity still holds at 0.005, which is the trap: by that one "
            "measure the model looks fine. Both cannot be satisfied when "
            "the groups differ this much - that is a proven result, "
            "Chouldechova 2017 and Kleinberg 2016, not our failure. We "
            "chose equal opportunity, because the harm that counts is to "
            "someone who would have repaid. And we did not patch it: "
            "removing the age field changes the gap by nothing, 0.966 "
            "before and after, because the credit score carries the same "
            "signal. Two likely questions. Is 88.7 percent good enough for "
            "the 46-60 group? By our declared tolerance, yes: the three "
            "younger groups differ by only 0.079, inside the 0.10 limit. "
            "The whole breach is the over-61. And does choosing equal "
            "opportunity mean the over-61 always go to a human? Not "
            "directly - those are two separate things. The fairness choice "
            "is what we report against; what actually routes them is the "
            "weak-group rule, which fires because the within-group AUC "
            "there is 0.52, below our 0.60 limit. The two happen to point "
            "at the same group.",
    },
    "rules": {
        "eyebrow": "STEPS 5 AND 6 \u00b7 WHEN A PERSON DECIDES, AND WHY",
        "source": "Source: deferral.py, human_control.py, control_zones.csv \u00b7 18",
        "figure": ("folie_regeln_zonen.png", WIDE),
        "notes":
            "Two kinds of rule. Three safety rules fire when there is no "
            "basis for an answer at all: a missing field, a value outside "
            "anything in training, or a group the model is demonstrably "
            "poor on. Two evidence rules are candidates, and they have to "
            "earn their place: a rule goes on only if the model is "
            "measurably worse on the cases it catches, by at least 0.03. "
            "Borderline qualifies - 77.7 percent against 85.0 overall. "
            "Disagreement does not: 85.5 against 85.0, the model is "
            "actually slightly better there. Switching it on would have "
            "sent 2,257 of 7,738 cases to a person for nothing. That is "
            "the control we rejected, and we rejected it on evidence. Note "
            "what disagreement actually compares: both models run on every "
            "applicant, and the rule fires when their two probabilities "
            "differ by more than 0.50 - not the amounts, the "
            "probabilities. The four zones below show where everything "
            "lands. Sample audit is not a narrow band around the "
            "threshold: it is every automatic refusal, so everything below "
            "0.80 that no rule caught, and a tenth of that stream is read "
            "by a person. The cases between 0.75 and 0.85 never get there "
            "- the borderline rule pulls them into human review first. Two "
            "thirds are answered automatically, 14.9 percent reach a "
            "person, and every single over-61 case is in safe-stop, where "
            "the system gives no answer at all. One word on the last "
            "column: really approved is the share of people in that zone "
            "who did get a loan in the data, the true label, not the "
            "model's output.",
    },
    "contract": {
        "eyebrow": "STEP 7A \u00b7 THE CONTRACT: FIX THE NUMBERS BEFORE ASKING",
        "source": "Source: contract.py, evaluation.py, evaluation_mcnemar.csv \u00b7 19",
        "figure": ("folie_contract.png", WIDE),
        "notes":
            "The design choice that makes the whole language part "
            "measurable. Before Qwen is asked anything, the code writes a "
            "list of every figure the answer is allowed to contain: the "
            "probabilities, the maximum amount, the SHAP shares, the "
            "applicant's own data. The model may quote from that list and "
            "nothing else. Afterwards every number in the answer is "
            "matched back against it, and every citation against the "
            "passages actually retrieved. If a number or a citation fails "
            "the check, the model gets one more try; if that fails too, a "
            "fixed safe answer replaces its own. In 72 runs the fallback "
            "was never needed. The three runs on the right are the "
            "comparison: A is Qwen with the applicant's data only, B adds "
            "our contract, C adds four retrieved passages on top. Only the "
            "support changes, which is what makes it a fair test. Three "
            "results, all on 24 questions over 4 applicants. Citations: "
            "the full system cites "
            "a real source in 15 of 15 source questions, the other two "
            "runs in zero - exact McNemar p = 0.000061. Invented markers: "
            "when the runs have no sources, all 24 answers invent one "
            "anyway, p = 0.00000012. And 97.7 percent of the 131 figures "
            "in run C trace back to something we computed. If anyone asks "
            "what p means: it is how likely a split this lopsided would be "
            "if the two set-ups were really equally good. Below 0.05 we "
            "call the difference real; 0.000061 means it would happen "
            "about once in sixteen thousand times by chance.",
    },
    "language": {
        "eyebrow": "STEP 7B \u00b7 WHAT THE LANGUAGE MODEL DOES BY ITSELF",
        "source": "Source: llm.py, counterfactual_age.py, evaluation_rubric_results.csv \u00b7 20",
        "figure": ("folie_sprachmodell.png", WIDE),
        "notes":
            "Three findings about the model rather than the pipeline. "
            "First, the prompt. Version 1 said 'cite the source it came "
            "from' - a clear principle, and it produced zero citations in "
            "15 questions. Version 2 said where to put it: 'end every "
            "paragraph about regulation with the marker, like [2]' - 15 of "
            "15. Same rule, different wording. You get what you write, not "
            "what you mean. Second, the age test, which is the cleanest "
            "experiment here: the same applicant is sent twice, as 30 and "
            "as 67, with every model output held exactly fixed, so a "
            "difference can only come from the language. Age was brought "
            "up 5 times for the older applicant against 1 - the direction "
            "is clear but 24 pairs is not enough, p = 0.219 - and in "
            "discouraging language there was no difference at all. Third, "
            "the human rating: the contract makes answers more correct and "
            "measurably harder to read. That is a trade-off to state, not "
            "to hide.",
    },
    "notcovered": {
        "eyebrow": "THE LIMITS \u00b7 WHAT THIS RECORD DOES NOT COVER",
        "source": "Source: provenance.json, governance_decision.md \u00b7 21",
        "figure": ("folie_nicht_abgedeckt.png", WIDE),
        "notes":
            "I want to end on this rather than on the results. Everything "
            "on the guiding slide had a file behind it - except these six, "
            "and they are declarations, not measurements. There is no "
            "operating owner after submission and no review date, because "
            "the project ends. We cannot say how long a human review takes "
            "or how often a person would overrule the model, because no "
            "person ever worked a case. Learning-to-defer would be the "
            "next step and we did not take it. Both datasets are "
            "synthetic, so every number describes invented people. And the "
            "0.10 tolerance was set by us, not derived from a regulation - "
            "declaring it in advance keeps it honest, but it is still our "
            "number. A governance record that lists only what went well is "
            "not a governance record.",
    },
    "shap": {
        "eyebrow": "BACKUP \u00b7 SHAP, THE MODEL EXPLANATION",
        "title": "How much each detail mattered, for one applicant",
        "blocks": [
            ("WHAT IT IS",
             "A share of the decision given to every input, for one applicant, not in general."),
            ("HOW IT IS DEFINED",
             "Start from the average applicant. Add each detail in turn. The change it causes is its share."),
        ],
        "line": "Credit score 0.210, income 0.065, loan-to-income 0.021, age 0.016. "
                "Additivity error 1.6 \u00d7 10\u207b\u00b9\u2074 over 1,000 applicants.",
        "source": "Source: shap_importance_dataset_1.csv, shap_fidelity.csv \u00b7 backup",
        "figure": ("shap_beeswarm_dataset_1.png", 7.9, 2.32, 4.0),
        "notes":
            "Hidden slide, for the question round. Every dot is one "
            "applicant. Left of the middle line the detail pushed the "
            "approval chance down, right of it up; the colour is whether "
            "that applicant's value was high or low. Credit score is three "
            "times the next input. SHAP is exact for tree models, not an "
            "approximation: the shares add up to the prediction, and over "
            "1,000 applicants the largest error in that sum is 1.6 times "
            "ten to the minus fourteen. The top five shares are what the "
            "assistant is allowed to put into words. The team covers SHAP "
            "on slide 6; this slide is here only if someone asks for the "
            "exactness check or the per-feature numbers.",
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
        found = re.search(r"Created ppt/slides/(slide\d+\.xml)",
                          out.stdout or "")
        if not found:
            print("  failed:", key,
                  ((out.stdout or "") + (out.stderr or ""))[-250:])
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
    and filled with the new text.
    """

    notes = slide.notes_slide
    if notes.notes_text_frame is None:
        body = donor.notes_slide.notes_placeholder
        notes.shapes._spTree.append(copy.deepcopy(body._element))
    notes.notes_text_frame.text = text


def hide(presentation, index):
    """Mark a slide as not shown when presenting.

    python-pptx has no API for this; PowerPoint reads it off the `show`
    attribute of the slide part's root element.
    """

    presentation.slides[index]._element.set("show", "0")


def place_figure(slide, figure):
    """Put the picture on the slide, either at a fixed spot or fitted.

    A fitted picture is scaled to the largest size that fits the given box
    and centred in it, so a figure whose shape changes cannot grow into the
    source line underneath.
    """

    from PIL import Image
    from pptx.util import Inches

    name = figure[0]
    path = FIGURES / name
    if not path.exists():
        print("    figure missing:", name)
        return

    if isinstance(figure[1], str) and figure[1].startswith("fit:"):
        top, box_width, box_height = [float(v)
                                      for v in figure[1][4:].split(",")]
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

    _, x, y, width = figure
    slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                             width=Inches(width))


def fill(slide, content):
    """Map content by what each box is, not by reading order.

    Reading order is not enough: in a two-column layout the right column's
    label sorts before the left column's text, which once put an
    85-character sentence into a 0.26-inch label box.
    """

    from pptx.util import Emu

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

    line = [s for s in middle if Emu(s.width).inches > 10]
    blocks = [s for s in middle if Emu(s.width).inches <= 10]
    wanted = content.get("blocks", [])

    # With a figure the text stacks down the LEFT column, because the right
    # half is where the picture goes.
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
            drop(labels[place])
            drop(texts[place])

    if line:
        if "line" in content:
            set_text(line[0], content["line"])
        else:
            drop(line[0])

    # The rounded cards the template draws behind the text blocks carry no
    # text of their own; under a full-width figure they would show as empty
    # boxes.
    if "figure" in content and not wanted:
        for shape in list(slide.shapes):
            if shape.has_text_frame and not shape.text_frame.text.strip():
                drop(shape)

    if "figure" in content:
        place_figure(slide, content["figure"])


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
        index = 13 + place
        slide = pres.slides[index]
        fill(slide, content)
        if content.get("notes"):
            set_notes(slide, pres.slides[12], content["notes"])
        if key in HIDDEN:
            hide(pres, index)
        marks = ("  + figure" if "figure" in content else "") + \
                ("  [HIDDEN]" if key in HIDDEN else "")
        name = content.get("title") or content["eyebrow"]
        print("  slide", 14 + place, "-", name, marks)

    out = "Trustworthy Loan Assessment Assistant - 22 slides.pptx"
    pres.save(out)
    print()
    print("saved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
