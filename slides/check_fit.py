"""Does everything fit on the slide, and does nothing overlap?

No renderer here, so this works from the shape geometry python-pptx
reports, plus an estimate of how many lines each string needs. Tables are
measured per cell: a cell whose text needs more lines than its row height
allows will push the whole table taller in PowerPoint.
"""

import sys

from pptx import Presentation
from pptx.util import Emu

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DECK = "Trustworthy Loan Assessment Assistant - simple.pptx"
FIRST, LAST = 14, 19
SLIDE_HEIGHT = 7.5
FOOTER_TOP = 6.80

pres = Presentation(DECK)
problems = 0


def lines_needed(text, width_inches, size_pt):
    """Characters that fit on one line, at roughly 0.52 em average."""
    if not text:
        return 1
    per_line = max(1, int(width_inches * 96 / (size_pt * 0.52)))
    longest = 0
    for part in text.split("\n"):
        longest += max(1, -(-len(part) // per_line))
    return longest


def first_run_size(frame, fallback=12.0):
    for paragraph in frame.paragraphs:
        for run in paragraph.runs:
            if run.font.size is not None:
                return run.font.size.pt
    return fallback


for number in range(FIRST, LAST + 1):
    slide = pres.slides[number - 1]
    print("=== slide", number)
    boxes = []

    for shape in slide.shapes:
        top = Emu(shape.top).inches
        left = Emu(shape.left).inches
        width = Emu(shape.width).inches
        height = Emu(shape.height).inches

        if shape.has_table:
            table = shape.table
            needed = 0.0
            for row_index, row in enumerate(table.rows):
                row_inches = Emu(row.height).inches
                worst = 1
                for column_index, cell in enumerate(row.cells):
                    column_inches = Emu(
                        table.columns[column_index].width).inches
                    size = first_run_size(cell.text_frame)
                    worst = max(worst, lines_needed(
                        cell.text, column_inches - 0.18, size))
                line_inches = size * 1.25 / 72.0
                needed += max(row_inches, worst * line_inches + 0.08)
            flag = ""
            if needed > height + 0.03:
                flag = "  TABLE GROWS by %.2f in" % (needed - height)
                problems += 1
            print("   table  %d rows  at y=%.2f  h=%.2f  needs %.2f%s"
                  % (len(table.rows), top, height, needed, flag))
            boxes.append((top, top + max(height, needed), "table"))
            continue

        if not (shape.has_text_frame and shape.text_frame.text.strip()):
            continue

        text = shape.text_frame.text
        size = first_run_size(shape.text_frame)
        rows = lines_needed(text, width - 0.36, size)
        needed = rows * size * 1.25 / 72.0 + 0.08
        flag = ""
        if needed > height + 0.10:
            flag = "  OVERFLOW by %.2f in" % (needed - height)
            problems += 1
        print("   text   %4d ch %4.1fpt at y=%.2f h=%.2f needs %.2f%s"
              % (len(text), size, top, height, needed, flag))
        if flag:
            print("          ", text[:72].replace("\n", " / "))
        boxes.append((top, top + max(height, needed), text[:28]))

    # Does anything run into the source line, or off the slide?
    for top, bottom, label in boxes:
        if top < FOOTER_TOP - 0.05 and bottom > FOOTER_TOP:
            print("   RUNS INTO THE FOOTER: %r ends at %.2f (footer %.2f)"
                  % (label, bottom, FOOTER_TOP))
            problems += 1
        if bottom > SLIDE_HEIGHT:
            print("   OFF THE SLIDE: %r ends at %.2f" % (label, bottom))
            problems += 1

print()
print("problems:", problems)
