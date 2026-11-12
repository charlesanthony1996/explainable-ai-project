"""Does every string fit the box it was put in?

No renderer here, so fit is estimated from the box width and the font size
on the run: roughly 1.9 characters per 0.1 inch at 11pt, scaled by size.
Conservative on purpose - it should complain before PowerPoint would.
"""

import sys

from pptx import Presentation
from pptx.util import Emu

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pres = Presentation("Trustworthy Loan Assessment Assistant - 19 slides.pptx")

problems = 0
for number in range(14, 20):
    slide = pres.slides[number - 1]
    print("=== slide", number)
    for shape in slide.shapes:
        if not (shape.has_text_frame and shape.text_frame.text.strip()):
            continue
        text = shape.text_frame.text
        runs = [r for p in shape.text_frame.paragraphs for r in p.runs]
        size = None
        for run in runs:
            if run.font.size is not None:
                size = run.font.size.pt
                break
        if size is None:
            size = 12.0

        width = Emu(shape.width).inches
        height = Emu(shape.height).inches
        # characters that fit on one line at this width and size
        per_line = max(1, width * 96 / (size * 0.52))
        lines_needed = -(-len(text) // int(per_line))
        line_height = size * 1.25 / 72.0
        needed = lines_needed * line_height

        flag = ""
        if needed > height * 1.35 + 0.05:
            flag = "  OVERFLOW"
            problems += 1
        print("   %4d chars  %4.1fpt  box %.2fx%.2f  needs %.2f%s"
              % (len(text), size, width, height, needed, flag))
        if flag:
            print("      ", text[:70])
print()
print("overflows:", problems)
