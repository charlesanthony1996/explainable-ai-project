"""Where every box sits on the template slides, in inches."""

import sys

from pptx import Presentation
from pptx.util import Emu

deck = sys.argv[1]
wanted = [int(n) for n in sys.argv[2:]] or [7, 10, 13]

pres = Presentation(deck)
print("slide size: %.2f x %.2f in" % (Emu(pres.slide_width).inches,
                                      Emu(pres.slide_height).inches))
for number in wanted:
    slide = pres.slides[number - 1]
    print()
    print("=== slide", number, "===")
    rows = []
    for shape in slide.shapes:
        text = ""
        size = ""
        if shape.has_text_frame:
            text = shape.text_frame.text.replace("\n", " / ")[:46]
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if run.font.size:
                        size = str(run.font.size.pt)
                        break
                if size:
                    break
        rows.append((Emu(shape.top).inches, Emu(shape.left).inches,
                     Emu(shape.width).inches, Emu(shape.height).inches,
                     size, shape.shape_type, text))
    for top, left, width, height, size, kind, text in sorted(rows):
        print("  y=%5.2f x=%5.2f  w=%5.2f h=%5.2f  %-5s %-22s %s"
              % (top, left, width, height, size, str(kind)[:22], text))
