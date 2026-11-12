"""Does any text run off its figure, or collide with the text above it?

Draws each figure, asks matplotlib for the rendered extent of every text
object in canvas units, and checks two things:

  * nothing sticks out past the left or right edge of the canvas
  * no two left-aligned texts in the same column overlap vertically
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\explainable-ai-project\src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explainable_ai_project import slide_figures as sf

FIGURES = [
    (sf.three_steps, "drei_schritte"),
    (sf.plan, "plan"),
    (sf.fairness_metrics, "fairness_metrics"),
    (sf.equal_opportunity, "equal_opportunity"),
    (sf.rules_and_zones, "regeln_zonen"),
    (sf.contract_figure, "contract"),
    (sf.language_model, "sprachmodell"),
    (sf.not_covered, "nicht_abgedeckt"),
    (sf.measurement, "messung"),
    (sf.governance, "governance"),
]


def extents(figure, axes):
    figure.canvas.draw()
    inverse = axes.transData.inverted()
    found = []
    for artist in axes.texts:
        box = artist.get_window_extent(figure.canvas.get_renderer())
        (x0, y0), (x1, y1) = inverse.transform([(box.x0, box.y0),
                                                (box.x1, box.y1)])
        found.append((x0, x1, y0, y1, artist.get_text(),
                      artist.get_horizontalalignment()))
    return found


problems = 0
for drawer, name in FIGURES:
    figure = drawer()
    axes = figure.axes[0]
    low, high = axes.get_ylim()
    boxes = extents(figure, axes)

    for x0, x1, y0, y1, text, align in boxes:
        first = text.split("\n")[0][:40]
        if x0 < -0.5 or x1 > 100.5:
            print(f"[{name}] OFF CANVAS   x {x0:.1f}..{x1:.1f}  {first!r}")
            problems += 1
        if y0 < low - 0.5:
            print(f"[{name}] BELOW VIEW   y {y0:.1f} < {low:.1f}  {first!r}")
            problems += 1

    # Overlap: two texts whose horizontal spans meet and whose vertical
    # spans meet are printed on top of each other.
    for first in range(len(boxes)):
        ax0, ax1, ay0, ay1, atext, _ = boxes[first]
        for second in range(first + 1, len(boxes)):
            bx0, bx1, by0, by1, btext, _ = boxes[second]
            overlap_x = min(ax1, bx1) - max(ax0, bx0)
            overlap_y = min(ay1, by1) - max(ay0, by0)
            if overlap_x > 0.6 and overlap_y > 0.6:
                print(f"[{name}] OVERLAP      {atext.splitlines()[0][:30]!r}"
                      f"  <>  {btext.splitlines()[0][:30]!r}")
                problems += 1
    plt.close(figure)

print()
print("problems:", problems)
