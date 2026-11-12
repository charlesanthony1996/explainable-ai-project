# Building the presentation slides

Michael's part of the deck is slides 14 to 19. They are generated, not hand
edited, so a changed number in `results/` can be carried into the deck by
rerunning two commands.

## The figures

```
uv run main.py slides
```

Writes four figures into `results/figures/`:

| File | Slide | What it shows |
|---|---|---|
| `folie_drei_schritte.png` | 14 | training, prediction, check and govern |
| `folie_equal_opportunity.png` | 17 | the approval table by age group |
| `folie_messung.png` | 18 | the two halves that get measured |
| `folie_governance.png` | 19 | the fairness conflict and our choice |

`folie_equal_opportunity.png` carries its numbers as text so the figure can
be drawn without the models on disk. `check_equal_opportunity_rows()` reads
`results/tables/parity_by_group_with_intervals.csv` on every run and refuses
to draw if the two disagree, so a recomputed table cannot leave a stale
figure behind.

The spacing in these figures is computed, not placed by eye: see the module
docstring in `src/explainable_ai_project/slide_figures.py`.

## The deck

`build_slides.py` takes the team's 13-slide deck, duplicates template
slides from it, and fills slides 14 to 19 with the content in `CONTENT`.
It needs `base.pptx` (the team deck, 13 slides) in the same folder:

```
uv run --with python-pptx --with lxml --with defusedxml --with Pillow python build_slides.py
```

Four of the six slides carry a full-width figure. Those drop the title box,
because each figure already has its own heading.

## Checking it

There is no renderer on this machine, so the checks are analytic:

```
uv run --with python-pptx python check_fit.py     # does the text fit its box
uv run --with python-pptx python geometry.py "<deck>.pptx" 14 17 19
uv run --with python-pptx --with lxml --with defusedxml --with Pillow python <pptx-skill>/scripts/office/validate.py "<deck>.pptx" --original base.pptx
```

`geometry.py` prints every box and picture in inches, which is how the
pictures were sized to clear the source line at 6.82 in.
