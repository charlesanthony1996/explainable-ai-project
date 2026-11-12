# Building the presentation slides

Michael's part of the deck is slides 14 to 22. They are generated, not hand
edited, so a changed number in `results/` can be carried into the deck by
rerunning two commands.

## The story

Slide 15 is the guiding slide: the seven steps of the work, what controls
each one, and which file proves it. Every slide after it redeems one of
those promises, in the same order, and each answers the same four
questions — what it is, what we did, what came out, what we decided.

| Slide | Step | Figure |
|---|---|---|
| 14 | the project in three steps | `folie_drei_schritte.png` |
| 15 | **the plan** — control, path, evidence | `folie_plan.png` |
| 16 | step 3: the four fairness measures | `folie_fairness_metrics.png` |
| 17 | step 4: the conflict, group by group | `folie_equal_opportunity.png` |
| 18 | steps 5 and 6: rules and zones | `folie_regeln_zonen.png` |
| 19 | step 7a: the contract and the check | `folie_contract.png` |
| 20 | step 7b: what the model does by itself | `folie_sprachmodell.png` |
| 21 | what this record does not cover | `folie_nicht_abgedeckt.png` |
| 22 | SHAP — **hidden**, kept for questions | `shap_beeswarm_dataset_1.png` |

Steps 1 and 2 of the plan (data, models) have no slide of their own: the
team covers them on slides 3 and 5.

`folie_messung.png` and `folie_governance.png` are still generated. They
summarise the same material one page each, which is useful in the report
and in the vault, but the deck now walks through it step by step instead.

## The figures

```
uv run main.py slides
```

Writes all ten figures into `results/figures/` and copies them to the
Obsidian vault.

Two things are built in rather than left to care:

* **Spacing is computed, not placed by eye.** Every bullet advances by its
  own height, counting the lines its detail actually has, and every card
  is as tall as its content needs. See the module docstring in
  `src/explainable_ai_project/slide_figures.py`.
* **The equal-opportunity figure cross-checks itself.** It carries its
  numbers as text so it can be drawn without the models on disk, so
  `check_equal_opportunity_rows()` re-reads
  `results/tables/parity_by_group_with_intervals.csv` on every run and
  refuses to draw if the two disagree.

## The deck

`build_slides.py` takes the team's 13-slide deck, duplicates template
slides from it, and fills slides 14 to 22. It needs `base.pptx` (the team
deck, 13 slides) in the same folder:

```
uv run --with python-pptx --with lxml --with defusedxml --with Pillow python build_slides.py
```

Most slides carry a full-width figure and drop the title box, because each
figure already has its own heading. Slide 22 is marked hidden by setting
`show="0"` on the slide element — python-pptx has no API for it.

## Checking it

There is no renderer on this machine, so the checks are analytic:

```
uv run python -I slides/check_figures.py        # figures: off-canvas text, overlaps
uv run --with python-pptx python check_fit.py   # deck: does the text fit its box
uv run --with python-pptx python geometry.py "<deck>.pptx" 14 17 21
uv run --with python-pptx --with lxml --with defusedxml --with Pillow python <pptx-skill>/scripts/office/validate.py "<deck>.pptx" --original base.pptx
```

`check_figures.py` measures the rendered extent of every string and
reports anything that leaves the canvas or lands on top of something else;
`geometry.py` prints every box and picture in inches, which is how the
pictures were sized to clear the source line at 6.82 in.
