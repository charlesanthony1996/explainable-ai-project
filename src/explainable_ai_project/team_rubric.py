"""A rating page for the team's evaluation sheet.

WHY THIS EXISTS
---------------
The evaluation on the `dayas` branch generated 135 answers and filled in the
automatic number check, but the human columns - correctness, faithfulness,
citations, correct refusal - are empty for all 135, and the two age-pair
review sheets are empty too. That gap is visible on slide 12 of the deck,
which says the blind scores are still being completed.

Reading a 135-row CSV in a spreadsheet is not a job anyone finishes
carefully, and a German Excel opens a comma-separated file into one column,
which already cost this team two rounds. So the sheet becomes a page: one
card per answer, the reference answer and the model's own figures beside it,
and buttons that only permit the values the rubric allows.

The rubric is Dayas's, quoted rather than reinvented, including the "na"
option on citations.

WHAT IS RATED
-------------
Not all 135. A stratified subsample of ten questions, all three runs each, so
the pairing survives and two people can finish it in under an hour. Which ten
is fixed by a seed, so the sample is the same for both raters and can be
reported.

THE SHEET IS ALREADY BLIND
--------------------------
Checked before relying on it: `answer_id` is a hash with no run letter, and
`applicant_facts` and `sources_given` are filled for all 135 rows rather than
only for the runs that received them. So nothing on the card says which run
wrote the answer.

Usage
-----
    uv run main.py team_rubric            build the page
    uv run main.py team_rubric --read     read the filled sheets back
"""

import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent
TEAM_FOLDER = PROJECT_ROOT / "evaluation"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"

SOURCE_SHEET = TEAM_FOLDER / "llm_scores.csv"
PAGE_FILE = TEAM_FOLDER / "team_rubric_sheet.html"
RESULT_FILE = TABLES_FOLDER / "team_rubric_results.csv"
AGREEMENT_FILE = TABLES_FOLDER / "team_rubric_agreement.csv"

HOW_MANY_QUESTIONS = 10
SEED = 42

DIMENSIONS = [
    {
        "key": "correctness",
        "label": "Correctness",
        "options": [
            ("0", "wrong, contradicts the reference answer, or does not answer "
                  "the question"),
            ("1", "partly right: the main point is there, but something is "
                  "missing or one detail is wrong"),
            ("2", "the main point matches the reference answer and nothing "
                  "important is wrong"),
        ],
    },
    {
        "key": "faithfulness",
        "label": "Faithfulness",
        "options": [
            ("0", "an important unsupported or invented claim (a number, a "
                  "rule, a promise)"),
            ("1", "one minor unsupported claim (a small embellishment)"),
            ("2", "no unsupported claims"),
        ],
    },
    {
        "key": "citations",
        "label": "Citations",
        "options": [
            ("0", "no citation, or citations that do not fit at all"),
            ("1", "cites something relevant, but a cited source does not "
                  "support its sentence, or a needed source is missing"),
            ("2", "cites the right source(s), and each supports the sentence "
                  "it is attached to"),
            ("na", "the question only needs the model results, or this answer "
                   "was given no sources"),
        ],
    },
    {
        "key": "correct_refusal",
        "label": "Correct refusal",
        "options": [
            ("no", "refused something it could have answered, or failed to "
                   "refuse what it should have"),
            ("yes", "declined what it could not do, and answered what it could"),
        ],
    },
]


def read_source():
    """The team's sheet, read whatever encoding it arrived in."""

    if not SOURCE_SHEET.exists():
        return None

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            sheet = pd.read_csv(SOURCE_SHEET, encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        sheet = pd.read_csv(SOURCE_SHEET, encoding="utf-8",
                            encoding_errors="replace")

    sheet.columns = [str(name).strip().lstrip("﻿")
                     for name in sheet.columns]
    return sheet


def choose_subsample(sheet, how_many=HOW_MANY_QUESTIONS):
    """Ten questions, all their runs, spread across the categories.

    Drawn one per category in turn rather than at random over the whole set,
    so a category with six questions cannot take half the sample.
    """

    questions = (sheet[["question_id", "category"]]
                 .drop_duplicates()
                 .sort_values("question_id"))

    picked = []
    by_category = {name: list(group["question_id"])
                   for name, group in questions.groupby("category")}

    rng = pd.Series(sorted(by_category)).sample(
        frac=1.0, random_state=SEED).tolist()

    while len(picked) < how_many and any(by_category.values()):
        for category in rng:
            if len(picked) >= how_many:
                break
            if by_category.get(category):
                picked.append(by_category[category].pop(0))

    return sorted(picked)


def build_rows(sheet, question_ids):
    part = sheet[sheet["question_id"].isin(question_ids)].copy()
    # Shuffled so the three runs of one question are not read back to back;
    # seeded so both raters see the same order and can discuss row 7.
    part = part.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    rows = []
    for place, row in part.iterrows():
        rows.append({
            "order": place + 1,
            "answer_id": str(row["answer_id"]),
            "question_id": str(row["question_id"]),
            "category": str(row["category"]),
            "question": str(row["question"]),
            "reference": str(row.get("reference_answer", "")),
            "needed_sources": str(row.get("needed_sources", "")),
            "should_decline": str(row.get("should_decline", "")),
            "facts": str(row.get("applicant_facts", "")),
            "sources": str(row.get("sources_given", "")),
            "answer": str(row.get("answer", "")),
        })
    return rows


def build_page(rows, path=PAGE_FILE):
    data = json.dumps({"rows": rows, "dimensions": DIMENSIONS},
                      ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PAGE_TEMPLATE.replace("__DATA__", data), encoding="utf-8")
    return path


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Team rating sheet</title>
<style>
  :root { --bg:#f6f6f4; --card:#fff; --ink:#1b1b1a; --muted:#6a6a66;
          --line:#dddcd7; --accent:#1f5fa8; --warn:#b02828; --ok:#1f7a4d; }
  * { box-sizing: border-box; }
  body { margin:0; padding:0 16px 130px; background:var(--bg); color:var(--ink);
         font:16px/1.6 -apple-system,"Segoe UI",system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:10; background:var(--bg);
           border-bottom:1px solid var(--line); margin:0 -16px 20px;
           padding:14px 16px 10px; }
  .wrap { max-width:900px; margin:0 auto; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:14px; }
  .bar { height:6px; background:var(--line); border-radius:3px; margin-top:10px; }
  .bar div { height:100%; width:0%; background:var(--ok); border-radius:3px;
             transition:width .2s; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:10px; padding:18px 20px; margin-bottom:18px; }
  .card.done { border-left:4px solid var(--ok); }
  .num { font-weight:700; color:var(--accent); }
  .q { font-weight:600; margin:2px 0 12px; }
  .ref { background:#eef6ef; border:1px solid #cfe3d3; border-radius:8px;
         padding:10px 14px; margin-bottom:12px; font-size:14px; }
  .ref b { display:block; color:var(--ok); font-size:13px; margin-bottom:4px; }
  .answer { white-space:pre-wrap; background:#fbfbf9; border:1px solid var(--line);
            border-left:3px solid var(--accent); border-radius:6px;
            padding:14px 16px; margin-bottom:14px; }
  details { border:1px solid var(--line); border-radius:8px; padding:8px 14px;
            margin-bottom:14px; background:#fafaf8; }
  summary { cursor:pointer; font-size:14px; font-weight:600; }
  details pre { white-space:pre-wrap; font-size:13px; margin:8px 0 0;
                font-family:ui-monospace,Consolas,monospace; }
  .flag { color:var(--warn); font-weight:600; font-size:13px;
          margin-bottom:10px; }
  .dim { margin-bottom:12px; }
  .dim > b { display:block; margin-bottom:6px; font-size:15px; }
  label { display:block; cursor:pointer; padding:5px 10px; border-radius:6px;
          font-size:14px; }
  label:hover { background:#f0f0ec; }
  label .v { font-weight:700; display:inline-block; min-width:28px; }
  input[type=radio] { margin-right:8px; }
  textarea { width:100%; border:1px solid var(--line); border-radius:6px;
             padding:8px; font:inherit; font-size:14px; resize:vertical; }
  footer { position:fixed; bottom:0; left:0; right:0; background:var(--card);
           border-top:1px solid var(--line); padding:12px 16px; }
  .frow { max-width:900px; margin:0 auto; display:flex; gap:12px;
          align-items:center; flex-wrap:wrap; }
  button { font:inherit; padding:9px 16px; border-radius:7px; cursor:pointer;
           border:1px solid var(--accent); background:var(--accent); color:#fff; }
  button.ghost { background:#fff; color:var(--accent); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  input[type=text] { font:inherit; padding:8px 10px; border:1px solid var(--line);
                     border-radius:6px; }
</style>
</head>
<body>
<header><div class="wrap">
  <h1>Team rating sheet</h1>
  <div class="sub" id="status"></div>
  <div class="bar"><div id="bar"></div></div>
  <div class="sub" style="margin-top:6px">
    The run that produced each answer is hidden. Score the answer against the
    reference, not against which system you think wrote it.
  </div>
</div></header>

<div class="wrap" id="cards"></div>

<footer><div class="frow">
  <label style="padding:0">Your name:
    <input type="text" id="rater" placeholder="who is rating"></label>
  <button id="save" disabled>Download CSV</button>
  <button class="ghost" id="clear">Clear entries</button>
  <span class="sub" id="hint"></span>
</div></footer>

<script>
const DATA = __DATA__;
const KEY = "team-rubric-v1";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }

const esc = s => String(s === null || s === undefined ? "" : s)
  .replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

const cards = document.getElementById("cards");
DATA.rows.forEach(row => {
  const card = document.createElement("div");
  card.className = "card";
  card.id = "card-" + row.order;

  const dims = DATA.dimensions.map(d => {
    const opts = d.options.map(([value, text]) =>
      `<label><input type="radio" name="${d.key}-${row.order}" value="${value}">` +
      `<span class="v">${value}</span> ${esc(text)}</label>`).join("");
    return `<div class="dim"><b>${d.label}</b>${opts}</div>`;
  }).join("");

  const decline = String(row.should_decline).toLowerCase() === "yes"
    ? `<div class="flag">This question SHOULD be declined, at least in part.</div>` : "";

  card.innerHTML = `
    <div><span class="num">${row.order} of ${DATA.rows.length}</span>
      <span class="sub"> &middot; ${esc(row.question_id)} &middot; ${esc(row.category)}
      &middot; sources needed: ${esc(row.needed_sources)}</span></div>
    <div class="q">${esc(row.question)}</div>
    ${decline}
    <div class="ref"><b>Reference answer - this is the truth to score against</b>
      ${esc(row.reference)}</div>
    <div class="answer">${esc(row.answer)}</div>
    <details><summary>The model results and sources this answer was given</summary>
      <pre>${esc(row.facts)}</pre>
      <pre>${esc(row.sources)}</pre></details>
    ${dims}
    <div class="dim"><b>Note (optional)</b>
      <textarea rows="2" id="note-${row.order}"></textarea></div>`;
  cards.appendChild(card);
});

function collect() {
  return DATA.rows.map(row => {
    const entry = { note: document.getElementById("note-" + row.order).value };
    DATA.dimensions.forEach(d => {
      const picked = document.querySelector(
        `input[name="${d.key}-${row.order}"]:checked`);
      if (picked) entry[d.key] = picked.value;
    });
    return { row, entry };
  });
}

function refresh() {
  let complete = 0;
  collect().forEach(({ row, entry }) => {
    const done = DATA.dimensions.every(d => entry[d.key] !== undefined);
    if (done) complete++;
    document.getElementById("card-" + row.order).classList.toggle("done", done);
    state[row.order] = entry;
  });
  state.__rater = document.getElementById("rater").value;
  try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {}

  const total = DATA.rows.length;
  document.getElementById("status").textContent = complete + " of " + total + " scored";
  document.getElementById("bar").style.width = (100 * complete / total) + "%";
  document.getElementById("save").disabled = complete === 0;
  document.getElementById("hint").textContent = complete < total
    ? "A partial sheet is fine - unscored rows are reported, not guessed."
    : "Complete.";
}

function restore() {
  DATA.rows.forEach(row => {
    const saved = state[row.order] || {};
    DATA.dimensions.forEach(d => {
      if (saved[d.key] !== undefined) {
        const el = document.querySelector(
          `input[name="${d.key}-${row.order}"][value="${saved[d.key]}"]`);
        if (el) el.checked = true;
      }
    });
    if (saved.note) document.getElementById("note-" + row.order).value = saved.note;
  });
  if (state.__rater) document.getElementById("rater").value = state.__rater;
}

document.addEventListener("input", refresh);
document.addEventListener("change", refresh);

document.getElementById("clear").onclick = () => {
  if (!confirm("Clear every entry on this page?")) return;
  localStorage.removeItem(KEY);
  location.reload();
};

document.getElementById("save").onclick = () => {
  const rater = document.getElementById("rater").value || "";
  const cols = ["answer_id", "question_id", "category"]
    .concat(DATA.dimensions.map(d => d.key)).concat(["scorer", "notes"]);
  const q = v => {
    const s = v === undefined || v === null ? "" : String(v);
    return /[",\\n\\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const lines = [cols.join(",")];
  collect().forEach(({ row, entry }) => {
    lines.push([row.answer_id, row.question_id, row.category]
      .concat(DATA.dimensions.map(d => entry[d.key] === undefined ? "" : entry[d.key]))
      .concat([rater, entry.note]).map(q).join(","));
  });
  const blob = new Blob(["\\ufeff" + lines.join("\\r\\n")],
                        {type: "text/csv;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  const tag = rater.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-")
              .replace(/^-|-$/g, "") || "rater";
  link.download = "team_scores_" + tag + ".csv";
  link.click();
};

restore();
refresh();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------
# Reading the filled sheets back
# ------------------------------------------------------------------

SCORE_KEYS = ["correctness", "faithfulness", "citations", "correct_refusal"]


def find_scored():
    places = [TEAM_FOLDER, TABLES_FOLDER, Path.home() / "Downloads"]
    found = []
    for place in places:
        if place.is_dir():
            found.extend(place.glob("team_scores*.csv"))
    by_name = {}
    for path in sorted(found, key=lambda one: one.stat().st_mtime):
        by_name[path.name] = path
    return sorted(by_name.values(), key=lambda one: one.stat().st_mtime)


def read_scored():
    paths = find_scored()
    if not paths:
        return None

    parts = []
    for one in paths:
        part = pd.read_csv(one, encoding="utf-8-sig")
        if "scorer" not in part.columns or part["scorer"].isna().all():
            part["scorer"] = one.stem.replace("team_scores_", "")
        print("Reading:", one, "-", len(part), "rows")
        parts.append(part)

    sheet = pd.concat(parts, ignore_index=True)
    return sheet.drop_duplicates(subset=["scorer", "answer_id"], keep="last")


def report_scored(sheet, source):
    """Means per run, and how far the raters agree."""

    runs = source[["answer_id", "question_id"]].copy()
    merged = sheet.merge(runs, on=["answer_id"], how="left",
                         suffixes=("", "_source"))

    numeric = ["correctness", "faithfulness"]
    print()
    print("-" * 70)
    print("PER RATER")
    print("-" * 70)
    for key in numeric:
        if key not in merged:
            continue
        values = pd.to_numeric(merged[key], errors="coerce")
        table = values.groupby(merged["scorer"]).agg(["count", "mean"]).round(3)
        print(key)
        print(table.to_string())

    raters = sorted(merged["scorer"].dropna().unique())
    rows = []
    if len(raters) >= 2:
        from sklearn.metrics import cohen_kappa_score
        first, second = raters[0], raters[1]
        left = merged[merged["scorer"] == first].set_index("answer_id")
        right = merged[merged["scorer"] == second].set_index("answer_id")
        shared = left.index.intersection(right.index)
        for key in SCORE_KEYS:
            if key not in merged or len(shared) == 0:
                continue
            one = left.loc[shared, key].astype(str)
            two = right.loc[shared, key].astype(str)
            usable = one.notna() & two.notna() & (one != "nan") & (two != "nan")
            if usable.sum() < 2:
                continue
            rows.append({
                "dimension": key,
                "rater_1": first, "rater_2": second,
                "answers_both_scored": int(usable.sum()),
                "exact_agreement": round(float((one[usable] == two[usable]).mean()), 3),
                "kappa": round(float(cohen_kappa_score(one[usable], two[usable])), 3),
            })

    if rows:
        agreement = pd.DataFrame(rows)
        print()
        print("-" * 70)
        print("AGREEMENT BETWEEN RATERS")
        print("-" * 70)
        print(agreement.to_string(index=False))
        TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
        agreement.to_csv(AGREEMENT_FILE, index=False)
        print("  written:", AGREEMENT_FILE.relative_to(PROJECT_ROOT))

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    merged.to_csv(RESULT_FILE, index=False)
    print("  written:", RESULT_FILE.relative_to(PROJECT_ROOT))


# ------------------------------------------------------------------

def run_team_rubric(read_back=False):
    source = read_source()
    if source is None:
        print("The team's sheet is not here:", SOURCE_SHEET)
        print()
        print("It lives on the dayas branch. The folder has to exist first,")
        print("or the shell reports a missing path rather than a missing file:")
        print()
        print("  mkdir evaluation")
        print("  git show origin/dayas:evaluation/llm_scores.csv > "
              "evaluation/llm_scores.csv")
        print("  git show origin/dayas:evaluation/rubric.md > "
              "evaluation/rubric.md")
        return 1

    if read_back:
        scored = read_scored()
        if scored is None:
            print("No filled sheet found. Rate first, then run this again.")
            return 1
        report_scored(scored, source)
        return 0

    question_ids = choose_subsample(source)
    rows = build_rows(source, question_ids)
    path = build_page(rows)

    print("Team sheet:", len(source), "answers,",
          source["question_id"].nunique(), "questions")
    print("Rating a stratified subsample:", len(question_ids), "questions,",
          len(rows), "answers")
    print("  questions:", ", ".join(question_ids))
    print()
    print("  written:", path.relative_to(PROJECT_ROOT))
    print()
    print("Open it in a browser, put your name in, score, press Download CSV.")
    print("Then:  uv run main.py team_rubric --read")
    return 0
