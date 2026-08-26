"""Extract per-model, per-language results for the "Full results per language" appendix.

For each model, prints a Markdown table with rows {Baseline, Few-shot, Paragraph,
Instructions} (+ delta rows) and columns {fra, ind, ukr, khm, jav, Avg} for each
of Translation Rating and Adaptedness. Conditions with no data are skipped.

Usage:
    PYTHONPATH=. .venv/bin/python paper/extract_full_results_tables.py
"""
from dotenv import load_dotenv
load_dotenv()

from judge.bouquet import EvaluatedRow

LANGS = ["fra_Latn", "ind_Latn", "ukr_Cyrl", "khm_Khmr", "jav_Latn"]
LANG_SHORT = {"fra_Latn": "fra", "ind_Latn": "ind", "ukr_Cyrl": "ukr",
              "khm_Khmr": "khm", "jav_Latn": "jav"}
MODELS = [
    ("gemma-3-4b-it",  "Gemma-3-4B"),
    ("gemma-3-12b-it", "Gemma-3-12B"),
    ("gemma-3-27b-it", "Gemma-3-27B"),
    ("gemma-4-31b-it", "Gemma-4-31B"),
]
SPLIT = "dev"
EVAL_MODEL = "gemini/gemini-3-flash-preview"

# (label, kwargs to EvaluatedRow.get_savepath)
CONDITIONS = [
    ("Baseline",     dict(use_instructions=False)),
    ("Few-shot",     dict(use_instructions=False, semantic_fewshot=True)),
    ("Paragraph",    dict(use_instructions=False, paragraph_level=True)),
    ("Paragraph-P2", dict(use_instructions=False, paragraph_level=True, numbered_lists=True)),
    ("Instructions", dict(use_instructions=True)),
]


def load_metrics(model: str, lang: str, **kwargs):
    """Load eval rows and return (rating_avg, adapt_avg) or None if missing/empty."""
    path = EvaluatedRow.get_savepath(
        model=model, lang=lang, split=SPLIT, eval_model=EVAL_MODEL, **kwargs
    )
    if not path.exists():
        return None
    rows = EvaluatedRow.load(path)
    if not rows:
        return None
    m = EvaluatedRow.compute_metrics(rows)
    return m.translation_rating_avg, m.adaptedness_avg


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return f"{x:.1f}" if x is not None else "--"


def fmt_delta(x):
    if x is None:
        return "--"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}"


def _print_metric_table(metric_label: str, metric_idx: int, present, scores):
    """Print one Markdown sub-table for either Rating or Adaptedness."""
    print(f"\n**{metric_label}**\n")
    headers = ["Condition"] + [LANG_SHORT[l] for l in LANGS] + ["Avg"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for cond in present:
        vals = scores[cond][metric_idx]
        cells = [cond] + [fmt(v) for v in vals] + [fmt(avg(vals))]
        print("| " + " | ".join(cells) + " |")

    base = scores["Baseline"][metric_idx]
    for cond in present:
        if cond == "Baseline":
            continue
        cur = scores[cond][metric_idx]
        d = [(c - b) if (c is not None and b is not None) else None
             for c, b in zip(cur, base)]
        cells = [f"Δ {cond}−Base"] + [fmt_delta(v) for v in d] + [fmt_delta(avg(d))]
        print("| " + " | ".join(cells) + " |")


def render_model_table(model_id: str, model_label: str):
    print(f"\n## {model_label} (`{model_id}`)")

    # scores[cond] = (rating_vals_per_lang, adapt_vals_per_lang)
    scores = {}
    for cond_label, kwargs in CONDITIONS:
        scores[cond_label] = ([None] * len(LANGS), [None] * len(LANGS))
        for i, lang in enumerate(LANGS):
            res = load_metrics(model_id, lang, **kwargs)
            if res is not None:
                scores[cond_label][0][i] = res[0]
                scores[cond_label][1][i] = res[1]

    present = [c for c, _ in CONDITIONS
               if any(v is not None for v in scores[c][0] + scores[c][1])]
    if "Baseline" not in present:
        print("> No baseline data; skipping model.\n")
        return

    _print_metric_table("Translation Rating", 0, present, scores)
    _print_metric_table("Adaptedness", 1, present, scores)


def main():
    print("# Full per-model results (BOUQuET dev, judge = gemini-3-flash-preview)")
    print("\nLeft block = Translation Rating (0--100); right block = Adaptedness (0--100).")
    for mid, label in MODELS:
        render_model_table(mid, label)


if __name__ == "__main__":
    main()
