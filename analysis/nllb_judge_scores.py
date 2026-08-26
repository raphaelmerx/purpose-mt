"""Print NLLB-3.3B vs Gemma-3 BOUQuET judge scores in the table format we want.

Reads judge eval files saved by judge.bouquet and
prints two side-by-side tables (rating + adaptedness) with rows:
  - NLLB-3.3B (raw scores)
  - Delta vs NLLB for each Gemma-3 size x instruction condition

Usage:
    PYTHONPATH=. .venv/bin/python report_nllb_judge_scores.py
"""
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import Optional

from judge.bouquet import EvaluatedRow

LANGS = ["fra_Latn", "ind_Latn", "ukr_Cyrl", "khm_Khmr", "jav_Latn"]
LANG_SHORT = {
    "fra_Latn": "fra", "ind_Latn": "ind", "ukr_Cyrl": "ukr",
    "khm_Khmr": "khm", "jav_Latn": "jav",
}
SPLIT = "dev"
EVAL_MODEL = "gemini/gemini-3-flash-preview"
EVAL_SLUG = EVAL_MODEL.replace("/", "-").replace(":", "-")
BASE = Path("translation_results/qualitative_analysis")


def load_metrics(model_name: str, lang: str, use_instr: bool) -> Optional[tuple[float, float]]:
    """Return (rating_avg, adaptedness_avg) or None if eval file missing.

    `model_name` may contain "/" (e.g. "facebook/nllb-200-3.3B"); the judge
    save path embeds it literally, which creates a subdirectory.
    """
    suffix = "with_instructions" if use_instr else "without_instructions"
    path = BASE / f"eval_bouquet_{lang}_{SPLIT}_{model_name}_{suffix}_{EVAL_SLUG}.jsonl"
    if not path.exists():
        return None
    rows = EvaluatedRow.load(path)
    metrics = EvaluatedRow.compute_metrics(rows)
    return metrics.translation_rating_avg, metrics.adaptedness_avg


def fmt_row_raw(label: str, scores: list[Optional[tuple[float, float]]]) -> str:
    """Format raw-score row: rating then adaptedness."""
    rating = [s[0] if s else None for s in scores]
    adapt = [s[1] if s else None for s in scores]
    rating_avg = sum(r for r in rating if r is not None) / max(1, sum(1 for r in rating if r is not None))
    adapt_avg = sum(a for a in adapt if a is not None) / max(1, sum(1 for a in adapt if a is not None))

    def cell(x):
        return f"{x:5.1f}" if x is not None else "  ?  "
    rating_str = "  ".join(cell(r) for r in rating) + f"  | {rating_avg:5.1f}"
    adapt_str = "  ".join(cell(a) for a in adapt) + f"  | {adapt_avg:5.1f}"
    return f"{label:<28}  {rating_str}    {adapt_str}"


def fmt_row_delta(label: str, scores: list[Optional[tuple[float, float]]],
                  ref: list[Optional[tuple[float, float]]]) -> str:
    rating_d = []
    adapt_d = []
    for s, r in zip(scores, ref):
        if s is None or r is None:
            rating_d.append(None); adapt_d.append(None)
        else:
            rating_d.append(s[0] - r[0])
            adapt_d.append(s[1] - r[1])
    valid_r = [x for x in rating_d if x is not None]
    valid_a = [x for x in adapt_d if x is not None]
    avg_r = sum(valid_r) / len(valid_r) if valid_r else 0.0
    avg_a = sum(valid_a) / len(valid_a) if valid_a else 0.0

    def cell(x):
        if x is None:
            return "  ?  "
        return f"{x:+5.1f}"
    rating_str = "  ".join(cell(x) for x in rating_d) + f"  | {avg_r:+5.1f}"
    adapt_str = "  ".join(cell(x) for x in adapt_d) + f"  | {avg_a:+5.1f}"
    return f"{label:<28}  {rating_str}    {adapt_str}"


def main():
    nllb = [load_metrics("facebook/nllb-200-3.3B", l, False) for l in LANGS]
    gemma = {}
    for size in ["4b", "12b", "27b"]:
        for cond in [False, True]:
            gemma[(size, cond)] = [
                load_metrics(f"gemma-3-{size}-it", l, cond) for l in LANGS
            ]

    header_langs = "  ".join(f"{LANG_SHORT[l]:>5}" for l in LANGS)
    print()
    print(f"{'':<28}  {'Translation Rating':^45}    {'Adaptedness':^45}")
    print(f"{'':<28}  {header_langs}  | {'Avg.':>5}    {header_langs}  | {'Avg.':>5}")
    print("-" * 130)
    print(fmt_row_raw("NLLB-3.3B", nllb))
    print("-" * 130)
    for size in ["4b", "12b", "27b"]:
        for cond in [False, True]:
            label = f"Gemma-3-{size.upper()} {'w/' if cond else 'w/o'}"
            print(fmt_row_delta(label, gemma[(size, cond)], nllb))
    print()


if __name__ == "__main__":
    main()
