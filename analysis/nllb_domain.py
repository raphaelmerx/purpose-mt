"""Per-domain Gemma-3-12B - NLLB-3.3B delta on BOUQuET dev.

Pools all 5 languages per domain (so Δ is the cross-language average gap on
that domain). Reports rating and adaptedness, both for Gemma-12B w/o instr
and Gemma-12B w/ instr.

Usage:
    PYTHONPATH=. .venv/bin/python report_nllb_12b_per_domain.py
"""
from dotenv import load_dotenv
load_dotenv()

from collections import defaultdict
from pathlib import Path

from judge.bouquet import EvaluatedRow

LANGS = ["fra_Latn", "ind_Latn", "ukr_Cyrl", "khm_Khmr", "jav_Latn"]
SPLIT = "dev"
EVAL_SLUG = "gemini-gemini-3-flash-preview"
BASE = Path("translation_results/qualitative_analysis")


def load(model_name: str, lang: str, use_instr: bool):
    suffix = "with_instructions" if use_instr else "without_instructions"
    path = BASE / f"eval_bouquet_{lang}_{SPLIT}_{model_name}_{suffix}_{EVAL_SLUG}.jsonl"
    if not path.exists():
        return None
    return EvaluatedRow.load(path)


def avg_by_domain(rows):
    by_dom_rating = defaultdict(list)
    by_dom_adapt = defaultdict(list)
    for er in rows:
        by_dom_rating[er.row.domain].append(er.evaluation.translation_rating_score)
        by_dom_adapt[er.row.domain].append(er.evaluation.adaptedness_score)
    return {
        d: (sum(by_dom_rating[d]) / len(by_dom_rating[d]),
            sum(by_dom_adapt[d]) / len(by_dom_adapt[d]),
            len(by_dom_rating[d]))
        for d in by_dom_rating
    }


def pool_all_langs(model_name: str, use_instr: bool):
    """Return {domain: (rating_avg, adapt_avg, n)} pooled across LANGS."""
    rating = defaultdict(list)
    adapt = defaultdict(list)
    for lang in LANGS:
        rows = load(model_name, lang, use_instr)
        if rows is None:
            print(f"  MISSING {model_name} {lang} instr={use_instr}")
            continue
        for er in rows:
            rating[er.row.domain].append(er.evaluation.translation_rating_score)
            adapt[er.row.domain].append(er.evaluation.adaptedness_score)
    return {
        d: (sum(rating[d]) / len(rating[d]),
            sum(adapt[d]) / len(adapt[d]),
            len(rating[d]))
        for d in rating
    }


def main():
    nllb = pool_all_langs("facebook/nllb-200-3.3B", False)
    g12_no = pool_all_langs("gemma-3-12b-it", False)
    g12_yes = pool_all_langs("gemma-3-12b-it", True)

    domains = sorted(nllb.keys(), key=lambda d: -nllb[d][2])  # most-frequent first

    print()
    print(f"{'Domain':<22} {'n':>4}   {'NLLB R':>6} {'NLLB A':>6}   {'Δ R w/o':>8} {'Δ A w/o':>8}   {'Δ R w/':>8} {'Δ A w/':>8}")
    print("-" * 90)
    for d in domains:
        nllb_r, nllb_a, n = nllb[d]
        g_no_r, g_no_a, _ = g12_no.get(d, (0, 0, 0))
        g_yes_r, g_yes_a, _ = g12_yes.get(d, (0, 0, 0))
        print(
            f"{d:<22} {n:>4}   {nllb_r:>6.1f} {nllb_a:>6.1f}   "
            f"{g_no_r - nllb_r:>+8.1f} {g_no_a - nllb_a:>+8.1f}   "
            f"{g_yes_r - nllb_r:>+8.1f} {g_yes_a - nllb_a:>+8.1f}"
        )
    # All-domain average
    all_n = sum(v[2] for v in nllb.values())
    nllb_r_avg = sum(v[0] * v[2] for v in nllb.values()) / all_n
    nllb_a_avg = sum(v[1] * v[2] for v in nllb.values()) / all_n
    g_no_r_avg = sum(v[0] * v[2] for v in g12_no.values()) / all_n
    g_no_a_avg = sum(v[1] * v[2] for v in g12_no.values()) / all_n
    g_yes_r_avg = sum(v[0] * v[2] for v in g12_yes.values()) / all_n
    g_yes_a_avg = sum(v[1] * v[2] for v in g12_yes.values()) / all_n
    print("-" * 90)
    print(
        f"{'ALL':<22} {all_n:>4}   {nllb_r_avg:>6.1f} {nllb_a_avg:>6.1f}   "
        f"{g_no_r_avg - nllb_r_avg:>+8.1f} {g_no_a_avg - nllb_a_avg:>+8.1f}   "
        f"{g_yes_r_avg - nllb_r_avg:>+8.1f} {g_yes_a_avg - nllb_a_avg:>+8.1f}"
    )


if __name__ == "__main__":
    main()
