"""
Find rows where LLM-as-judge and COMET disagree on the effect of instructions.

Specifically: rows where the LLM judge rates the "with instruction" translation higher
(adaptedness), but COMET rates it lower than the "without instruction" translation.

Usage:
    python -m metrics.comet_judge_divergence --model google/gemma-3-27b-it --tgt-lang fra_Latn
    python -m metrics.comet_judge_divergence --all-langs
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from translate.bouquet_vllm import BouquetInstructionRow
from judge.bouquet import EvaluatedRow
from metrics.comet import get_savepath as get_comet_savepath


EVAL_MODEL = "gemini/gemini-3-flash-preview"
ALL_LANGS = ["fra_Latn", "ind_Latn", "ukr_Cyrl", "khm_Khmr", "jav_Latn"]


def parse_args():
    parser = argparse.ArgumentParser(description="Find LLM-judge vs COMET divergent rows")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--tgt-lang", type=str, default="fra_Latn")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--comet-model", type=str, default="Unbabel/XCOMET-XL")
    parser.add_argument("--eval-model", type=str, default=EVAL_MODEL)
    parser.add_argument("--top-n", type=int, default=30,
                        help="Number of top divergent rows to print")
    parser.add_argument("--all-langs", action="store_true",
                        help="Run analysis for all 5 target languages")
    return parser.parse_args()


def load_comet_scores(model: str, lang: str, split: str, use_instructions: bool, comet_model: str) -> dict[str, float]:
    """Load COMET scores into a dict keyed by uniq_id."""
    path = get_comet_savepath(model, lang, split, use_instructions, comet_model)
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            scores[data["uniq_id"]] = data["comet_score"]
    return scores


def load_judge_scores(model: str, lang: str, split: str, use_instructions: bool, eval_model: str) -> dict[str, EvaluatedRow]:
    """Load LLM judge evaluations into a dict keyed by uniq_id."""
    model_name = model.split("/")[-1]
    path = EvaluatedRow.get_savepath(model_name, lang, split, use_instructions, eval_model)
    rows = EvaluatedRow.load(path)
    return {er.evaluation.uniq_id: er for er in rows}


def load_translations(model: str, lang: str, split: str, use_instructions: bool) -> dict[str, BouquetInstructionRow]:
    """Load raw translations into a dict keyed by uniq_id."""
    path = BouquetInstructionRow.get_savepath(model, lang, split, use_instructions)
    rows = BouquetInstructionRow.load(path)
    return {r.uniq_id: r for r in rows}


def analyze_lang(args, lang: str, print_examples: bool = True) -> dict:
    """Analyze judge-vs-COMET divergence for a single language. Returns per-domain stats."""
    print(f"\nLoading data for {args.model} / {lang} / {args.split}...")

    comet_with = load_comet_scores(args.model, lang, args.split, True, args.comet_model)
    comet_without = load_comet_scores(args.model, lang, args.split, False, args.comet_model)

    judge_with = load_judge_scores(args.model, lang, args.split, True, args.eval_model)
    judge_without = load_judge_scores(args.model, lang, args.split, False, args.eval_model)

    trans_with = load_translations(args.model, lang, args.split, True)
    trans_without = load_translations(args.model, lang, args.split, False)

    # Join on uniq_id
    all_ids = set(comet_with) & set(comet_without) & set(judge_with) & set(judge_without)
    print(f"Joined {len(all_ids)} rows across all files.")

    # Compute deltas
    divergent = []
    all_rows = []
    for uid in sorted(all_ids):
        adapt_delta = judge_with[uid].evaluation.adaptedness_score - judge_without[uid].evaluation.adaptedness_score
        rating_delta = judge_with[uid].evaluation.translation_rating_score - judge_without[uid].evaluation.translation_rating_score
        comet_delta = comet_with[uid] - comet_without[uid]

        row_info = {
            "uniq_id": uid,
            "domain": trans_with[uid].domain if uid in trans_with else "?",
            "register": trans_with[uid].register if uid in trans_with else "?",
            "adapt_delta": adapt_delta,
            "rating_delta": rating_delta,
            "comet_delta": comet_delta,
            "adapt_with": judge_with[uid].evaluation.adaptedness_score,
            "adapt_without": judge_without[uid].evaluation.adaptedness_score,
            "rating_with": judge_with[uid].evaluation.translation_rating_score,
            "rating_without": judge_without[uid].evaluation.translation_rating_score,
            "comet_with": comet_with[uid],
            "comet_without": comet_without[uid],
        }
        all_rows.append(row_info)

        # Judge says "with" is better but COMET says "without" is better
        if adapt_delta > 0 and comet_delta < 0:
            divergent.append(row_info)

    # Sort by largest divergence (adapt_delta - comet_delta, i.e. biggest disagreement)
    divergent.sort(key=lambda r: r["adapt_delta"] - r["comet_delta"] * 100, reverse=True)

    # --- Aggregate stats ---
    print(f"\n{'=' * 80}")
    print(f"AGGREGATE STATS — {lang}")
    print(f"{'=' * 80}")
    print(f"Total rows: {len(all_rows)}")
    print(f"Divergent rows (judge prefers w/, COMET prefers w/o): {len(divergent)} ({100*len(divergent)/len(all_rows):.1f}%)")

    # Count other cases
    n_both_prefer_with = sum(1 for r in all_rows if r["adapt_delta"] > 0 and r["comet_delta"] > 0)
    n_both_prefer_without = sum(1 for r in all_rows if r["adapt_delta"] < 0 and r["comet_delta"] < 0)
    n_comet_prefers_with_judge_not = sum(1 for r in all_rows if r["adapt_delta"] <= 0 and r["comet_delta"] > 0)
    n_tied = sum(1 for r in all_rows if r["adapt_delta"] == 0)
    print(f"Both prefer w/ instructions: {n_both_prefer_with}")
    print(f"Both prefer w/o instructions: {n_both_prefer_without}")
    print(f"COMET prefers w/ but judge doesn't: {n_comet_prefers_with_judge_not}")
    print(f"Judge tied (adapt_delta == 0): {n_tied}")

    # Average deltas
    avg_adapt = sum(r["adapt_delta"] for r in all_rows) / len(all_rows)
    avg_comet = sum(r["comet_delta"] for r in all_rows) / len(all_rows)
    print(f"\nAvg adaptedness delta: {avg_adapt:+.1f}")
    print(f"Avg COMET delta: {avg_comet:+.4f}")

    # Per-domain breakdown with mean deltas
    domain_stats = defaultdict(lambda: {
        "divergent": 0, "total": 0,
        "comet_delta_sum": 0.0, "adapt_delta_sum": 0.0,
    })
    for r in all_rows:
        d = domain_stats[r["domain"]]
        d["total"] += 1
        d["comet_delta_sum"] += r["comet_delta"]
        d["adapt_delta_sum"] += r["adapt_delta"]
    for r in divergent:
        domain_stats[r["domain"]]["divergent"] += 1

    print(f"\nPer-domain breakdown:")
    print(f"  {'Domain':<25} {'Div%':>5} {'Div/Tot':>8} {'mean ΔC':>8} {'mean ΔA':>8}")
    print(f"  {'-'*25} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    for domain in sorted(domain_stats, key=lambda d: domain_stats[d]["comet_delta_sum"] / domain_stats[d]["total"]):
        s = domain_stats[domain]
        mean_comet = s["comet_delta_sum"] / s["total"]
        mean_adapt = s["adapt_delta_sum"] / s["total"]
        div_pct = 100 * s["divergent"] / s["total"]
        print(f"  {domain:<25} {div_pct:>4.0f}% {s['divergent']:>3}/{s['total']:>3} {mean_comet:>+8.4f} {mean_adapt:>+7.1f}")

    # --- Print top divergent rows ---
    if print_examples:
        top_n = min(args.top_n, len(divergent))
        print(f"\n{'=' * 80}")
        print(f"TOP {top_n} DIVERGENT ROWS (judge prefers w/, COMET prefers w/o)")
        print(f"{'=' * 80}")

        for i, r in enumerate(divergent[:top_n]):
            uid = r["uniq_id"]
            tw = trans_with.get(uid)
            two = trans_without.get(uid)

            print(f"\n--- [{i+1}] {uid} | {r['domain']} | register: {r['register']} ---")
            if tw:
                print(f"  Source (EN):    {tw.tgt_text}")
                print(f"  Reference:     {tw.src_text}")
                print(f"  Instruction:   {tw.user_instruction}")
            if two:
                print(f"  W/o instr:     {two.pred_text}")
            if tw:
                print(f"  W/  instr:     {tw.pred_text}")
            print(f"  Judge adapt:   {r['adapt_without']:>3} → {r['adapt_with']:>3}  (Δ {r['adapt_delta']:+d})")
            print(f"  Judge rating:  {r['rating_without']:>3} → {r['rating_with']:>3}  (Δ {r['rating_delta']:+d})")
            print(f"  COMET:         {r['comet_without']:.4f} → {r['comet_with']:.4f}  (Δ {r['comet_delta']:+.4f})")

            # Show judge comments
            jw = judge_with.get(uid)
            jwo = judge_without.get(uid)
            if jwo:
                print(f"  Judge w/o:     {jwo.evaluation.comment}")
            if jw:
                print(f"  Judge w/:      {jw.evaluation.comment}")

    return dict(domain_stats)


def main():
    args = parse_args()

    if args.all_langs:
        langs = ALL_LANGS
    else:
        langs = [args.tgt_lang]

    all_domain_stats = {}  # lang -> domain -> stats
    for lang in langs:
        # Only print examples for single-language mode
        stats = analyze_lang(args, lang, print_examples=len(langs) == 1)
        all_domain_stats[lang] = stats

    # Cross-language summary when running all langs
    if len(langs) > 1:
        print(f"\n{'=' * 80}")
        print(f"CROSS-LANGUAGE PER-DOMAIN SUMMARY (mean COMET Δ)")
        print(f"{'=' * 80}")

        # Collect all domains
        all_domains = sorted({d for stats in all_domain_stats.values() for d in stats})

        header = f"  {'Domain':<25}" + "".join(f" {l[:3]:>8}" for l in langs)
        print(header)
        print(f"  {'-'*25}" + " --------" * len(langs))
        for domain in sorted(all_domains, key=lambda d: sum(
            all_domain_stats[l].get(d, {"comet_delta_sum": 0, "total": 1})["comet_delta_sum"]
            / all_domain_stats[l].get(d, {"comet_delta_sum": 0, "total": 1})["total"]
            for l in langs
        ) / len(langs)):
            row = f"  {domain:<25}"
            for lang in langs:
                s = all_domain_stats[lang].get(domain)
                if s:
                    mean_c = s["comet_delta_sum"] / s["total"]
                    row += f" {mean_c:>+8.4f}"
                else:
                    row += f" {'n/a':>8}"
            print(row)

        # Also print divergence % cross-language table
        print(f"\n  {'Domain':<25}" + "".join(f" {l[:3]:>8}" for l in langs))
        print(f"  {'-'*25}" + " --------" * len(langs))
        for domain in sorted(all_domains, key=lambda d: sum(
            all_domain_stats[l].get(d, {"divergent": 0, "total": 1})["divergent"]
            / all_domain_stats[l].get(d, {"divergent": 0, "total": 1})["total"]
            for l in langs
        ) / len(langs), reverse=True):
            row = f"  {domain:<25}"
            for lang in langs:
                s = all_domain_stats[lang].get(domain)
                if s:
                    pct = 100 * s["divergent"] / s["total"]
                    row += f" {pct:>7.0f}%"
                else:
                    row += f" {'n/a':>8}"
            print(row)


if __name__ == "__main__":
    main()
