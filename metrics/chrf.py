"""
Calculate per-sentence ChrF++ scores for BOUQuET LLM translations.

Uses sacrebleu's sentence_chrf (word_order=2) to compute ChrF++ for each
translation row and saves one JSONL line per sentence.

Output:
    translation_results/chrf/chrf_bouquet_{lang}_{split}_{model}_{suffix}.jsonl

Usage:
    python -m metrics.chrf --model google/gemma-3-27b-it --tgt-lang fra_Latn
    python -m metrics.chrf --model google/gemma-3-4b-it --tgt-lang ind_Latn --use-user-instructions

    # Multiple models and languages:
    python -m metrics.chrf --model google/gemma-3-4b-it,google/gemma-3-12b-it --tgt-lang fra_Latn,ind_Latn
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional

from sacrebleu import sentence_chrf

from translate.bouquet_vllm import BouquetInstructionRow


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate ChrF++ for BOUQuET translations")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                        help="LLM(s) used for translation (comma-separated for multiple)")
    parser.add_argument("--tgt-lang", type=str, default="fra_Latn",
                        help="Target language code(s) (comma-separated for multiple)")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"],
                        help="Dataset split to use")
    parser.add_argument("--use-user-instructions", action="store_true",
                        help="Use the translations that had user instructions in the prompt")
    return parser.parse_args()


def get_savepath(model: str, lang: str, split: str, use_instructions: bool) -> Path:
    """Get the path for saving ChrF results."""
    model_name = model.split("/")[-1]
    suffix = "with_instructions" if use_instructions else "without_instructions"
    return Path(f"translation_results/chrf/chrf_bouquet_{lang}_{split}_{model_name}_{suffix}.jsonl")


def calculate_chrf(rows: List[BouquetInstructionRow]) -> List[float]:
    """Compute per-sentence ChrF++ scores. Returns a list of scores (one per row)."""
    scores = []
    for row in rows:
        hyp = row.pred_text or ""
        ref = row.src_text  # English reference
        score = sentence_chrf(hyp, [ref], word_order=2).score
        scores.append(score)
    return scores


def evaluate_single(
    model: str,
    tgt_lang: str,
    split: str,
    use_user_instructions: bool,
) -> Optional[dict]:
    """Evaluate a single model/language combination. Returns results dict or None."""
    savepath = get_savepath(model, tgt_lang, split, use_user_instructions)
    if savepath.exists():
        print(f"ChrF scores already exist at {savepath}, skipping...")
        scores = []
        with open(savepath, "r", encoding="utf-8") as f:
            for line in f:
                scores.append(json.loads(line)["chrf_score"])
        overall = sum(scores) / len(scores) if scores else 0.0
        return {"model": model, "tgt_lang": tgt_lang, "use_instructions": use_user_instructions, "overall": overall}

    # Load translation results
    trans_path = BouquetInstructionRow.get_savepath(model, tgt_lang, split, use_user_instructions)
    if not trans_path.exists():
        print(f"Warning: Translation results not found at {trans_path}, skipping...")
        return None

    print(f"\nLoading translations from {trans_path}...")
    rows = BouquetInstructionRow.load(trans_path)
    valid_rows = [r for r in rows if r.pred_text is not None]
    print(f"Loaded {len(valid_rows)} rows with predictions (out of {len(rows)} total)")

    if not valid_rows:
        print("No valid predictions to evaluate.")
        return None

    scores = calculate_chrf(valid_rows)
    overall = sum(scores) / len(scores)

    # Per-domain
    domain_scores: dict[str, List[float]] = {}
    for row, score in zip(valid_rows, scores):
        domain_scores.setdefault(row.domain, []).append(score)

    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"Language: {tgt_lang}")
    print(f"Instructions: {'yes' if use_user_instructions else 'no'}")
    print(f"{'='*50}")
    print(f"Overall ChrF++: {overall:.2f}")
    print(f"\nPer-domain scores:")
    for domain in sorted(domain_scores):
        ds = domain_scores[domain]
        print(f"  {domain}: {sum(ds)/len(ds):.2f} (n={len(ds)})")

    # Save per-row results
    savepath.parent.mkdir(parents=True, exist_ok=True)
    with open(savepath, "w", encoding="utf-8") as f:
        for row, score in zip(valid_rows, scores):
            f.write(json.dumps({"uniq_id": row.uniq_id, "domain": row.domain, "chrf_score": score}) + "\n")
    print(f"Saved per-row scores to {savepath}")

    return {"model": model, "tgt_lang": tgt_lang, "use_instructions": use_user_instructions, "overall": overall}


def main():
    args = parse_args()

    models = [m.strip() for m in args.model.split(",")]
    tgt_langs = [l.strip() for l in args.tgt_lang.split(",")]

    print(f"Models: {models}")
    print(f"Target languages: {tgt_langs}")

    all_results = []
    for model in models:
        for tgt_lang in tgt_langs:
            result = evaluate_single(model, tgt_lang, args.split, args.use_user_instructions)
            if result:
                all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'Language':<12} {'ChrF++':>10}")
        print("-" * 60)
        for r in all_results:
            model_short = r['model'].split('/')[-1]
            print(f"{model_short:<30} {r['tgt_lang']:<12} {r['overall']:>10.2f}")


if __name__ == "__main__":
    main()
