"""
Calculate COMET scores for BOUQuET LLM translations.

Usage:
    python -m metrics.comet --model google/gemma-3-27b-it --tgt-lang fra_Latn
    python -m metrics.comet --model google/gemma-3-4b-it --tgt-lang ind_Latn --use-user-instructions
    
    # Multiple models and languages (COMET model loaded only once):
    python -m metrics.comet --model google/gemma-3-4b-it,google/gemma-3-12b-it --tgt-lang fra_Latn,ind_Latn
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm

from common.comet_utils import load_comet_model, comet_cached_predict
from translate.bouquet_vllm import BouquetInstructionRow


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate COMET for BOUQuET translation")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it", 
                        help="LLM(s) used for translation (comma-separated for multiple)")
    parser.add_argument("--tgt-lang", type=str, default="fra_Latn",
                        help="Target language code(s) (comma-separated for multiple)")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"],
                        help="Dataset split to use")
    parser.add_argument("--use-user-instructions", action="store_true",
                        help="Use the translations that had user instructions in the prompt")
    parser.add_argument("--comet-model", type=str, 
                        default="Unbabel/XCOMET-XL",
                        help="COMET model to use (e.g., Unbabel/XCOMET-XL, Unbabel/wmt22-comet-da)")
    parser.add_argument("--paragraph-level", action="store_true",
                        help="Evaluate paragraph-level translations")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for COMET inference")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use")
    return parser.parse_args()


def calculate_comet(
    rows: List[BouquetInstructionRow],
    comet_model,
    batch_size: int = 8,
    gpus: int = 1,
    comet_model_name: str = "Unbabel/XCOMET-XL",
) -> List[float]:
    """
    Calculate COMET scores for translation results.
    Returns a list of scores (one per row). Higher scores are better (0-1 range).

    Based on https://github.com/Unbabel/COMET
    """
    # Prepare data in COMET format
    # For BOUQuET: tgt_text is the source (non-English), src_text is the reference (English)
    # pred_text is the machine translation output
    data = [
        {
            "src": row.tgt_text,  # Original source text
            "mt": row.pred_text or "",  # Machine translation
            "ref": row.src_text,  # Reference translation
        }
        for row in rows
    ]

    return comet_cached_predict(
        comet_model, data, batch_size=batch_size, gpus=gpus, model_name=comet_model_name,
    )


def evaluate_comet(rows: List[BouquetInstructionRow], scores: List[float]) -> dict:
    """Compute COMET scores overall and per domain."""
    if not scores:
        return {"overall": 0.0, "by_domain": {}}

    # Overall score
    overall = sum(scores) / len(scores)

    # Per-domain scores
    domain_scores: dict[str, List[float]] = {}
    for row, score in zip(rows, scores):
        if row.domain not in domain_scores:
            domain_scores[row.domain] = []
        domain_scores[row.domain].append(score)

    by_domain = {}
    for domain in sorted(domain_scores.keys()):
        domain_list = domain_scores[domain]
        by_domain[domain] = {
            "comet": sum(domain_list) / len(domain_list),
            "count": len(domain_list),
        }

    return {"overall": overall, "by_domain": by_domain}


def get_savepath(model: str, lang: str, split: str, use_instructions: bool, comet_model: str, paragraph_level: bool = False) -> Path:
    """Get the path for saving COMET results."""
    model_name = model.split("/")[-1]
    comet_name = comet_model.split("/")[-1]
    suffix = "with_instructions" if use_instructions else "without_instructions"
    if paragraph_level:
        suffix += "_paragraph"
    return Path(f"translation_results/comet/comet_{comet_name}_bouquet_{lang}_{split}_{model_name}_{suffix}.jsonl")


def evaluate_single(
    model: str,
    tgt_lang: str,
    split: str,
    use_user_instructions: bool,
    comet_model,
    comet_model_name: str,
    batch_size: int,
    gpus: int,
    paragraph_level: bool = False,
) -> Optional[dict]:
    """Evaluate a single model/language combination. Returns results dict or None if failed."""
    # Check if COMET scores already computed
    savepath = get_savepath(model, tgt_lang, split, use_user_instructions, comet_model_name, paragraph_level)
    if savepath.exists():
        print(f"COMET scores already exist at {savepath}, skipping...")
        # Load existing scores for summary
        scores = []
        with open(savepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                scores.append(data["comet_score"])
        overall = sum(scores) / len(scores) if scores else 0.0
        return {
            "model": model,
            "tgt_lang": tgt_lang,
            "use_instructions": use_user_instructions,
            "overall": overall,
        }
    
    # Load translation results
    trans_path = BouquetInstructionRow.get_savepath(
        model, tgt_lang, split, use_user_instructions, paragraph_level=paragraph_level
    )
    
    if not trans_path.exists():
        print(f"Warning: Translation results not found at {trans_path}, skipping...")
        return None

    print(f"\nLoading translations from {trans_path}...")
    rows = BouquetInstructionRow.load(trans_path)
    
    # Filter rows with predictions
    valid_rows = [r for r in rows if r.pred_text is not None]
    print(f"Loaded {len(valid_rows)} rows with predictions (out of {len(rows)} total)")

    if not valid_rows:
        print("No valid predictions to evaluate.")
        return None

    # Calculate COMET scores
    scores = calculate_comet(
        valid_rows,
        comet_model=comet_model,
        batch_size=batch_size,
        gpus=gpus,
        comet_model_name=comet_model_name,
    )

    # Evaluate
    results = evaluate_comet(valid_rows, scores)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Model: {model}")
    print(f"Language: {tgt_lang}")
    print(f"Instructions: {'yes' if use_user_instructions else 'no'}")
    print(f"COMET model: {comet_model_name}")
    print(f"{'=' * 50}")
    print(f"Overall COMET: {results['overall']:.4f} (higher is better)")
    print(f"\nPer-domain scores:")
    for domain, stats in results['by_domain'].items():
        print(f"  {domain}: {stats['comet']:.4f} (n={stats['count']})")

    # Save per-row results
    savepath = get_savepath(model, tgt_lang, split, use_user_instructions, comet_model_name, paragraph_level)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(savepath, "w", encoding="utf-8") as f:
        for row, score in zip(valid_rows, scores):
            result = {
                "uniq_id": row.uniq_id,
                "domain": row.domain,
                "comet_score": score,
            }
            f.write(json.dumps(result) + "\n")
    
    print(f"Saved per-row scores to {savepath}")
    
    return {
        "model": model,
        "tgt_lang": tgt_lang,
        "use_instructions": use_user_instructions,
        "overall": results["overall"],
    }


def main():
    args = parse_args()

    # Parse comma-separated values
    models = [m.strip() for m in args.model.split(",")]
    tgt_langs = [l.strip() for l in args.tgt_lang.split(",")]
    
    print(f"Models: {models}")
    print(f"Target languages: {tgt_langs}")
    
    # Load COMET model once
    comet_model = load_comet_model(args.comet_model)
    
    # Track all results for summary
    all_results = []
    
    # Iterate over all combinations
    for model in models:
        for tgt_lang in tgt_langs:
            result = evaluate_single(
                model=model,
                tgt_lang=tgt_lang,
                split=args.split,
                use_user_instructions=args.use_user_instructions,
                comet_model=comet_model,
                comet_model_name=args.comet_model,
                batch_size=args.batch_size,
                gpus=args.gpus,
                paragraph_level=args.paragraph_level,
            )
            if result:
                all_results.append(result)
    
    # Print final summary if multiple combinations
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Model':<30} {'Language':<12} {'COMET':>10}")
        print("-" * 60)
        for r in all_results:
            model_short = r['model'].split('/')[-1]
            print(f"{model_short:<30} {r['tgt_lang']:<12} {r['overall']:>10.4f}")


if __name__ == "__main__":
    main()
