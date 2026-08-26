"""
Win rate analysis: compare translations with vs without instructions.

For each row, use Gemini to judge which translation is better:
- T1: translation with instructions (randomized position)
- T2: translation without instructions (randomized position)

Output: A table with one row per model showing win rate of "with instructions".

Usage:
    python -m judge.win_rate --tgt-lang ind_Latn
"""
from dotenv import load_dotenv
load_dotenv()

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Literal

from pydantic import BaseModel
from tqdm import tqdm

from translate.bouquet_vllm import BouquetInstructionRow
from common.bouquet_utils import get_lang_name
from common.trans_utils import cached_completion


# Configuration
BASEPATH = Path("translation_results/win_rate/")
EVAL_MODEL = "gemini/gemini-3-flash-preview"
MAX_WORKERS = 8
MODELS = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]
DEFAULT_LANG = "fra_Latn"
DEFAULT_SPLIT = "dev"

# Seed for reproducible randomization
RANDOM_SEED = 42


class PairwiseJudgment(BaseModel):
    """LLM judgment comparing two translations."""
    uniq_id: str
    winner: Literal["T1", "T2", "tie"]
    reasoning: str
    # Track which position had "with instructions" (for de-randomization)
    with_instr_position: Literal["T1", "T2"]


class TranslationPair(BaseModel):
    """A pair of translations for the same source text."""
    uniq_id: str
    src_text: str  # English source
    ref_text: str  # French reference
    user_instruction: str
    trans_with_instr: str  # Translation with instructions
    trans_without_instr: str  # Translation without instructions


SYSTEM_PROMPT_TEMPLATE = """You are an expert translation evaluator. You will compare two translations of the same English text into {lang_name}.

You will be given:
- Source text (English)
- User instruction about desired tone/context
- Translation T1 (into {lang_name})
- Translation T2 (into {lang_name})
- Reference translation ({lang_name} ground truth)

Your task: Determine which translation is better overall, considering:
1. Accuracy: Does it convey the meaning of the source correctly?
2. Fluency: Does it sound natural in {lang_name}?
3. Context adaptation: Does it follow the user's instruction about tone/context?

Respond in JSON format:
{{
    "winner": "T1" or "T2" or "tie",
    "reasoning": "Brief explanation (1-2 sentences)"
}}

Choose "tie" only if both translations are truly equivalent in quality."""


def build_pair_messages(pair: TranslationPair, t1_text: str, t2_text: str, lang_name: str) -> List[dict]:
    """Build messages for pairwise LLM evaluation."""
    user_content = f"""Source (English): {pair.src_text}

User instruction: {pair.user_instruction}

Translation T1: {t1_text}

Translation T2: {t2_text}

Reference translation: {pair.ref_text}

Which translation is better?"""
    
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=lang_name)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def judge_pair(pair: TranslationPair, rng: random.Random, lang_name: str) -> PairwiseJudgment:
    """Judge a single translation pair using the LLM."""
    # Randomize order to avoid position bias
    if rng.random() < 0.5:
        t1_text = pair.trans_with_instr
        t2_text = pair.trans_without_instr
        with_instr_position = "T1"
    else:
        t1_text = pair.trans_without_instr
        t2_text = pair.trans_with_instr
        with_instr_position = "T2"
    
    messages = build_pair_messages(pair, t1_text, t2_text, lang_name)
    
    response_schema = {
        "type": "object",
        "properties": {
            "winner": {
                "type": "string",
                "enum": ["T1", "T2", "tie"],
            },
            "reasoning": {
                "type": "string",
            },
        },
        "required": ["winner", "reasoning"],
    }
    
    response = cached_completion(
        model=EVAL_MODEL,
        messages=messages,
        response_format={"type": "json_object", "response_schema": response_schema}
    )
    
    result = json.loads(response)
    
    return PairwiseJudgment(
        uniq_id=pair.uniq_id,
        winner=result["winner"],
        reasoning=result.get("reasoning", ""),
        with_instr_position=with_instr_position
    )


def judge_pairs_parallel(pairs: List[TranslationPair], lang_name: str, desc: str = "Judging") -> List[PairwiseJudgment]:
    """Judge all pairs in parallel."""
    results = []
    errors = []
    
    # Create per-pair RNGs for reproducible randomization
    base_rng = random.Random(RANDOM_SEED)
    pair_seeds = {pair.uniq_id: base_rng.randint(0, 2**32) for pair in pairs}
    
    def judge_with_rng(pair: TranslationPair) -> PairwiseJudgment:
        rng = random.Random(pair_seeds[pair.uniq_id])
        return judge_pair(pair, rng, lang_name)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(judge_with_rng, pair): pair for pair in pairs}
        for future in tqdm(as_completed(futures), total=len(pairs), desc=desc):
            pair = futures[future]
            try:
                judgment = future.result()
                results.append(judgment)
            except Exception as e:
                errors.append((pair.uniq_id, str(e)))
                print(f"Error judging {pair.uniq_id}: {e}")
    
    if errors:
        print(f"\n{len(errors)} errors occurred during judgment")
    
    return results


def get_winrate_savepath(model: str, lang: str, split: str) -> Path:
    """Get path for saving win rate results."""
    return BASEPATH / f"winrate_bouquet_{lang}_{split}_{model}.jsonl"


def save_judgments(judgments: List[PairwiseJudgment], filepath: Path):
    """Save judgments to JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for j in judgments:
            f.write(j.model_dump_json() + "\n")


def load_judgments(filepath: Path) -> List[PairwiseJudgment]:
    """Load judgments from JSONL."""
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            results.append(PairwiseJudgment.model_validate_json(line))
    return results


def compute_win_rate(judgments: List[PairwiseJudgment]) -> dict:
    """Compute win rate for 'with instructions' translations.
    
    Ties count as 0.5 for each side.
    """
    wins = 0.0
    losses = 0.0
    ties = 0
    
    for j in judgments:
        if j.winner == "tie":
            ties += 1
            wins += 0.5
            losses += 0.5
        elif j.winner == j.with_instr_position:
            # "with instructions" won
            wins += 1
        else:
            # "without instructions" won
            losses += 1
    
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0
    
    return {
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "total": len(judgments),
    }


def load_translation_pairs(model: str, lang: str, split: str) -> List[TranslationPair]:
    """Load and pair translations with and without instructions."""
    model_id = model if "/" in model else f"google/{model}"
    path_with = BouquetInstructionRow.get_savepath(model_id, lang, split, use_instructions=True)
    path_without = BouquetInstructionRow.get_savepath(model_id, lang, split, use_instructions=False)
    
    if not path_with.exists():
        print(f"File not found: {path_with}")
        return []
    if not path_without.exists():
        print(f"File not found: {path_without}")
        return []
    
    rows_with = {r.uniq_id: r for r in BouquetInstructionRow.load(str(path_with))}
    rows_without = {r.uniq_id: r for r in BouquetInstructionRow.load(str(path_without))}
    
    # Match by uniq_id
    common_ids = set(rows_with.keys()) & set(rows_without.keys())
    
    pairs = []
    for uid in common_ids:
        rw = rows_with[uid]
        rwo = rows_without[uid]
        pairs.append(TranslationPair(
            uniq_id=uid,
            src_text=rw.tgt_text,  # English source
            ref_text=rw.src_text,  # French reference
            user_instruction=rw.user_instruction,
            trans_with_instr=rw.pred_text or "",
            trans_without_instr=rwo.pred_text or "",
        ))
    
    return pairs


def run_winrate_evaluation(model: str, lang: str, split: str, ignore_cache: bool = False) -> dict:
    """Run win rate evaluation for a single model."""
    pairs = load_translation_pairs(model, lang, split)
    if not pairs:
        return None
    
    lang_name = get_lang_name(lang)
    savepath = get_winrate_savepath(model, lang, split)
    
    if savepath.exists() and not ignore_cache:
        print(f"Loading cached judgments from {savepath}")
        judgments = load_judgments(savepath)
    else:
        judgments = judge_pairs_parallel(pairs, lang_name, desc=f"{model}")
        save_judgments(judgments, savepath)
        print(f"Saved judgments to {savepath}")
    
    return compute_win_rate(judgments)


def print_summary_table(results: Dict[str, dict], models: List[str] = None):
    """Print a markdown-formatted summary table."""
    if models is None:
        models = MODELS
    print("\n## Win Rate: With Instructions vs Without\n")
    print("| Model | Win Rate (with instr) | Wins | Ties | Losses | Total |")
    print("|-------|----------------------|------|------|--------|-------|")

    for model in models:
        metrics = results.get(model, {})
        if not metrics:
            print(f"| {model} | N/A | N/A | N/A | N/A | N/A |")
            continue
        
        win_rate = metrics["win_rate"]
        wins = metrics["wins"]
        ties = metrics["ties"]
        losses = metrics["losses"]
        total = metrics["total"]
        
        # Display wins/losses as integers if they are whole numbers
        wins_str = f"{wins:.1f}" if wins != int(wins) else str(int(wins))
        losses_str = f"{losses:.1f}" if losses != int(losses) else str(int(losses))
        
        print(f"| {model} | {win_rate:.1%} | {wins_str} | {ties} | {losses_str} | {total} |")


def main():
    """Run win rate evaluation for all models."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore-cache", action="store_true", help="Re-run evaluations ignoring cache")
    parser.add_argument("--tgt-lang", type=str, default=DEFAULT_LANG,
                        help=f"Target language code (default: {DEFAULT_LANG})")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["dev", "test"],
                        help=f"Dataset split (default: {DEFAULT_SPLIT})")
    parser.add_argument("--models", type=str, nargs="+", default=MODELS,
                        help=f"Models to evaluate (default: {' '.join(MODELS)})")
    args = parser.parse_args()
    
    lang_name = get_lang_name(args.tgt_lang)
    print(f"Target language: {args.tgt_lang} ({lang_name})")
    print(f"Split: {args.split}")
    
    results = {}
    
    for model in args.models:
        print(f"\n{'='*50}")
        print(f"Evaluating {model}...")

        metrics = run_winrate_evaluation(model, args.tgt_lang, args.split, args.ignore_cache)
        if metrics:
            results[model] = metrics
            print(f"  Win rate: {metrics['win_rate']:.1%}")
            print(f"  Wins: {metrics['wins']}, Ties: {metrics['ties']}, Losses: {metrics['losses']}")

    print_summary_table(results, args.models)


if __name__ == "__main__":
    main()
