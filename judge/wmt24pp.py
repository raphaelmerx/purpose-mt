"""
LLM judge evaluation for WMT24++ translations.

Reuses the evaluation framework from judge/bouquet.py
(SYSTEM_PROMPT_TEMPLATE, TranslationEvaluation, EvaluationSummaryMetrics).

Usage:
    python -m judge.wmt24pp --lang en-fr_FR --models gemma-3-4b-it gemma-3-12b-it gemma-3-27b-it
    python -m judge.wmt24pp --lang en-fr_FR --models gemma-3-27b-it --print-per-domain
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pycountry
from pydantic import BaseModel
from tqdm import tqdm

from common.wmt24pp_utils import Wmt24ppInstruction, Wmt24ppInstructionRow
from judge.bouquet import (
    SYSTEM_PROMPT_TEMPLATE,
    TranslationEvaluation,
    EvaluationSummaryMetrics,
)
from common.trans_utils import cached_completion


BASEPATH = Path("translation_results/wmt24pp_qualitative/")
DEFAULT_EVAL_MODEL = "gemini/gemini-3-flash-preview"
MAX_WORKERS = 8
DEFAULT_MODELS = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]


def get_lang_name_from_wmt(lang: str) -> str:
    """Get language name from WMT24++ locale (e.g. 'en-fr_FR' -> 'French')."""
    tgt_locale = lang.split("-", 1)[1]  # "fr_FR"
    lang_code = tgt_locale.split("_")[0]  # "fr"
    py_lang = pycountry.languages.get(alpha_2=lang_code)
    return py_lang.name if py_lang else tgt_locale


class Wmt24ppEvaluatedRow(BaseModel):
    """A WMT24++ translation row with its evaluation."""
    row: Wmt24ppInstructionRow
    evaluation: TranslationEvaluation

    @classmethod
    def get_savepath(
        cls, model: str, lang: str, use_instructions: bool, eval_model: str,
        enable_thinking: bool = False, paragraph_level: bool = False,
        numbered_lists: bool = False, instruction_type: str = "full",
    ) -> Path:
        suffix = "with_instructions" if use_instructions else "without_instructions"
        if instruction_type != "full":
            suffix += f"_{instruction_type}"
        if paragraph_level:
            suffix += "_paragraph"
        if numbered_lists:
            suffix += "_numbered"
        if enable_thinking:
            suffix += "_thinking"
        eval_slug = eval_model.replace("/", "-").replace(":", "-")
        return BASEPATH / f"eval_wmt24pp_{lang}_{model}_{suffix}_{eval_slug}.jsonl"

    @classmethod
    def save(cls, rows: List["Wmt24ppEvaluatedRow"], filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for er in rows:
                f.write(er.model_dump_json() + "\n")

    @classmethod
    def load(cls, filepath: Path) -> List["Wmt24ppEvaluatedRow"]:
        results = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                results.append(cls.model_validate_json(line))
        return results

    @classmethod
    def compute_metrics(cls, rows: List["Wmt24ppEvaluatedRow"]) -> EvaluationSummaryMetrics:
        if not rows:
            return EvaluationSummaryMetrics.empty()
        rating_scores = [er.evaluation.translation_rating_score for er in rows]
        adaptedness_scores = [er.evaluation.adaptedness_score for er in rows]
        critical_count = sum(1 for er in rows if er.evaluation.is_critical_failure)
        return EvaluationSummaryMetrics(
            translation_rating_avg=sum(rating_scores) / len(rating_scores),
            adaptedness_avg=sum(adaptedness_scores) / len(adaptedness_scores),
            critical_count=critical_count,
            critical_pct=(critical_count / len(rows) * 100),
            total=len(rows),
        )

    @classmethod
    def compute_metrics_by_domain(cls, rows: List["Wmt24ppEvaluatedRow"]) -> dict[str, EvaluationSummaryMetrics]:
        by_domain: dict[str, List["Wmt24ppEvaluatedRow"]] = {}
        for er in rows:
            by_domain.setdefault(er.row.domain, []).append(er)
        return {domain: cls.compute_metrics(domain_rows) for domain, domain_rows in sorted(by_domain.items())}


class Wmt24ppTranslationEvaluator:
    """LLM-based translation evaluator for WMT24++ rows."""

    def __init__(self, lang_name: str, eval_model: str, max_workers: int = MAX_WORKERS):
        self.lang_name = lang_name
        self.eval_model = eval_model
        self.max_workers = max_workers

    def _build_messages(self, row: Wmt24ppInstructionRow) -> List[dict]:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=self.lang_name, reference_line="")
        user_content = f"""Instruction: {row.user_instruction}

Source sentence (English): {row.source}

{self.lang_name} Translation to evaluate: {row.pred_text}

Evaluate this translation."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def evaluate(self, row: Wmt24ppInstructionRow) -> TranslationEvaluation:
        messages = self._build_messages(row)
        response = cached_completion(
            model=self.eval_model,
            messages=messages,
            response_format=TranslationEvaluation.get_response_format(self.eval_model),
        )
        result = json.loads(response)
        return TranslationEvaluation.from_llm_response(row.segment_key, result)

    def evaluate_batch(
        self, rows: List[Wmt24ppInstructionRow], desc: str = "Evaluating",
    ) -> List[Wmt24ppEvaluatedRow]:
        results = []
        errors = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.evaluate, row): row for row in rows}
            for future in tqdm(as_completed(futures), total=len(rows), desc=desc):
                row = futures[future]
                try:
                    evaluation = future.result()
                    results.append(Wmt24ppEvaluatedRow(row=row, evaluation=evaluation))
                except Exception as e:
                    errors.append((row.segment_key, str(e)))
                    print(f"Error evaluating {row.segment_key}: {e}")

        if errors:
            print(f"\n{len(errors)} errors occurred during evaluation")

        return results


def run_evaluation(
    model: str, lang: str, use_instructions: bool, eval_model: str,
    enable_thinking: bool = False, paragraph_level: bool = False,
    numbered_lists: bool = False, instruction_type: str = "full",
) -> Optional[EvaluationSummaryMetrics]:
    """Run evaluation for a single model/condition combination."""
    trans_path = Wmt24ppInstructionRow.get_savepath(
        f"google/{model}", lang, use_instructions,
        enable_thinking=enable_thinking, paragraph_level=paragraph_level,
        numbered_lists=numbered_lists, instruction_type=instruction_type,
    )
    if not trans_path.exists():
        print(f"Translation file not found: {trans_path}")
        return None

    rows = Wmt24ppInstructionRow.load(trans_path)

    # For non-curated instruction variants (e.g. self-para), the translation was
    # produced with the variant instruction but the judge always scores
    # adaptedness against the FULL curated instruction, so the % gap closure vs.
    # the curated condition is apples-to-apples.
    if instruction_type != "full":
        curated = Wmt24ppInstruction.load("full")
        for row in rows:
            if row.segment_key in curated:
                row.user_instruction = curated[row.segment_key].user_instruction

    lang_name = get_lang_name_from_wmt(lang)
    eval_path = Wmt24ppEvaluatedRow.get_savepath(
        model, lang, use_instructions, eval_model,
        enable_thinking=enable_thinking, paragraph_level=paragraph_level,
        numbered_lists=numbered_lists, instruction_type=instruction_type,
    )

    suffix = "with" if use_instructions else "without"
    evaluator = Wmt24ppTranslationEvaluator(lang_name, eval_model)
    evaluated = evaluator.evaluate_batch(rows, desc=f"{model} ({suffix} instr)")

    Wmt24ppEvaluatedRow.save(evaluated, eval_path)
    print(f"Saved evaluations to {eval_path}")

    return Wmt24ppEvaluatedRow.compute_metrics(evaluated)


def main():
    parser = argparse.ArgumentParser(description="WMT24++ LLM judge evaluation")
    parser.add_argument("--lang", type=str, default="en-fr_FR",
                        help="WMT24++ language pair (default: en-fr_FR)")
    parser.add_argument("--eval-model", type=str, default=DEFAULT_EVAL_MODEL,
                        help=f"Evaluation model (default: {DEFAULT_EVAL_MODEL})")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                        help=f"Translation models to evaluate (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--print-per-domain", action="store_true",
                        help="Print per-domain summary tables")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Evaluate translations produced with --enable-thinking")
    parser.add_argument("--paragraph-level", action="store_true",
                        help="Evaluate translations produced with --paragraph-level")
    parser.add_argument("--numbered-lists", action="store_true",
                        help="Evaluate translations produced with --numbered-lists (P2)")
    parser.add_argument("--instruction-type", type=str, default="full",
                        help="Instruction variant to evaluate: 'full' (curated) or "
                             "e.g. 'self-para'. Non-full variants are judged against "
                             "the curated instruction.")
    args = parser.parse_args()

    lang_name = get_lang_name_from_wmt(args.lang)
    print(f"Language: {args.lang} ({lang_name})")
    print(f"Models: {', '.join(args.models)}")
    print(f"Eval model: {args.eval_model}")
    print(f"Enable thinking: {args.enable_thinking}")

    # Non-curated instruction variants only exist with instructions on.
    instr_options = [True] if args.instruction_type != "full" else [True, False]

    results = {}
    for model in args.models:
        for use_instructions in instr_options:
            print(f"\n{'='*50}")
            suffix = "with" if use_instructions else "without"
            print(f"Evaluating {model} ({suffix} instructions)...")

            metrics = run_evaluation(
                model, args.lang, use_instructions, args.eval_model,
                enable_thinking=args.enable_thinking, paragraph_level=args.paragraph_level,
                numbered_lists=args.numbered_lists, instruction_type=args.instruction_type,
            )
            if metrics:
                results[(model, use_instructions)] = metrics
                print(f"  Critical percentage: {metrics.critical_pct:.1f}%")
                print(f"  Avg translation rating: {metrics.translation_rating_avg:.1f}/100")
                print(f"  Avg adaptedness: {metrics.adaptedness_avg:.1f}/100")

    # Print summary table
    EvaluationSummaryMetrics.print_summary_table(results, args.lang, args.models)

    # Print per-domain tables
    if args.print_per_domain:
        print("\n## Per-Domain Breakdown\n")
        for model in args.models:
            eval_path_with = Wmt24ppEvaluatedRow.get_savepath(
                model, args.lang, True, args.eval_model,
                enable_thinking=args.enable_thinking, paragraph_level=args.paragraph_level,
                numbered_lists=args.numbered_lists, instruction_type=args.instruction_type,
            )
            eval_path_without = Wmt24ppEvaluatedRow.get_savepath(
                model, args.lang, False, args.eval_model,
                enable_thinking=args.enable_thinking, paragraph_level=args.paragraph_level,
                numbered_lists=args.numbered_lists, instruction_type=args.instruction_type,
            )
            if eval_path_with.exists() and eval_path_without.exists():
                rows_with = Wmt24ppEvaluatedRow.load(eval_path_with)
                rows_without = Wmt24ppEvaluatedRow.load(eval_path_without)
                domain_metrics_with = Wmt24ppEvaluatedRow.compute_metrics_by_domain(rows_with)
                domain_metrics_without = Wmt24ppEvaluatedRow.compute_metrics_by_domain(rows_without)
                EvaluationSummaryMetrics.print_domain_summary_table(
                    domain_metrics_with, domain_metrics_without, model
                )


if __name__ == "__main__":
    main()
