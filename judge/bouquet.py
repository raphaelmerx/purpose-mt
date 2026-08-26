"""
Qualitative translation analysis using LLM judge.

For each translation, use gemini-3-flash to evaluate:
1. Translation rating (0-100): Overall translation quality / meaning preserved.
2. Adaptedness score (0-100): Is it adapted to the tone/context requested?

This matches the human annotation setup in Label Studio.

Usage:
    python -m judge.bouquet --tgt-lang ind_Latn
    python -m judge.bouquet --tgt-lang khm_Khmr --eval-model gpt-5.2
    python -m judge.bouquet --tgt-lang fra_Latn --semantic-fewshot
    python -m judge.bouquet --models gemma-3-4b-it gemma-3-12b-it
"""
from dotenv import load_dotenv
load_dotenv()

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set, Tuple
import pycountry

from pydantic import BaseModel
from tqdm import tqdm

from translate.bouquet_vllm import BouquetInstruction, BouquetInstructionRow
from common.bouquet_utils import parse_bool, get_lang_name
from common.trans_utils import cached_completion


# Configuration
BASEPATH = Path("translation_results/qualitative_analysis/")
DEFAULT_EVAL_MODEL = "gemini/gemini-3-flash-preview"
MAX_WORKERS = 8
DEFAULT_MODELS = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]
DEFAULT_LANG = "fra_Latn"
DEFAULT_SPLIT = "dev"

class ErrorSpan(BaseModel):
    """An error span in the translation (matches Label Studio error_labels)."""
    text: str  # The problematic text span from the translation
    label: str  # "Major" (meaning change, mistranslation) or "Minor" (grammar, style)


class EvaluationSummaryMetrics(BaseModel):
    """Summary metrics for a set of translation evaluations."""
    translation_rating_avg: float
    adaptedness_avg: float
    critical_count: int
    critical_pct: float
    total: int

    def __sub__(self, other: "EvaluationSummaryMetrics") -> "EvaluationSummaryMetrics":
        """Compute difference between two metrics (self - other)."""
        return EvaluationSummaryMetrics(
            translation_rating_avg=self.translation_rating_avg - other.translation_rating_avg,
            adaptedness_avg=self.adaptedness_avg - other.adaptedness_avg,
            critical_count=self.critical_count - other.critical_count,
            critical_pct=self.critical_pct - other.critical_pct,
            total=self.total,  # Keep the total from self
        )

    @classmethod
    def empty(cls) -> "EvaluationSummaryMetrics":
        """Return an empty metrics object (all zeros)."""
        return cls(
            translation_rating_avg=0.0,
            adaptedness_avg=0.0,
            critical_count=0,
            critical_pct=0.0,
            total=0,
        )

    @classmethod
    def print_summary_table(
        cls,
        results: dict[tuple[str, bool], "EvaluationSummaryMetrics"],
        lang: str,
        models: List[str],
    ):
        """Print a markdown-formatted summary table comparing with/without instructions."""
        print("\n## Translation Quality Evaluation Results\n")
        print(f"Language: {lang}\n")

        # Header
        print("| Model | Critical trans errors without instructions | Critical trans errors with instructions | Rating without instructions | Rating with instructions | Adaptedness without instructions | Adaptedness with instructions |")
        print("|-------|--------------------------------------------|------------------------------------------|------------------------------|---------------------------|---------------------------------|------------------------------|")

        for model in models:
            model_name = model.split("/")[-1] if "/" in model else model.replace("gemma-3-", "Gemma-3-").replace("b-it", "B")
            metrics_no = results.get((model, False))
            metrics_yes = results.get((model, True))

            def fmt_pct(m: Optional["EvaluationSummaryMetrics"]) -> str:
                return f"{m.critical_pct:.1f}%" if m else "N/A"

            def fmt_score(m: Optional["EvaluationSummaryMetrics"], attr: str) -> str:
                return f"{getattr(m, attr):.1f}" if m else "N/A"

            print(
                "| {model} | {crit_no} | {crit_yes} | {rating_no} | {rating_yes} | "
                "{adapt_no} | {adapt_yes} |".format(
                    model=model_name,
                    crit_no=fmt_pct(metrics_no),
                    crit_yes=fmt_pct(metrics_yes),
                    rating_no=fmt_score(metrics_no, "translation_rating_avg"),
                    rating_yes=fmt_score(metrics_yes, "translation_rating_avg"),
                    adapt_no=fmt_score(metrics_no, "adaptedness_avg"),
                    adapt_yes=fmt_score(metrics_yes, "adaptedness_avg"),
                )
            )

    @classmethod
    def print_domain_summary_table(
        cls,
        domain_metrics_with: dict[str, "EvaluationSummaryMetrics"],
        domain_metrics_without: dict[str, "EvaluationSummaryMetrics"],
        model: str,
    ):
        """Print per-domain metrics showing gain from instructions."""
        model_name = model.replace("gemma-3-", "Gemma-3-").replace("b-it", "B")

        print(f"\n### {model_name} - Per Domain (with - without instructions)\n")
        print("| Domain | Rating Diff | Adaptedness Diff | Critical % Diff | Count |")
        print("|--------|-------------|------------------|-----------------|-------|")

        # Get all domains from both sets
        all_domains = sorted(set(domain_metrics_with.keys()) | set(domain_metrics_without.keys()))
        empty = cls.empty()

        for domain in all_domains:
            m_with = domain_metrics_with.get(domain, empty)
            m_without = domain_metrics_without.get(domain, empty)
            diff = m_with - m_without

            print(
                "| {domain} | {rating} | {adapt} | {crit} | {count} |".format(
                    domain=domain.replace("|", "/"),
                    rating=f"{diff.translation_rating_avg:.1f}",
                    adapt=f"{diff.adaptedness_avg:.1f}",
                    crit=f"{diff.critical_pct:.1f}",
                    count=m_with.total or m_without.total,
                )
            )


class TranslationEvaluation(BaseModel):
    """LLM evaluation of a single translation (matches Label Studio human annotation)."""
    uniq_id: str
    error_spans: List[ErrorSpan]  # Labeled spans (matches error_labels in Label Studio)
    is_critical_failure: bool  # True if output is not a translation at all
    translation_rating_score: int  # 0-100: Overall translation quality
    adaptedness_score: int  # 0-100: Is it adapted to the tone/context requested?
    comment: str  # Brief explanation from LLM (matches annotator_comment field)

    @classmethod
    def from_llm_response(cls, uniq_id: str, response_json: dict) -> "TranslationEvaluation":
        """Parse LLM JSON response into a TranslationEvaluation."""
        # Parse error spans
        error_spans = [
            ErrorSpan(text=span.get("text", ""), label=span.get("label", "Minor"))
            for span in response_json.get("error_spans", [])
        ]

        critical_error = parse_bool(response_json.get("is_critical_failure", False))
        # If a critical error, set rating and adaptedness to 0
        translation_rating_score = 0 if critical_error else max(
            0, min(100, response_json.get("translation_rating_score", 0))
        )
        adaptedness_score = 0 if critical_error else max(
            0, min(100, response_json.get("adaptedness_score", 0))
        )

        return cls(
            uniq_id=uniq_id,
            error_spans=error_spans,
            is_critical_failure=critical_error,
            translation_rating_score=translation_rating_score,
            adaptedness_score=adaptedness_score,
            comment=response_json.get("comment", "")
        )
    
    @classmethod
    def get_response_format(cls, eval_model: str) -> dict:
        """Get the response format for the evaluation model."""
        if eval_model.startswith("gemini/"):
            return {"type": "json_object", "response_schema": cls.model_json_schema()}
        else:
            return cls.model_json_schema()


class EvaluatedRow(BaseModel):
    """A translation row with its evaluation."""
    row: BouquetInstructionRow
    evaluation: TranslationEvaluation

    @staticmethod
    def _get_eval_model_slug(eval_model: str) -> str:
        """Convert an eval model name to a filename-safe slug."""
        return eval_model.replace("/", "-").replace(":", "-")

    @classmethod
    def get_savepath(
        cls,
        model: str,
        lang: str,
        split: str,
        use_instructions: bool,
        eval_model: str,
        semantic_fewshot: bool = False,
        paragraph_level: bool = False,
        numbered_lists: bool = False,
        instruction_type: str = "full",
        enable_thinking: bool = False,
        reference_based: bool = False,
        judge_inputs: str = "joint",
    ) -> Path:
        """Get path for saving evaluation results."""
        suffix = "with_instructions" if use_instructions else "without_instructions"
        if semantic_fewshot:
            suffix += "_semantic"
        if paragraph_level:
            suffix += "_paragraph"
        if numbered_lists:
            suffix += "_numbered"
        if instruction_type != "full":
            suffix += f"_{instruction_type}"
        if enable_thinking:
            suffix += "_thinking"
        if reference_based:
            suffix += "_refbased"
        if judge_inputs != "joint":
            suffix += f"_{judge_inputs.replace('-', '')}"
        eval_model_slug = cls._get_eval_model_slug(eval_model)
        return BASEPATH / f"eval_bouquet_{lang}_{split}_{model}_{suffix}_{eval_model_slug}.jsonl"

    @classmethod
    def save(cls, rows: List["EvaluatedRow"], filepath: Path):
        """Save evaluated rows to JSONL."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for er in rows:
                f.write(er.model_dump_json() + "\n")

    @classmethod
    def load(cls, filepath: Path) -> List["EvaluatedRow"]:
        """Load evaluated rows from JSONL."""
        results = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                results.append(cls.model_validate_json(line))
        return results

    @classmethod
    def compute_metrics(cls, rows: List["EvaluatedRow"]) -> EvaluationSummaryMetrics:
        """Compute summary metrics from evaluated rows."""
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
    def compute_metrics_by_domain(cls, rows: List["EvaluatedRow"]) -> dict[str, EvaluationSummaryMetrics]:
        """Compute summary metrics grouped by domain."""
        # Group rows by domain
        by_domain: dict[str, List["EvaluatedRow"]] = {}
        for er in rows:
            domain = er.row.domain
            by_domain.setdefault(domain, []).append(er)

        # Compute metrics per domain
        return {domain: cls.compute_metrics(domain_rows) for domain, domain_rows in sorted(by_domain.items())}


SYSTEM_PROMPT_TEMPLATE = """You are an expert translation evaluator. You will be given:
- Instruction: the desired tone/context for the translation
- Source sentence (English){reference_line}
- Translation to evaluate (into {lang_name})

Your task has three parts:

## Part 1: Identify Error Spans
Identify specific text spans in the translation that contain errors. For each error, provide:
- "text": the exact text from the translation (copy it exactly)
- "label": either "Major" or "Minor"
  - **Major**: meaning change, mistranslation, wrong language, or omission of important content
  - **Minor**: grammar issues, style problems, awkward phrasing, or minor typos

Return an empty list if there are no errors.

## Part 2: Critical Failure Flag
Set "is_critical_failure" to true if the output is not a translation at all (not in {lang_name}, unrelated content, or not translating the source). Otherwise, false.

## Part 3: Rate the Translation (0-100 scale)

1. TRANSLATION RATING (0-100): Overall translation quality.
   - 0: No meaning preserved at all
   - 33: Some meaning preserved, but significant errors or omissions
   - 66: Most meaning preserved, minor issues
   - 100: Perfect

2. ADAPTEDNESS (0-100): Is it adapted to the tone/context requested in the Instruction?
   - 0: Not at all adapted (e.g., informal when formal was requested)
   - 33: Minimally adapted
   - 66: Mostly adapted
   - 100: Perfectly adapted to the requested tone/context

If the translation is not valid at all (wrong language or not a translation of the source), give 0 for all criteria.
"""


# Judge-input ablations (reviewer request): decouple the evaluation criterion from what the
# judge is shown. In `source-only` the judge never sees the instruction and rates meaning
# preservation alone; in `instruction-only` it never sees the source and rates adaptedness alone.
# Both reuse the TranslationEvaluation schema, leaving the criterion that was not requested at 0.
SOURCE_ONLY_PROMPT_TEMPLATE = """You are an expert translation evaluator. You will be given:
- Source sentence (English)
- Translation to evaluate (into {lang_name})

Your task has three parts:

## Part 1: Identify Error Spans
Identify specific text spans in the translation that contain errors. For each error, provide:
- "text": the exact text from the translation (copy it exactly)
- "label": either "Major" or "Minor"
  - **Major**: meaning change, mistranslation, wrong language, or omission of important content
  - **Minor**: grammar issues, style problems, awkward phrasing, or minor typos

Return an empty list if there are no errors.

## Part 2: Critical Failure Flag
Set "is_critical_failure" to true if the output is not a translation at all (not in {lang_name}, unrelated content, or not translating the source). Otherwise, false.

## Part 3: Rate the Translation (0-100 scale)

1. TRANSLATION RATING (0-100): Overall translation quality.
   - 0: No meaning preserved at all
   - 33: Some meaning preserved, but significant errors or omissions
   - 66: Most meaning preserved, minor issues
   - 100: Perfect

Judge meaning preservation relative to the source only. You are not told the intended tone, audience, or purpose, so do not reward or penalise the translation for its register.

Always set "adaptedness_score" to 0; it is not evaluated in this mode.

If the translation is not valid at all (wrong language or not a translation of the source), give 0 for all criteria.
"""

INSTRUCTION_ONLY_PROMPT_TEMPLATE = """You are an expert translation evaluator. You will be given:
- Instruction: the desired tone/context for the translation
- Text to evaluate (in {lang_name})

You are deliberately not shown the source sentence. Judge only whether the text suits the requested tone, audience, and purpose, not whether it is an accurate translation.

Your task has three parts:

## Part 1: Identify Error Spans
Identify specific text spans that clash with the requested tone or context. For each, provide:
- "text": the exact text (copy it exactly)
- "label": either "Major" or "Minor"
  - **Major**: register or tone plainly at odds with the instruction
  - **Minor**: wording that is merely awkward for the requested context

Return an empty list if there are no such spans.

## Part 2: Critical Failure Flag
Set "is_critical_failure" to true if the text is not in {lang_name} at all. Otherwise, false.

## Part 3: Rate the Adaptation (0-100 scale)

1. ADAPTEDNESS (0-100): Is it adapted to the tone/context requested in the Instruction?
   - 0: Not at all adapted (e.g., informal when formal was requested)
   - 33: Minimally adapted
   - 66: Mostly adapted
   - 100: Perfectly adapted to the requested tone/context

Always set "translation_rating_score" to 0; it is not evaluated in this mode.
"""

JUDGE_INPUTS_CHOICES = ("joint", "source-only", "instruction-only")


class TranslationEvaluator:
    """Runs LLM-based translation evaluation."""

    def __init__(
        self,
        lang_name: str,
        eval_model: str,
        max_workers: int = MAX_WORKERS,
        reference_based: bool = False,
        judge_inputs: str = "joint",
    ):
        if judge_inputs not in JUDGE_INPUTS_CHOICES:
            raise ValueError(f"judge_inputs must be one of {JUDGE_INPUTS_CHOICES}, got {judge_inputs!r}")
        if judge_inputs != "joint" and reference_based:
            raise ValueError("reference_based is only supported with judge_inputs='joint'")
        self.lang_name = lang_name
        self.eval_model = eval_model
        self.max_workers = max_workers
        self.reference_based = reference_based
        self.judge_inputs = judge_inputs

    def _build_messages(self, row: BouquetInstructionRow) -> List[dict]:
        """Build LLM prompt messages for evaluation."""
        if self.judge_inputs == "source-only":
            # No instruction shown: meaning preservation only.
            user_content = f"""Source sentence (English): {row.tgt_text}

{self.lang_name} Translation to evaluate: {row.pred_text}

Evaluate this translation."""
            system_prompt = SOURCE_ONLY_PROMPT_TEMPLATE.format(lang_name=self.lang_name)
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

        if self.judge_inputs == "instruction-only":
            # No source shown: adaptedness only.
            user_content = f"""Instruction: {row.user_instruction}

{self.lang_name} text to evaluate: {row.pred_text}

Evaluate how well this text suits the instruction."""
            system_prompt = INSTRUCTION_ONLY_PROMPT_TEMPLATE.format(lang_name=self.lang_name)
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

        reference_section = ""
        reference_line = ""
        if self.reference_based:
            if not row.src_text:
                raise ValueError(
                    f"reference_based=True but row {row.uniq_id} has no reference (src_text is empty)"
                )
            reference_section = f"\n\nReference translation ({self.lang_name}): {row.src_text}"
            reference_line = f"\n- Reference translation (in {self.lang_name})"

        user_content = f"""Instruction: {row.user_instruction}

Source sentence (English): {row.tgt_text}{reference_section}

{self.lang_name} Translation to evaluate: {row.pred_text}

Evaluate this translation."""

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            lang_name=self.lang_name, reference_line=reference_line
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    def evaluate(self, row: BouquetInstructionRow) -> TranslationEvaluation:
        """Evaluate a single translation using the LLM."""
        messages = self._build_messages(row)

        response = cached_completion(
            model=self.eval_model,
            messages=messages,
            response_format=TranslationEvaluation.get_response_format(self.eval_model),
        )

        result = json.loads(response)
        return TranslationEvaluation.from_llm_response(row.uniq_id, result)

    def evaluate_batch(
        self,
        rows: List[BouquetInstructionRow],
        desc: str = "Evaluating",
    ) -> List[EvaluatedRow]:
        """Evaluate all rows in parallel."""
        results = []
        errors = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.evaluate, row): row for row in rows}
            for future in tqdm(as_completed(futures), total=len(rows), desc=desc):
                row = futures[future]
                try:
                    evaluation = future.result()
                    results.append(EvaluatedRow(row=row, evaluation=evaluation))
                except Exception as e:
                    errors.append((row.uniq_id, str(e)))
                    print(f"Error evaluating {row.uniq_id}: {e}")

        if errors:
            print(f"\n{len(errors)} errors occurred during evaluation")

        return results


def run_evaluation(
    model: str,
    lang: str,
    split: str,
    use_instructions: bool,
    eval_model: str,
    semantic_fewshot: bool = False,
    paragraph_level: bool = False,
    numbered_lists: bool = False,
    instruction_type: str = "full",
    enable_thinking: bool = False,
    reference_based: bool = False,
    judge_inputs: str = "joint",
) -> Optional[EvaluationSummaryMetrics]:
    """Run evaluation for a single model/condition combination."""
    # Load translation results
    model_id = model if "/" in model else f"google/{model}"
    trans_path = BouquetInstructionRow.get_savepath(model_id, lang, split, use_instructions, semantic_fewshot, paragraph_level, numbered_lists, instruction_type, enable_thinking)
    if not trans_path.exists():
        print(f"Translation file not found: {trans_path}")
        return None

    rows = BouquetInstructionRow.load(trans_path)

    # For ablation variants, the judge always sees the FULL instruction
    if instruction_type != "full":
        full_instructions = BouquetInstruction.load(split, "full")
        for row in rows:
            if row.uniq_id in full_instructions:
                row.user_instruction = full_instructions[row.uniq_id].user_instruction

    lang_name = get_lang_name(lang)
    eval_path = EvaluatedRow.get_savepath(
        model, lang, split, use_instructions, eval_model,
        semantic_fewshot, paragraph_level, numbered_lists, instruction_type, enable_thinking,
        reference_based=reference_based, judge_inputs=judge_inputs,
    )

    suffix = "with" if use_instructions else "without"
    if instruction_type != "full":
        suffix += f" {instruction_type}"
    if reference_based:
        suffix += ", ref-based"
    if judge_inputs != "joint":
        suffix += f", {judge_inputs}"
    evaluator = TranslationEvaluator(
        lang_name, eval_model, reference_based=reference_based, judge_inputs=judge_inputs
    )
    evaluated = evaluator.evaluate_batch(rows, desc=f"{model} ({suffix} instr)")

    EvaluatedRow.save(evaluated, eval_path)
    print(f"Saved evaluations to {eval_path}")

    return EvaluatedRow.compute_metrics(evaluated)


def main():
    """Run evaluation for all models and conditions."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt-lang", type=str, default=DEFAULT_LANG,
                        help=f"Target language code (default: {DEFAULT_LANG})")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["dev", "test"],
                        help=f"Dataset split (default: {DEFAULT_SPLIT})")
    parser.add_argument("--eval-model", type=str, default=DEFAULT_EVAL_MODEL,
                        help=f"Evaluation model (default: {DEFAULT_EVAL_MODEL})")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to evaluate (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--print-per-domain",
        action="store_true",
        help="Print per-domain summary tables for each model/instruction variant",
    )
    parser.add_argument(
        "--semantic-fewshot",
        action="store_true",
        help="Only retrieve rows translated with --semantic-fewshot in translate/bouquet_vllm.py",
    )
    parser.add_argument(
        "--paragraph-level",
        action="store_true",
        help="Evaluate paragraph-level translations (without instructions only)",
    )
    parser.add_argument(
        "--numbered-lists",
        action="store_true",
        help="Evaluate numbered-list paragraph translations (requires --paragraph-level)",
    )
    parser.add_argument(
        "--instruction-type",
        type=str,
        default="full",
        help="Instruction variant; matches the slug used in translate/bouquet_vllm.py "
             "(default: full). Judge always uses full instruction.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Evaluate translations produced with --enable-thinking in translate/bouquet_vllm.py",
    )
    parser.add_argument(
        "--reference-based",
        action="store_true",
        help="Include the reference translation in the judge prompt (reference-based rating).",
    )
    parser.add_argument(
        "--judge-inputs",
        type=str,
        default="joint",
        choices=list(JUDGE_INPUTS_CHOICES),
        help="What the judge sees. 'joint' (default) shows source + instruction and scores both "
             "criteria. 'source-only' hides the instruction and scores translation rating only. "
             "'instruction-only' hides the source and scores adaptedness only.",
    )
    args = parser.parse_args()

    lang_name = get_lang_name(args.tgt_lang)
    print(f"Target language: {args.tgt_lang} ({lang_name})")
    print(f"Split: {args.split}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Semantic few-shot: {args.semantic_fewshot}")
    print(f"Paragraph level: {args.paragraph_level}")
    print(f"Numbered lists: {args.numbered_lists}")
    print(f"Instruction type: {args.instruction_type}")
    print(f"Enable thinking: {args.enable_thinking}")
    print(f"Reference based: {args.reference_based}")
    print(f"Judge inputs: {args.judge_inputs}")

    # Paragraph-level translations have no instructions variant
    # Ablation variants (context/purpose) only run with instructions
    if args.paragraph_level:
        instruction_variants = [False]
    elif args.instruction_type != "full":
        instruction_variants = [True]
    else:
        instruction_variants = [True, False]

    results = {}
    for model in args.models:
        for use_instructions in instruction_variants:
            print(f"\n{'='*50}")
            suffix = "with" if use_instructions else "without"
            if args.instruction_type != "full":
                suffix += f" {args.instruction_type}"
            print(f"Evaluating {model} ({suffix} instructions)...")

            metrics = run_evaluation(
                model,
                args.tgt_lang,
                args.split,
                use_instructions,
                args.eval_model,
                semantic_fewshot=args.semantic_fewshot,
                paragraph_level=args.paragraph_level,
                numbered_lists=args.numbered_lists,
                instruction_type=args.instruction_type,
                enable_thinking=args.enable_thinking,
                reference_based=args.reference_based,
                judge_inputs=args.judge_inputs,
            )
            if metrics:
                results[(model, use_instructions)] = metrics
                print(f"  Critical percentage: {metrics.critical_pct:.1f}%")
                print(f"  Avg translation rating: {metrics.translation_rating_avg:.1f}/100")
                print(f"  Avg adaptedness: {metrics.adaptedness_avg:.1f}/100")

    EvaluationSummaryMetrics.print_summary_table(results, args.tgt_lang, args.models)

    # Print per-domain tables if requested
    if args.print_per_domain:
        print("\n## Per-Domain Breakdown\n")
        for model in args.models:
            eval_path_with = EvaluatedRow.get_savepath(
                model, args.tgt_lang, args.split, True, args.eval_model, args.semantic_fewshot, args.paragraph_level, args.numbered_lists, enable_thinking=args.enable_thinking, reference_based=args.reference_based, judge_inputs=args.judge_inputs
            )
            eval_path_without = EvaluatedRow.get_savepath(
                model, args.tgt_lang, args.split, False, args.eval_model, args.semantic_fewshot, args.paragraph_level, args.numbered_lists, enable_thinking=args.enable_thinking, reference_based=args.reference_based, judge_inputs=args.judge_inputs
            )
            if eval_path_with.exists() and eval_path_without.exists():
                rows_with = EvaluatedRow.load(eval_path_with)
                rows_without = EvaluatedRow.load(eval_path_without)
                domain_metrics_with = EvaluatedRow.compute_metrics_by_domain(rows_with)
                domain_metrics_without = EvaluatedRow.compute_metrics_by_domain(rows_without)
                EvaluationSummaryMetrics.print_domain_summary_table(
                    domain_metrics_with, domain_metrics_without, model
                )


if __name__ == "__main__":
    main()