#!/usr/bin/env python3
"""
Run LLM-as-judge evaluation on Label Studio export JSONL rows.

The evaluation uses only the fields shown to annotators:
instruction, source text, and model translation.

Usage:
    python -m human_eval.match_judge \
        --annotations annotation_exports/human/project_9_annotations.jsonl # fra_Latn
    python -m human_eval.match_judge \
        --annotations annotation_exports/human/project_10_annotations.jsonl # ind_Latn
    python -m human_eval.match_judge \
        --annotations annotation_exports/human/project_11_annotations.jsonl # khm_Khmr
    python -m human_eval.match_judge \
        --annotations annotation_exports/human/project_13_annotations.jsonl # ukr_Cyrl
    python -m human_eval.match_judge \
        --annotations annotation_exports/human/project_14_annotations.jsonl # jav_Latn
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from common.bouquet_utils import get_lang_name
from translate.bouquet_vllm import BouquetInstructionRow
from judge.bouquet import (
    DEFAULT_EVAL_MODEL,
    MAX_WORKERS,
    EvaluatedRow,
    TranslationEvaluator,
)


REQUIRED_ROW_FIELDS = ("source_text", "user_instruction", "model_translation")


def load_annotation_rows(path: Path) -> list[dict]:
    """Load Label Studio export JSONL."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_bouquet_row(row: dict) -> BouquetInstructionRow | None:
    """Convert a Label Studio export row into a BouquetInstructionRow."""
    meta = row.get("meta", {})
    uniq_id = meta.get("uniq_id")
    src_lang = meta.get("src_lang")
    tgt_lang = meta.get("tgt_lang", "")

    if not uniq_id or not src_lang:
        return None

    for field in REQUIRED_ROW_FIELDS:
        if not row.get(field):
            return None

    return BouquetInstructionRow(
        uniq_id=uniq_id,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_text=row.get("reference_translation", "") or "",
        tgt_text=row.get("source_text", ""),
        domain=meta.get("domain", "") or "",
        par_comment=meta.get("par_comment", "") or "",
        tags=meta.get("tags", "") or "",
        register=meta.get("register", "") or "",
        user_instruction=row.get("user_instruction", ""),
        pred_text=row.get("model_translation", ""),
    )


def group_rows_by_lang(rows: Iterable[dict]) -> tuple[dict[str, list[BouquetInstructionRow]], int]:
    """Group valid rows by target language (meta.src_lang)."""
    grouped: dict[str, list[BouquetInstructionRow]] = defaultdict(list)
    skipped = 0
    for row in rows:
        meta = row.get("meta", {})
        lang = meta.get("src_lang")
        bouquet_row = to_bouquet_row(row)
        if not (lang and bouquet_row):
            skipped += 1
            continue
        grouped[lang].append(bouquet_row)
    return grouped, skipped


def default_output_path(input_path: Path, eval_model: str, reference_based: bool = False) -> Path:
    eval_slug = EvaluatedRow._get_eval_model_slug(eval_model)
    suffix = "_refbased" if reference_based else ""
    return Path("annotation_exports/llm") / f"{input_path.stem}_{eval_slug}{suffix}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judge evaluation for Label Studio exports")
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to Label Studio exported annotations JSONL",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help=f"LLM-as-judge model (default: {DEFAULT_EVAL_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output JSONL path (default: annotation_exports/llm/<input>_<eval>.jsonl)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Max parallel workers (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--reference-based",
        action="store_true",
        help="Include the reference translation in the judge prompt (reference-based rating).",
    )
    args = parser.parse_args()

    annotation_path = Path(args.annotations)
    rows = load_annotation_rows(annotation_path)
    if not rows:
        raise ValueError(f"No annotation rows found in {annotation_path}")

    grouped, skipped = group_rows_by_lang(rows)
    if not grouped:
        raise ValueError("No rows had sufficient data for evaluation.")

    missing_reference = 0
    if args.reference_based:
        filtered: dict[str, list[BouquetInstructionRow]] = {}
        for lang, lang_rows in grouped.items():
            kept = [r for r in lang_rows if r.src_text]
            missing_reference += len(lang_rows) - len(kept)
            if kept:
                filtered[lang] = kept
        grouped = filtered
        if not grouped:
            raise ValueError("--reference-based set but no rows have a reference translation.")

    evaluated_rows: list[EvaluatedRow] = []
    for lang, lang_rows in grouped.items():
        lang_name = get_lang_name(lang)
        evaluator = TranslationEvaluator(
            lang_name,
            args.eval_model,
            max_workers=args.max_workers,
            reference_based=args.reference_based,
        )
        evaluated_rows.extend(
            evaluator.evaluate_batch(lang_rows, desc=f"Evaluating {lang} ({lang_name})")
        )

    output_path = Path(args.output) if args.output else default_output_path(annotation_path, args.eval_model, args.reference_based)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    EvaluatedRow.save(evaluated_rows, output_path)

    print("\n## LLM Annotation Export\n")
    print(f"Input: {annotation_path}")
    print(f"Output: {output_path}")
    print(f"Rows evaluated: {len(evaluated_rows)}")
    print(f"Rows skipped: {skipped}")
    if args.reference_based:
        print(f"Rows skipped for missing reference: {missing_reference}")


if __name__ == "__main__":
    main()
