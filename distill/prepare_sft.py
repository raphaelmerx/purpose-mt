"""Build SFT train/eval JSONL from teacher translations.

chosen = 27B first candidate becomes the SFT completion target.

Usage:
    python -m distill.prepare_sft --lang ukr_Cyrl
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from distill.translate_teacher import TranslationCandidateRow


DEFAULT_LANG = "ukr_Cyrl"


def build_prompt_messages(row: TranslationCandidateRow) -> List[Dict[str, str]]:
    """Conversational format with plain-string content (text-only model)."""
    return [
        {
            "role": "user",
            "content": f"Instruction: {row.user_instruction}\n\nText: {row.en_paragraph}",
        }
    ]


def build_completion_messages(text: str) -> List[Dict[str, str]]:
    return [{"role": "assistant", "content": text}]
DEFAULT_TEACHER = "google/gemma-3-27b-it"
DEFAULT_STUDENT = "google/gemma-3-4b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT dataset from teacher translations.")
    parser.add_argument("--lang", type=str, default=DEFAULT_LANG)
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--student-model", type=str, default=DEFAULT_STUDENT)
    parser.add_argument(
        "--candidates-dir", type=Path, default=None,
        help="Default: datasets/smol/<lang>_candidates",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    candidates_dir = args.candidates_dir or Path(f"datasets/smol/{args.lang}_candidates")
    teacher_path = TranslationCandidateRow.get_output_path(candidates_dir, args.teacher_model)

    student_slug = args.student_model.split("/")[-1]
    output_train = candidates_dir / f"sft_teacher_train_{student_slug}.jsonl"
    output_eval = candidates_dir / f"sft_teacher_eval_{student_slug}.jsonl"

    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher candidate file missing: {teacher_path}")
    for p in [output_train, output_eval]:
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {p}. Use --overwrite.")

    print(f"Teacher: {teacher_path}")
    teacher_rows = TranslationCandidateRow.load(teacher_path)
    print(f"Teacher rows: {len(teacher_rows)}")

    records: List[Dict[str, Any]] = []
    for row in teacher_rows:
        if not row.candidates:
            continue
        completion_text = row.candidates[0]
        record = {
            "key": row.key,
            "messages": build_prompt_messages(row) + build_completion_messages(completion_text),
        }
        records.append(record)

    print(f"Built {len(records)} SFT records.")

    rng = random.Random(args.seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * args.eval_ratio))
    eval_rows = shuffled[:n_eval]
    train_rows = shuffled[n_eval:]

    save_jsonl(train_rows, output_train)
    save_jsonl(eval_rows, output_eval)
    print(f"Saved {len(train_rows)} train records to {output_train}")
    print(f"Saved {len(eval_rows)} eval records to {output_eval}")


if __name__ == "__main__":
    main()
