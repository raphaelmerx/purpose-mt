#!/usr/bin/env python3
"""
Create a Label Studio project with 160 Bouquet rows for a specific lang.

Usage:
    python -m human_eval.setup_project --tgt-lang ind_Latn
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import random
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from human_eval.label_studio_client import LabelStudioClient
from human_eval.labeling_configs import TRANSLATION_QUALITY_CONFIG
from common.bouquet_utils import get_lang_name
from judge.bouquet import EvaluatedRow, DEFAULT_EVAL_MODEL


SPLIT = "dev"
MODELS = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]
TOTAL_ROWS = 160
RANDOM_SEED = 13
BASEPATH = Path("translation_results")


def evaluated_row_to_task(row: EvaluatedRow, model: str, use_instructions: bool) -> dict:
    """Convert an evaluated row dict to a Label Studio task format."""
    row_data = row.row
    evaluation = row.evaluation
    return {
        "data": {
            "source_text": row_data.tgt_text,  # English source
            "user_instruction": row_data.user_instruction or "(No instruction provided)",
            "model_translation": row_data.pred_text,
            "reference_translation": row_data.src_text,
        },
        "meta": {
            "uniq_id": row_data.uniq_id,
            "model": model,
            "instruction_variant": "with_instructions" if use_instructions else "without_instructions",
            "domain": row_data.domain,
            "src_lang": row_data.src_lang,
            "tgt_lang": row_data.tgt_lang,
            "register": row_data.register,
            "tags": row_data.tags,
            # LLM evaluation stored for later comparison (hidden from annotators)
            "_llm_is_critical_failure": evaluation.is_critical_failure,
            "_llm_translation_rating_score": evaluation.translation_rating_score,
            "_llm_adaptedness_score": evaluation.adaptedness_score,
            "_llm_comment": evaluation.comment,
        },
    }


def load_bucket_tasks(model: str, use_instructions: bool, tgt_lang: str) -> list[dict]:
    """Load and sample one bucket of evaluated rows, returning Label Studio tasks."""
    eval_path = EvaluatedRow.get_savepath(model, tgt_lang, SPLIT, use_instructions, DEFAULT_EVAL_MODEL)

    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {eval_path}")

    rows = EvaluatedRow.load(eval_path)
    rows_per_bucket = TOTAL_ROWS // (len(MODELS) * 2)  # 1 bucket with instr, one without, for each model
    if len(rows) < rows_per_bucket:
        raise ValueError(
            f"Not enough rows in {eval_path}: need {rows_per_bucket}, found {len(rows)}"
        )

    variant = "with_instructions" if use_instructions else "without_instructions"
    rng = random.Random(f"{RANDOM_SEED}-{model}-{variant}")
    sampled_rows = rng.sample(rows, rows_per_bucket)
    return [evaluated_row_to_task(row, model, use_instructions) for row in sampled_rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Label Studio project with 160 Bouquet rows for a target language."
    )
    parser.add_argument(
        "--tgt-lang",
        default="fra_Latn",
        help="Target language code for Bouquet rows (default: fra_Latn).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate task creation without creating or importing a project.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tgt_lang = args.tgt_lang
    lang_name = get_lang_name(tgt_lang)
    project_title = f"Bouquet {lang_name} 160 rows"
    project_description = f"""
Human verification of translation quality for {lang_name} Bouquet rows.
Sampled 160 rows across Gemma 3 outputs with/without instructions.
"""

    base_url = os.environ.get("LABEL_STUDIO_HOST", "http://localhost:8080")

    print(f"🔗 Connecting to Label Studio at {base_url}...")
    print("\n📋 Import configuration:")
    print(f"   Project: {project_title}")
    print(f"   Target language: {tgt_lang}")
    print(f"   Split: {SPLIT}")
    print(f"   Models: {', '.join(MODELS)}")
    print(f"   Total rows: {TOTAL_ROWS}")
    print(f"   Random seed: {RANDOM_SEED}")

    print("\n📦 Loading evaluation data...")
    tasks: list[dict] = []
    for model in MODELS:
        for use_instructions in [True, False]:
            variant = "with" if use_instructions else "without"
            print(f"   - {model} ({variant} instructions)")
            tasks.extend(load_bucket_tasks(model, use_instructions, tgt_lang))

    # shuffle tasks
    random.shuffle(tasks)

    expected_total = TOTAL_ROWS
    if len(tasks) != expected_total:
        raise RuntimeError(f"Expected {expected_total} tasks, got {len(tasks)}")

    if args.dry_run:
        print("\n🧪 Dry run enabled: no project will be created or imported.")
        print(f"   Prepared {len(tasks)} tasks for {project_title}")
        return

    api_key = os.environ.get("LABEL_STUDIO_API_KEY")
    if not api_key:
        raise ValueError("Please set LABEL_STUDIO_API_KEY environment variable")

    client = LabelStudioClient(base_url, api_key)

    existing = client.get_project_by_title(project_title)
    if existing:
        print(f"\n⚠️  Project '{project_title}' already exists (ID: {existing['id']})")
        if input("   Use existing project? [y/N]: ").strip().lower() != "y":
            print("   Aborting.")
            return
        project_id = existing["id"]
    else:
        print(f"\n📁 Creating project: {project_title}")
        project = client.create_project(
            title=project_title,
            label_config=TRANSLATION_QUALITY_CONFIG,
            description=project_description.strip(),
        )
        project_id = project["id"]
        print(f"✅ Project created! ID: {project_id}")

    print(f"\n📤 Importing {len(tasks)} tasks...")
    result = client.import_tasks(project_id, tasks)
    print(f"✅ Imported {result.get('task_count', len(tasks))} tasks")

    print("\n🎉 Setup complete!")
    print(f"   Project URL: {base_url}/projects/{project_id}")


if __name__ == "__main__":
    main()
