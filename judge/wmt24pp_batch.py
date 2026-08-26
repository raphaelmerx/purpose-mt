"""Gemini Batch API version of WMT24++ LLM-judge evaluation.

Replaces the per-row cached_completion approach with Gemini Batch API jobs.
Reuses evaluation framework from judge/bouquet.py.

Usage:
    # Submit a batch (dry-run first to verify prompts match)
    python -m judge.wmt24pp_batch --lang en-fr_FR --models gemma-3-27b-it --dry-run
    python -m judge.wmt24pp_batch --lang en-fr_FR --models gemma-3-27b-it --submit

    # Check status of pending batches and download results for completed ones
    python -m judge.wmt24pp_batch --poll

    # Parse downloaded results into evaluation JSONL + print metrics
    python -m judge.wmt24pp_batch --lang en-fr_FR --models gemma-3-27b-it --parse
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pycountry
from pydantic import BaseModel

from google import genai
from google.genai import types
from google.genai.types import BatchJob, JobState

from common.wmt24pp_utils import Wmt24ppInstructionRow
from judge.bouquet import (
    SYSTEM_PROMPT_TEMPLATE,
    TranslationEvaluation,
    EvaluationSummaryMetrics,
)
from judge.wmt24pp import (
    Wmt24ppEvaluatedRow,
    Wmt24ppTranslationEvaluator,
    get_lang_name_from_wmt,
)


BASEPATH = Path("translation_results/wmt24pp_qualitative/")
DEFAULT_EVAL_MODEL = "models/gemini-3-flash-preview"
DEFAULT_MODELS = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]
TRACKER_CSV = BASEPATH / "batch_tracker.csv"

TRACKER_FIELDS = [
    "batch_name", "display_name", "created_at", "model", "lang",
    "use_instructions", "eval_model", "num_requests", "state",
    "updated_at", "request_file", "result_file",
]


# ---------------------------------------------------------------------------
# Batch request building
# ---------------------------------------------------------------------------

def get_response_schema_dict() -> dict:
    """Build a Gemini-compatible JSON schema for TranslationEvaluation.

    Gemini's batch API rejects $defs/$ref and title/description fields.
    We manually build a flat schema that matches the expected output.
    """
    return {
        "type": "object",
        "properties": {
            "uniq_id": {"type": "string"},
            "error_spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "label": {"type": "string"},
                    },
                    "required": ["text", "label"],
                },
            },
            "is_critical_failure": {"type": "boolean"},
            "translation_rating_score": {"type": "integer"},
            "adaptedness_score": {"type": "integer"},
            "comment": {"type": "string"},
        },
        "required": [
            "uniq_id", "error_spans", "is_critical_failure",
            "translation_rating_score", "adaptedness_score", "comment",
        ],
    }


def build_inline_request(
    row: Wmt24ppInstructionRow, lang_name: str, response_schema: dict,
) -> dict:
    """Build a single inline batch request dict for the Gemini Batch API.

    The system prompt and user content are identical to
    Wmt24ppTranslationEvaluator._build_messages().
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=lang_name)
    user_content = (
        f"Instruction: {row.user_instruction}\n\n"
        f"Source sentence (English): {row.source}\n\n"
        f"{lang_name} Translation to evaluate: {row.pred_text}\n\n"
        "Evaluate this translation."
    )

    return {
        "metadata": {"key": row.segment_key},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "config": {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
    }


def get_request_path(lang: str, model: str, use_instructions: bool) -> Path:
    suffix = "with_instructions" if use_instructions else "without_instructions"
    return BASEPATH / f"batch_requests_{lang}_{model}_{suffix}.jsonl"


def get_result_path(lang: str, model: str, use_instructions: bool) -> Path:
    suffix = "with_instructions" if use_instructions else "without_instructions"
    return BASEPATH / f"batch_results_{lang}_{model}_{suffix}.jsonl"


def build_inline_requests(
    rows: List[Wmt24ppInstructionRow], lang_name: str, response_schema: dict,
) -> List[dict]:
    return [build_inline_request(row, lang_name, response_schema) for row in rows]


def save_inline_requests(requests: List[dict], path: Path) -> None:
    """Save inline requests to JSONL for reference/debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"Saved {len(requests)} requests to {path}")


# ---------------------------------------------------------------------------
# Batch tracker (CSV)
# ---------------------------------------------------------------------------

class BatchTrackerEntry(BaseModel):
    batch_name: str
    display_name: str
    created_at: str
    model: str
    lang: str
    use_instructions: bool
    eval_model: str
    num_requests: int
    state: str
    updated_at: str
    request_file: str
    result_file: str


class BatchTracker:
    """CSV-based tracker for batch jobs."""

    def __init__(self, path: Path = TRACKER_CSV):
        self.path = path

    def _ensure_file(self):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TRACKER_FIELDS)
                writer.writeheader()

    def load_all(self) -> List[BatchTrackerEntry]:
        self._ensure_file()
        entries = []
        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["use_instructions"] = row["use_instructions"] == "True"
                row["num_requests"] = int(row["num_requests"])
                entries.append(BatchTrackerEntry(**row))
        return entries

    def append(self, entry: BatchTrackerEntry):
        self._ensure_file()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRACKER_FIELDS)
            writer.writerow(entry.model_dump())

    def rewrite(self, entries: List[BatchTrackerEntry]):
        """Rewrite the entire CSV (used when updating states)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRACKER_FIELDS)
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry.model_dump())

    def find(self, lang: str, model: str, use_instructions: bool) -> Optional[BatchTrackerEntry]:
        """Find the latest tracker entry for a given lang/model/instructions combo."""
        entries = self.load_all()
        matches = [
            e for e in entries
            if e.lang == lang and e.model == model and e.use_instructions == use_instructions
        ]
        return matches[-1] if matches else None



# ---------------------------------------------------------------------------
# Batch API operations
# ---------------------------------------------------------------------------

def create_batch_job(
    client: genai.Client, inline_requests: List[dict], eval_model: str, display_name: str,
) -> BatchJob:
    return client.batches.create(
        model=eval_model, src=inline_requests, config={"display_name": display_name},
    )


def is_completed(job: BatchJob) -> bool:
    return job.state in {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
    }


def save_inline_responses(responses: list, output_path: Path) -> None:
    """Save inline batch responses to JSONL for uniform parsing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for resp in responses:
            # Convert SDK response objects to dicts
            if hasattr(resp, "model_dump"):
                f.write(json.dumps(resp.model_dump(), default=str, ensure_ascii=False) + "\n")
            elif hasattr(resp, "to_dict"):
                f.write(json.dumps(resp.to_dict(), default=str, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(resp, default=str, ensure_ascii=False) + "\n")


def download_batch_results(client: genai.Client, result_file: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_content = client.files.download(file=result_file)
    if isinstance(file_content, (bytes, bytearray)):
        output_path.write_bytes(file_content)
        return
    if hasattr(file_content, "decode"):
        output_path.write_text(file_content.decode("utf-8"), encoding="utf-8")
        return
    output_path.write_text(str(file_content), encoding="utf-8")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def clean_model_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def extract_response_text(payload: dict) -> str:
    response = payload.get("response") or payload.get("inlineResponse")
    if response is None and "candidates" in payload:
        response = payload
    if not response:
        return ""
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    return clean_model_text(text)


def parse_batch_results(
    result_path: Path,
    rows_by_key: Dict[str, Wmt24ppInstructionRow],
) -> Tuple[List[Wmt24ppEvaluatedRow], List[str]]:
    parsed: List[Wmt24ppEvaluatedRow] = []
    errors: List[str] = []

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            key = payload.get("key") or payload.get("metadata", {}).get("key")
            error = payload.get("error")
            if error:
                errors.append(f"{key}: {error}")
                continue
            if not key:
                errors.append("Missing key in batch result row")
                continue

            source_row = rows_by_key.get(key)
            if source_row is None:
                errors.append(f"{key}: missing source row")
                continue

            text = extract_response_text(payload)
            if not text:
                errors.append(f"{key}: empty response text")
                continue

            try:
                result = json.loads(text)
                evaluation = TranslationEvaluation.from_llm_response(key, result)
                parsed.append(Wmt24ppEvaluatedRow(row=source_row, evaluation=evaluation))
            except Exception as e:
                errors.append(f"{key}: parse error: {e}")

    return parsed, errors


# ---------------------------------------------------------------------------
# CLI actions
# ---------------------------------------------------------------------------

def action_dry_run(args):
    """Verify that batch request prompts match the existing evaluator prompts."""
    lang_name = get_lang_name_from_wmt(args.lang)
    response_schema = get_response_schema_dict()
    evaluator = Wmt24ppTranslationEvaluator(lang_name, args.eval_model)

    for model in args.models:
        for use_instructions in [True, False]:
            suffix = "with" if use_instructions else "without"
            trans_path = Wmt24ppInstructionRow.get_savepath(f"google/{model}", args.lang, use_instructions)
            if not trans_path.exists():
                print(f"[skip] {model} ({suffix} instr): {trans_path} not found")
                continue

            rows = Wmt24ppInstructionRow.load(trans_path)
            check_rows = rows[:3]
            print(f"\n{'='*60}")
            print(f"Dry run: {model} ({suffix} instructions), {len(check_rows)} rows")

            for row in check_rows:
                # Batch version
                batch_req = build_inline_request(row, lang_name, response_schema)
                batch_system = batch_req["config"]["system_instruction"]
                batch_user = batch_req["contents"][0]["parts"][0]["text"]

                # Existing version
                messages = evaluator._build_messages(row)
                existing_system = messages[0]["content"]
                existing_user = messages[1]["content"]

                sys_match = batch_system == existing_system
                user_match = batch_user == existing_user
                status = "OK" if (sys_match and user_match) else "MISMATCH"
                print(f"  {row.segment_key}: {status}")
                if not sys_match:
                    print(f"    SYSTEM MISMATCH:")
                    print(f"    batch:    {batch_system[:100]}...")
                    print(f"    existing: {existing_system[:100]}...")
                if not user_match:
                    print(f"    USER MISMATCH:")
                    print(f"    batch:    {batch_user[:200]}...")
                    print(f"    existing: {existing_user[:200]}...")

    print(f"\nResponse schema:\n{json.dumps(response_schema, indent=2)[:500]}...")


def action_submit(args):
    """Build request JSONL, upload, create batch job, log to tracker."""
    lang_name = get_lang_name_from_wmt(args.lang)
    response_schema = get_response_schema_dict()
    tracker = BatchTracker()
    client = genai.Client()

    for model in args.models:
        for use_instructions in [True, False]:
            suffix = "with" if use_instructions else "without"

            # Skip if already tracked
            existing = tracker.find(args.lang, model, use_instructions)
            if existing:
                print(f"[skip] {model} ({suffix} instr): already tracked as {existing.batch_name} ({existing.state})")
                continue

            trans_path = Wmt24ppInstructionRow.get_savepath(f"google/{model}", args.lang, use_instructions)
            if not trans_path.exists():
                print(f"[skip] {model} ({suffix} instr): {trans_path} not found")
                continue

            rows = Wmt24ppInstructionRow.load(trans_path)
            if args.limit:
                rows = rows[:args.limit]

            inline_requests = build_inline_requests(rows, lang_name, response_schema)
            request_path = get_request_path(args.lang, model, use_instructions)
            save_inline_requests(inline_requests, request_path)

            display_name = f"wmt24pp-eval-{args.lang}-{model}-{suffix}-{int(time.time())}"
            try:
                job = create_batch_job(client, inline_requests, args.eval_model, display_name)
            except Exception as e:
                print(f"[error] {model} ({suffix} instr): {e}")
                continue
            count = len(inline_requests)
            print(f"Created batch job: {job.name} (state={job.state})")

            now = datetime.now().isoformat(timespec="seconds")
            result_path = get_result_path(args.lang, model, use_instructions)
            tracker.append(BatchTrackerEntry(
                batch_name=job.name or "",
                display_name=display_name,
                created_at=now,
                model=model,
                lang=args.lang,
                use_instructions=use_instructions,
                eval_model=args.eval_model,
                num_requests=count,
                state=str(job.state) if job.state else "UNKNOWN",
                updated_at=now,
                request_file=str(request_path),
                result_file=str(result_path),
            ))


def action_poll(args):
    """Check current status of pending batches (non-blocking). Download results for completed jobs."""
    tracker = BatchTracker()
    entries = tracker.load_all()
    client = genai.Client()

    pending_states = {
        "JobState.JOB_STATE_PENDING", "JobState.JOB_STATE_RUNNING",
        "JOB_STATE_PENDING", "JOB_STATE_RUNNING",
    }

    updated = False
    for entry in entries:
        if entry.state not in pending_states:
            continue

        job = client.batches.get(name=entry.batch_name)
        new_state = str(job.state) if job.state else "UNKNOWN"

        if new_state != entry.state:
            print(f"{entry.batch_name}: {entry.state} -> {new_state}")
            entry.state = new_state
            entry.updated_at = datetime.now().isoformat(timespec="seconds")
            updated = True

            if job.state == JobState.JOB_STATE_SUCCEEDED:
                output_path = Path(entry.result_file)
                result_file = job.dest.file_name if job.dest else None
                if result_file:
                    download_batch_results(client, result_file, output_path)
                    print(f"  Downloaded results to {output_path}")
                elif job.dest and job.dest.inlined_responses:
                    save_inline_responses(job.dest.inlined_responses, output_path)
                    print(f"  Saved {len(job.dest.inlined_responses)} inline responses to {output_path}")
                else:
                    print(f"  WARNING: Job succeeded but no results found")
            elif job.state == JobState.JOB_STATE_FAILED:
                print(f"  Job failed.")
                if hasattr(job, "error") and job.error:
                    print(f"  Error: {job.error}")
        else:
            print(f"{entry.batch_name}: still {entry.state}")

    if updated:
        tracker.rewrite(entries)
    if not any(e.state in pending_states for e in entries):
        print("No pending batches.")


def action_parse(args):
    """Parse downloaded batch results into Wmt24ppEvaluatedRow JSONL."""
    results = {}

    for model in args.models:
        for use_instructions in [True, False]:
            suffix = "with" if use_instructions else "without"
            result_path = get_result_path(args.lang, model, use_instructions)
            if not result_path.exists():
                print(f"[skip] {model} ({suffix} instr): {result_path} not found")
                continue

            trans_path = Wmt24ppInstructionRow.get_savepath(f"google/{model}", args.lang, use_instructions)
            if not trans_path.exists():
                print(f"[skip] {model} ({suffix} instr): translation file {trans_path} not found")
                continue

            rows = Wmt24ppInstructionRow.load(trans_path)
            rows_by_key = {r.segment_key: r for r in rows}

            parsed, errors = parse_batch_results(result_path, rows_by_key)
            if errors:
                print(f"\n{len(errors)} errors for {model} ({suffix} instr):")
                for err in errors[:10]:
                    print(f"  - {err}")

            if not parsed:
                print(f"No parsed results for {model} ({suffix} instr)")
                continue

            eval_path = Wmt24ppEvaluatedRow.get_savepath(
                model, args.lang, use_instructions, args.eval_model,
            )
            Wmt24ppEvaluatedRow.save(parsed, eval_path)
            print(f"Saved {len(parsed)} evaluations to {eval_path}")

            metrics = Wmt24ppEvaluatedRow.compute_metrics(parsed)
            results[(model, use_instructions)] = metrics
            print(f"  Critical: {metrics.critical_pct:.1f}%, Rating: {metrics.translation_rating_avg:.1f}, Adapted: {metrics.adaptedness_avg:.1f}")

    if results:
        EvaluationSummaryMetrics.print_summary_table(results, args.lang, args.models)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WMT24++ LLM judge evaluation via Gemini Batch API")
    parser.add_argument("--lang", type=str, default="en-fr_FR",
                        help="WMT24++ language pair (default: en-fr_FR)")
    parser.add_argument("--eval-model", type=str, default=DEFAULT_EVAL_MODEL,
                        help=f"Evaluation model (default: {DEFAULT_EVAL_MODEL})")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                        help=f"Translation models to evaluate (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of rows per model/condition (for testing)")

    # Mutually exclusive actions
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true",
                        help="Verify prompts match existing evaluator (no API calls)")
    action.add_argument("--submit", action="store_true",
                        help="Build requests, upload, and create batch jobs")
    action.add_argument("--poll", action="store_true",
                        help="Poll pending batches and download results")
    action.add_argument("--parse", action="store_true",
                        help="Parse downloaded results into evaluation JSONL")

    args = parser.parse_args()

    if args.dry_run:
        action_dry_run(args)
    elif args.submit:
        action_submit(args)
    elif args.poll:
        action_poll(args)
    elif args.parse:
        action_parse(args)


if __name__ == "__main__":
    main()
