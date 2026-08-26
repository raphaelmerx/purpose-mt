"""Generate BOUQuET user instructions with the Gemini Batch API (paper §3.2, Fig 1a).

For each BOUQuET row, Gemini drafts a language-agnostic instruction describing the
translation's intended setting, audience, and tone, conditioning on the row's domain,
`par_comment` and linguistic tags. The instructions used in the paper were then
manually revised by a human annotator; this script produces the pre-revision drafts.

BOUQuET metadata is shared across all language pairs, so any single language can be
used to load it (`--reference-lang`).

Usage:
    # submit a new batch job and poll until it finishes
    python -m instructions.generate_bouquet --split dev

    # resume polling an existing job
    python -m instructions.generate_bouquet --split dev --batch-name batches/abc123

    # parse already-downloaded results without touching the API
    python -m instructions.generate_bouquet --split dev --parse-only

Output: datasets/bouquet_instructions_{split}.jsonl
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from google import genai
from google.genai import types
from google.genai.types import BatchJob, JobState

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.bouquet_utils import BouquetRow
from translate.bouquet_vllm import BouquetInstruction

DEFAULT_MODEL = "models/gemini-3-flash-preview"
DEFAULT_REFERENCE_LANG = "fra_Latn"
POLL_SECONDS = 10


# -- Prompt -------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert at creating translation instructions. Given metadata about a text segment (domain, context description, linguistic tags, and a reference translation), generate a concise user instruction that a translator would use to properly translate the source text.

The instruction should:
1. Specify the translation setting (e.g., blog post, formal document, social media, novel)
2. Define the target audience (e.g., general readers, professionals, fans of a genre)
3. Indicate the appropriate tone and style

Output ONLY the instruction, nothing else. Keep it to 1-3 sentences."""

FEW_SHOT_EXAMPLES = [
    {
        "input": {
            "domain": "web misc.",
            "par_comment": "This paragraph exemplifies a job posting on a social media platform or online bulletin board. It contains informal language identifiable by interrogative and exclamatory sentences. It also includes a common strategy for gender neutral language. The second person is generic and refers to anyone interested in the job post. The first person plural is exclusive and refers to the company",
            "tags": "<na>",
            "tgt_text": "Are you interested in the technology sector?",
        },
        "output": "Translate as a casual job posting for a tech company's social media page. Use informal, engaging language with gender-neutral terms. The tone should be friendly and inviting to potential applicants.",
    },
    {
        "input": {
            "domain": "social posts",
            "par_comment": "Blog of a movie review",
            "tags": "phrase:normalisation",
            "tgt_text": "However, as a film with a local Hong Kong theme, it has achieved a certain balance between niche characteristics and the market.",
        },
        "output": "Translate as a blog post reviewing a movie for film enthusiasts. Use a conversational yet analytical tone typical of online movie criticism. The audience is general readers interested in cinema.",
    },
    {
        "input": {
            "domain": "narration",
            "par_comment": "Imaginary description of a scene in a novel (probably set in the past) in which a young woman is compelled by her father to go to a party where she will meet a young man who wants to approach her. Diane is of the feminine grammatical gender; Aymeric of the masculine grammatical gender.",
            "tags": "clause:comparative, clause:preposition_phrase, phrase:verb_tense, phrase:adjective_order",
            "tgt_text": "The reception hall where Diane entered was bathed in cold light and had been decked with sparkling colors without any regard for harmony.",
        },
        "output": "Translate as literary fiction set in a historical period. Use elegant, descriptive prose with attention to atmosphere and imagery. Maintain the narrative voice of a third-person omniscient narrator for a novel's readership.",
    },
]


def format_user_message(domain: str, par_comment: str, tags: str, tgt_text: str) -> str:
    return f"""Domain: {domain}
Context: {par_comment}
Linguistic tags: {tags}
Reference translation: {tgt_text}

Generate a user instruction for translating this text."""


def build_contents(row: BouquetRow) -> List[dict]:
    """Gemini contents list: few-shot examples followed by the row's prompt."""
    contents: List[dict] = []
    for example in FEW_SHOT_EXAMPLES:
        inp = example["input"]
        contents.append({
            "role": "user",
            "parts": [{"text": format_user_message(
                inp["domain"], inp["par_comment"], inp["tags"], inp["tgt_text"]
            )}],
        })
        contents.append({"role": "model", "parts": [{"text": example["output"]}]})

    contents.append({
        "role": "user",
        "parts": [{"text": format_user_message(
            row.domain, row.par_comment, row.tags, row.tgt_text
        )}],
    })
    return contents


# -- Batch API ----------------------------------------------------------------


def build_batch_request(row: BouquetRow) -> dict:
    return {
        "key": row.uniq_id,
        "request": {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": build_contents(row),
        },
    }


def write_batch_requests(rows: Iterable[BouquetRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(build_batch_request(row), ensure_ascii=False) + "\n")


def is_completed(job: BatchJob) -> bool:
    return job.state in {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
    }


def create_batch_job(client: genai.Client, input_path: Path, split: str, model: str) -> BatchJob:
    uploaded = client.files.upload(
        file=str(input_path),
        config=types.UploadFileConfig(
            display_name=f"bouquet-instructions-{split}",
            mime_type="application/jsonl",
        ),
    )
    return client.batches.create(
        model=model,
        src=uploaded.name,
        config={"display_name": f"bouquet-generate-instructions-{split}"},
    )


def poll_batch_job(client: genai.Client, job: BatchJob, poll_seconds: int = POLL_SECONDS) -> BatchJob:
    while not is_completed(job):
        time.sleep(poll_seconds)
        job = client.batches.get(name=job.name)
    return job


def download_batch_results(client: genai.Client, result_file: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_content = client.files.download(file=result_file)
    if isinstance(file_content, (bytes, bytearray)):
        output_path.write_bytes(file_content)
    elif hasattr(file_content, "decode"):
        output_path.write_text(file_content.decode("utf-8"), encoding="utf-8")
    else:
        output_path.write_text(str(file_content), encoding="utf-8")


def parse_batch_results(
    result_path: Path, rows_by_id: Dict[str, BouquetRow]
) -> List[BouquetInstruction]:
    instructions: List[BouquetInstruction] = []
    errors: List[str] = []

    with open(result_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            key = payload.get("key") or payload.get("metadata", {}).get("key")
            response = payload.get("response") or payload.get("inlineResponse")
            error = payload.get("error")

            if error:
                errors.append(f"{key}: {error}")
                continue

            if response is None and "candidates" in payload:
                response = payload

            if not response or "candidates" not in response:
                errors.append(f"{key}: missing response candidates")
                continue

            parts = response["candidates"][0]["content"]["parts"]
            text = "".join(
                part.get("text", "") for part in parts if isinstance(part, dict)
            ).strip()
            if not text:
                errors.append(f"{key}: empty response text")
                continue

            row = rows_by_id.get(key)
            if not row:
                errors.append(f"{key}: missing row metadata")
                continue

            instructions.append(
                BouquetInstruction(
                    uniq_id=row.uniq_id,
                    tgt_text=row.tgt_text,
                    domain=row.domain,
                    par_comment=row.par_comment,
                    tags=row.tags,
                    register=row.register,
                    user_instruction=text,
                )
            )

    if errors:
        print(f"Encountered {len(errors)} errors while parsing batch results")
        for err in errors[:10]:
            print(f"- {err}")

    return instructions


# -- CLI ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate BOUQuET user instructions via the Gemini Batch API."
    )
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--reference-lang", default=DEFAULT_REFERENCE_LANG,
        help="Any BOUQuET language; metadata is shared across all of them.",
    )
    p.add_argument(
        "--batch-name", default=None,
        help="Resume polling an existing batch job, e.g. batches/abc123.",
    )
    p.add_argument(
        "--parse-only", action="store_true",
        help="Parse an already-downloaded results file without calling the API.",
    )
    p.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    requests_path = Path(f"datasets/bouquet_generate_instructions_{args.split}_requests.jsonl")
    results_path = Path(f"datasets/bouquet_generate_instructions_{args.split}_results.jsonl")
    output_path = BouquetInstruction.get_data_path(args.split)

    if output_path.exists() and not args.overwrite:
        print(f"Output exists: {output_path}. Use --overwrite.")
        return

    rows = BouquetRow.load(args.reference_lang, args.split)
    print(f"Loaded {len(rows)} rows from BOUQuET ({args.split} split)")

    if not args.parse_only:
        client = genai.Client()
        if args.batch_name:
            job = client.batches.get(name=args.batch_name)
            print(f"Resuming batch job: {job.name} (state: {job.state})")
        else:
            write_batch_requests(rows, requests_path)
            print(f"Wrote batch requests to {requests_path}")
            job = create_batch_job(client, requests_path, args.split, args.model)
            print(f"Created batch job: {job.name}")

        job = poll_batch_job(client, job, args.poll_seconds)
        print(f"Batch job finished with state: {job.state}")

        if job.state != JobState.JOB_STATE_SUCCEEDED:
            print(f"Batch job did not succeed: {job.error}")
            return

        result_file = job.dest.file_name if job.dest else None
        if not result_file:
            print("Batch job succeeded but no result file was found.")
            return
        if results_path.exists():
            print(f"Results already downloaded: {results_path}")
        else:
            download_batch_results(client, result_file, results_path)
            print(f"Downloaded results to {results_path}")

    if not results_path.exists():
        print(f"No batch results file at {results_path}")
        return

    rows_by_id = {row.uniq_id: row for row in rows}
    instructions = parse_batch_results(results_path, rows_by_id)
    print(f"Parsed {len(instructions)} instructions")

    random.seed(42)
    for inst in random.sample(instructions, min(5, len(instructions))):
        print(f"\nID: {inst.uniq_id}")
        print(f"  EN:     {inst.tgt_text[:100]}")
        print(f"  Meta:   {inst.domain} | {inst.tags}")
        print(f"  Instr:  {inst.user_instruction}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for inst in instructions:
            f.write(inst.model_dump_json() + "\n")
    print(f"\nSaved {len(instructions)} instructions to {output_path}")


if __name__ == "__main__":
    main()
