"""Generate self-instructions for BOUQuET sentences using vLLM + Gemma-3.

For each BOUQuET sentence, prompt an LLM to generate a user_instruction from
one of three input variants:

- sent: only the English target sentence
- para: target sentence + its paragraph siblings (target marked with [TARGET])
- meta: domain + par_comment + tags + the paragraph + target sentence

Output is written in the same schema as datasets/bouquet_instructions_dev.jsonl
(BouquetInstruction), so translate/bouquet_vllm.py --use-user-instructions can
consume it with just a path swap.

Only English (tgt_text) is used as input. BOUQuET is multi-parallel and the
metadata fields (domain, par_comment, tags, register) are loaded once from
fra_Latn to mirror the gold-instruction pipeline in
instructions/generate_bouquet.py.

Few-shot examples are sampled from datasets/bouquet_instructions_dev.jsonl
with a fixed seed; for the para and meta variants the few-shot inputs are
re-rendered in the variant's format (their paragraph siblings are
reconstructed from the BOUQuET load), while the assistant output is the gold
instruction in all variants.

Usage:
    python -m self_instruct.bouquet --variant para
    python -m self_instruct.bouquet --variant sent --split test
    python -m self_instruct.bouquet --variant meta --model google/gemma-3-27b-it
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.bouquet_utils import BouquetRow

# -- Constants ----------------------------------------------------------------

REFERENCE_LANG = "fra_Latn"  # mirror instructions/generate_bouquet.py
DEFAULT_MODEL = "google/gemma-3-27b-it"
DEFAULT_SPLIT = "dev"
DEFAULT_VARIANT = "para"
VARIANTS = ("sent", "para", "meta")
NUM_FEW_SHOT = 3
FEW_SHOT_SEED = 42
GOLD_INSTRUCTIONS_PATH = Path("datasets/bouquet_instructions_dev.jsonl")
TEMPERATURE = 0.0
MAX_TOKENS = 256
MAX_MODEL_LEN = 4096


# -- Schemas ------------------------------------------------------------------

class BouquetInstruction(BaseModel):
    """Mirrors the schema of datasets/bouquet_instructions_dev.jsonl."""
    uniq_id: str
    tgt_text: str
    domain: str
    par_comment: str
    user_instruction: str
    tags: str
    register: str

    @classmethod
    def load(cls, path: Path) -> List["BouquetInstruction"]:
        out: List[BouquetInstruction] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(cls.model_validate_json(line))
        return out

    @classmethod
    def save(cls, rows: List["BouquetInstruction"], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(row.model_dump_json() + "\n")


# -- Data loading -------------------------------------------------------------

def load_bouquet(split: str) -> Tuple[List[BouquetRow], Dict[str, List[BouquetRow]]]:
    """Load BOUQuET via fra_Latn and group by par_id (sorted by uniq_id)."""
    rows = BouquetRow.load(REFERENCE_LANG, split)
    paragraphs: Dict[str, List[BouquetRow]] = defaultdict(list)
    for r in rows:
        paragraphs[r.par_id].append(r)
    for par_id in paragraphs:
        paragraphs[par_id].sort(key=lambda r: r.uniq_id)
    return rows, dict(paragraphs)


# -- Prompt building ----------------------------------------------------------

SYSTEM_PROMPTS = {
    "sent": (
        "You are an expert at writing translation instructions. Given a single "
        "English sentence, write a concise user instruction for a translator. "
        "The instruction should specify:\n"
        "1. The likely translation setting (e.g., blog post, formal document, "
        "social media, novel)\n"
        "2. The target audience\n"
        "3. The appropriate tone and style\n\n"
        "Output ONLY the instruction, nothing else. Keep it to 1-3 sentences."
    ),
    "para": (
        "You are an expert at writing translation instructions. Given an English "
        "paragraph with one target sentence marked, write a concise user "
        "instruction for translating the target sentence, using the surrounding "
        "paragraph as context. The instruction should specify:\n"
        "1. The translation setting (e.g., blog post, formal document, social "
        "media, novel)\n"
        "2. The target audience\n"
        "3. The appropriate tone and style\n\n"
        "Output ONLY the instruction, nothing else. Keep it to 1-3 sentences."
    ),
    "meta": (
        "You are an expert at writing translation instructions. Given metadata "
        "about a text segment (domain, context description, linguistic tags), "
        "the surrounding English paragraph, and a target sentence, write a "
        "concise user instruction for translating the target sentence. "
        "The instruction should specify:\n"
        "1. The translation setting (e.g., blog post, formal document, social "
        "media, novel)\n"
        "2. The target audience\n"
        "3. The appropriate tone and style\n\n"
        "Output ONLY the instruction, nothing else. Keep it to 1-3 sentences."
    ),
}


def format_paragraph(par_rows: List[BouquetRow], target_uniq_id: str) -> str:
    lines = []
    for i, r in enumerate(par_rows, start=1):
        marker = " [TARGET]" if r.uniq_id == target_uniq_id else ""
        lines.append(f"S{i}{marker}: {r.tgt_text}")
    return "\n".join(lines)


def format_input(
    uniq_id: str,
    tgt_text: str,
    domain: str,
    par_comment: str,
    tags: str,
    par_rows: List[BouquetRow],
    variant: str,
) -> str:
    if variant == "sent":
        return (
            f"Sentence: {tgt_text}\n\n"
            "Generate a user instruction for translating this sentence."
        )
    if variant == "para":
        return (
            "Paragraph (target sentence marked):\n"
            f"{format_paragraph(par_rows, uniq_id)}\n\n"
            f"Target sentence: {tgt_text}\n\n"
            "Generate a user instruction for translating the target sentence."
        )
    if variant == "meta":
        return (
            f"Domain: {domain}\n"
            f"Context: {par_comment}\n"
            f"Linguistic tags: {tags}\n\n"
            "Paragraph (target sentence marked):\n"
            f"{format_paragraph(par_rows, uniq_id)}\n\n"
            f"Target sentence: {tgt_text}\n\n"
            "Generate a user instruction for translating the target sentence."
        )
    raise ValueError(f"Unknown variant: {variant}")


def select_few_shot(
    gold_path: Path,
    rows_by_id: Dict[str, BouquetRow],
    n: int = NUM_FEW_SHOT,
    seed: int = FEW_SHOT_SEED,
) -> List[BouquetInstruction]:
    """Sample n few-shot examples from gold instructions, restricted to rows
    present in the loaded BOUQuET split (so paragraph siblings are available).
    """
    gold = BouquetInstruction.load(gold_path)
    eligible = [g for g in gold if g.uniq_id in rows_by_id]
    assert len(eligible) >= n, (
        f"Need at least {n} eligible few-shot examples, got {len(eligible)} "
        f"(gold rows: {len(gold)}, loaded BOUQuET rows: {len(rows_by_id)}). "
        "Likely cause: gold instructions and the loaded BOUQuET split have "
        "disjoint uniq_ids — load the dev BOUQuET split for few-shot context."
    )
    rng = random.Random(seed)
    return rng.sample(eligible, n)


def build_messages(
    row: BouquetRow,
    par_rows: List[BouquetRow],
    few_shot: List[BouquetInstruction],
    fs_paragraphs: Dict[str, List[BouquetRow]],
    fs_rows_by_id: Dict[str, BouquetRow],
    variant: str,
) -> List[dict]:
    """Build the chat messages for one target row.

    `par_rows` is the paragraph for the *target* sentence (from the generation
    split). `fs_paragraphs` / `fs_rows_by_id` are loaded from the dev split,
    since gold instructions are dev-only and we need their paragraph siblings
    to render few-shot inputs in the para/meta variants.
    """
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPTS[variant]}]
    for fs in few_shot:
        fs_row = fs_rows_by_id[fs.uniq_id]
        fs_par_rows = fs_paragraphs[fs_row.par_id]
        messages.append({
            "role": "user",
            "content": format_input(
                uniq_id=fs.uniq_id,
                tgt_text=fs.tgt_text,
                domain=fs.domain,
                par_comment=fs.par_comment,
                tags=fs.tags,
                par_rows=fs_par_rows,
                variant=variant,
            ),
        })
        messages.append({"role": "assistant", "content": fs.user_instruction})
    messages.append({
        "role": "user",
        "content": format_input(
            uniq_id=row.uniq_id,
            tgt_text=row.tgt_text,
            domain=row.domain,
            par_comment=row.par_comment,
            tags=row.tags,
            par_rows=par_rows,
            variant=variant,
        ),
    })
    return messages


# -- Output paths -------------------------------------------------------------

def output_path(split: str, model: str, variant: str) -> Path:
    """Filename matches the bouquet_instructions_{split}_{instruction_type}.jsonl
    convention used by translate/bouquet_vllm.py --instruction-type, so the file
    can be consumed with no code changes (instruction_type = "self-{variant}-{model_slug}").
    """
    model_slug = model.split("/")[-1]
    instruction_type = f"self-{variant}-{model_slug}"
    return Path(f"datasets/bouquet_instructions_{split}_{instruction_type}.jsonl")


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BOUQuET self-instructions via vLLM."
    )
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    from vllm import LLM, SamplingParams

    args = parse_args()
    out_path = args.output_path or output_path(args.split, args.model, args.variant)

    print(f"Variant: {args.variant}")
    print(f"Split:   {args.split}")
    print(f"Model:   {args.model}")
    print(f"Output:  {out_path}")
    print(f"Few-shot: {NUM_FEW_SHOT} from {GOLD_INSTRUCTIONS_PATH} (seed={FEW_SHOT_SEED})")
    print(f"Reference lang for BOUQuET load: {REFERENCE_LANG}")

    if out_path.exists() and not args.ignore_cache:
        print(f"\nResults already exist at {out_path}")
        print("Use --ignore-cache to regenerate")
        existing = BouquetInstruction.load(out_path)
        print(f"Loaded {len(existing)} existing rows")
        return

    rows, paragraphs = load_bouquet(args.split)
    rows_by_id = {r.uniq_id: r for r in rows}
    print(f"Loaded {len(rows)} BOUQuET rows ({len(paragraphs)} paragraphs)")

    # Few-shot context always comes from the dev split, since
    # bouquet_instructions_dev.jsonl is the only gold pool we have and we need
    # its paragraph siblings to render the para/meta variants.
    if args.split == "dev":
        fs_rows_by_id, fs_paragraphs = rows_by_id, paragraphs
    else:
        fs_rows, fs_paragraphs = load_bouquet("dev")
        fs_rows_by_id = {r.uniq_id: r for r in fs_rows}
        print(f"Loaded {len(fs_rows)} dev BOUQuET rows for few-shot context")

    if args.limit:
        rows = rows[: args.limit]
        print(f"Limiting to {len(rows)} rows")

    few_shot = select_few_shot(GOLD_INSTRUCTIONS_PATH, fs_rows_by_id)
    print(f"\nFew-shot examples ({len(few_shot)}):")
    for fs in few_shot:
        print(f"  [{fs.uniq_id}] {fs.domain}: {fs.user_instruction[:80]}...")

    all_messages = [
        build_messages(
            row=row,
            par_rows=paragraphs[row.par_id],
            few_shot=few_shot,
            fs_paragraphs=fs_paragraphs,
            fs_rows_by_id=fs_rows_by_id,
            variant=args.variant,
        )
        for row in rows
    ]

    print("\nSample prompt for first row:")
    for msg in all_messages[0]:
        snippet = msg["content"].replace("\n", " ⏎ ")
        print(f"  [{msg['role']}]: {snippet}")

    print(f"\nLoading model {args.model}...")
    llm = LLM(args.model, max_model_len=MAX_MODEL_LEN)

    sampling_params = SamplingParams(
        n=1,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    print(f"Generating instructions for {len(all_messages)} sentences...")
    outputs = llm.chat(all_messages, sampling_params=sampling_params)

    results: List[BouquetInstruction] = []
    for row, output in zip(rows, outputs):
        text = output.outputs[0].text.strip()
        results.append(BouquetInstruction(
            uniq_id=row.uniq_id,
            tgt_text=row.tgt_text,
            domain=row.domain,
            par_comment=row.par_comment,
            tags=row.tags,
            register=row.register,
            user_instruction=text,
        ))

    BouquetInstruction.save(results, out_path)
    print(f"\nSaved {len(results)} self-instructions to {out_path}")

    print("\nSample outputs:")
    for r in results[:5]:
        print(f"  [{r.uniq_id}] {r.domain}")
        print(f"    EN:    {r.tgt_text[:100]}")
        print(f"    Instr: {r.user_instruction}")
        print()


if __name__ == "__main__":
    main()
