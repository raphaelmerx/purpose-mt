"""Generate self-instructions for WMT24++ segments using vLLM + Gemma-3.

The WMT24++ analogue of self_instruct/bouquet.py (BOUQuET). For
each WMT24++ segment we prompt an LLM with the English document, marking the
target sentence with a [TARGET] tag, and ask it to draft a concise user
instruction (setting / audience / tone). This is the "para" variant of the
self-instruction pipeline (\\S sec:self-instr), with the WMT24++ document group
playing the role of the BOUQuET paragraph.

Language-agnostic: instructions are generated once from the English source
(loaded via en-fr_FR), shared across all WMT24++ language subsets, exactly like
generate_wmt24pp_instructions.py.

Output is written in the Wmt24ppInstruction schema to
datasets/wmt24pp_instructions_self-para.jsonl, so trans_wmt24pp_vllm.py
--instruction-type self-para can consume it with no code changes.

Usage:
    python -m self_instruct.wmt24pp --dry-run   # inspect row-0 prompt
    python -m self_instruct.wmt24pp             # full run
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
from pathlib import Path
from typing import List

from common.wmt24pp_utils import (
    Wmt24ppDocumentGroup,
    Wmt24ppInstruction,
    Wmt24ppInstructionRow,
    get_instructions_path,
)


DEFAULT_MODEL = "google/gemma-3-27b-it"
INSTRUCTION_TYPE = "self-para"
TEMPERATURE = 0.0
MAX_TOKENS = 256
MAX_MODEL_LEN = 8192
REFERENCE_LANG = "en-fr_FR"  # instructions are language-agnostic; load English once


SYSTEM_PROMPT = (
    "You are an expert at writing translation instructions. Given an English "
    "document with one target sentence marked, write a concise user "
    "instruction for translating the target sentence, using the surrounding "
    "document as context. The instruction should specify:\n"
    "1. The translation setting (e.g., news article, social media post, "
    "literary fiction, spoken transcript)\n"
    "2. The target audience\n"
    "3. The appropriate tone and style\n\n"
    "Output ONLY the instruction, nothing else. Keep it to 1-3 sentences."
)

# Hand-written few-shot examples: each is a short marked document -> instruction,
# demonstrating the marked-sentence-within-document task across WMT24++ domains.
FEW_SHOT_EXAMPLES = [
    {
        "doc": [
            "ok i finally caved and tried it after seeing it on my feed for months",
            'celeste is my favorite "tiny good freeware game to big bloated '
            'commercial game with stupidly detailed art" corruption arc',
            "anyway back to doomscrolling lol",
        ],
        "target_idx": 1,
        "instruction": (
            "Translate as an informal social media post by a gamer. Use casual, "
            "opinionated language with internet-native phrasing suitable for a "
            "gaming community audience."
        ),
    },
    {
        "doc": [
            "Siso's depictions of land, water center new gallery exhibition",
            "The show opens Friday at the downtown arts center and runs through "
            "the end of the month.",
            "Curators say the works reflect a decade of the artist's fieldwork.",
        ],
        "target_idx": 0,
        "instruction": (
            "Translate as a news headline for a general readership. Use a "
            "neutral, informative journalistic tone appropriate for an arts and "
            "culture section."
        ),
    },
    {
        "doc": [
            "The house had stood empty for years.",
            "The fruit trees in the small garden behind the house had long since "
            "stopped bearing, but the roses, roses and more roses still gave off "
            "a velvety warmth.",
            "She lingered at the gate, unwilling to step inside.",
        ],
        "target_idx": 1,
        "instruction": (
            "Translate as literary fiction with rich, evocative prose. Maintain "
            "the poetic imagery and rhythm of the original, targeting readers of "
            "literary novels."
        ),
    },
]


def format_document(lines: List[str], target_idx: int) -> str:
    """Render a document as an S1:/S2:/... numbered list, marking the target."""
    out = []
    for i, line in enumerate(lines):
        marker = " [TARGET]" if i == target_idx else ""
        out.append(f"S{i + 1}{marker}: {line}")
    return "\n".join(out)


def format_user_message(doc_text: str, target_sentence: str) -> str:
    return (
        "Document (target sentence marked):\n"
        f"{doc_text}\n\n"
        f"Target sentence: {target_sentence}\n\n"
        "Generate a user instruction for translating the target sentence."
    )


def build_messages(dg: Wmt24ppDocumentGroup, target_idx: int) -> List[dict]:
    """Build chat messages for one target segment within its document group."""
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        doc_text = format_document(ex["doc"], ex["target_idx"])
        target_sentence = ex["doc"][ex["target_idx"]]
        messages.append({
            "role": "user",
            "content": format_user_message(doc_text, target_sentence),
        })
        messages.append({"role": "assistant", "content": ex["instruction"]})

    doc_lines = [r.source for r in dg.rows]
    doc_text = format_document(doc_lines, target_idx)
    messages.append({
        "role": "user",
        "content": format_user_message(doc_text, dg.rows[target_idx].source),
    })
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate WMT24++ self-instructions from document context."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of segments (for testing)")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the first segment's prompt and exit (no model load)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = get_instructions_path(INSTRUCTION_TYPE)

    print(f"Model:   {args.model}")
    print(f"Output:  {out_path}")
    print(f"Few-shot: {len(FEW_SHOT_EXAMPLES)} hand-written marked-document examples")

    if out_path.exists() and not args.ignore_cache and not args.dry_run:
        print(f"\nResults already exist at {out_path}")
        print("Use --ignore-cache to regenerate")
        return

    # Load segments once (language-agnostic) and group into documents.
    rows = Wmt24ppInstructionRow.load_for_lang(REFERENCE_LANG)
    doc_groups = Wmt24ppDocumentGroup.group_by_document(rows)
    print(f"Loaded {len(rows)} segments in {len(doc_groups)} documents")

    # One (doc_group, idx) job per segment, paired with the segment row.
    jobs = []  # (Wmt24ppInstructionRow, messages)
    for dg in doc_groups:
        for idx, row in enumerate(dg.rows):
            jobs.append((row, build_messages(dg, idx)))

    if args.limit:
        jobs = jobs[: args.limit]
        print(f"Limiting to {len(jobs)} segments")

    print("\nSample prompt for first segment:")
    for msg in jobs[0][1]:
        snippet = msg["content"].replace("\n", " / ")
        print(f"  [{msg['role']}]: {snippet}")

    if args.dry_run:
        print("\n--dry-run: stopping before model load.")
        return

    from vllm import LLM, SamplingParams

    print(f"\nLoading model {args.model}...")
    llm = LLM(args.model, max_model_len=MAX_MODEL_LEN)
    sampling_params = SamplingParams(n=1, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    print(f"Generating self-instructions for {len(jobs)} segments...")
    outputs = llm.chat([m for _, m in jobs], sampling_params=sampling_params)

    results: List[Wmt24ppInstruction] = []
    for (row, _), output in zip(jobs, outputs):
        text = output.outputs[0].text.strip()
        results.append(Wmt24ppInstruction(
            segment_key=row.segment_key,
            source=row.source,
            domain=row.domain,
            document_id=row.document_id,
            user_instruction=text,
        ))

    results.sort(key=lambda x: x.segment_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for inst in results:
            f.write(inst.model_dump_json() + "\n")
    print(f"\nSaved {len(results)} self-instructions to {out_path}")

    print("\nSample outputs:")
    for inst in results[:5]:
        print(f"  [{inst.domain}] {inst.source[:80]}")
        print(f"    -> {inst.user_instruction}")
        print()


if __name__ == "__main__":
    main()
