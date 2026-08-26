"""
Generate per-segment translation instructions for WMT24++ using cached_completion.

Language-agnostic: instructions are generated from English source text + domain only,
shared across all WMT24++ language subsets.

Usage:
    python -m instructions.generate_wmt24pp --limit 10  # test first
    python -m instructions.generate_wmt24pp              # full run
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from common.trans_utils import Wmt24ppSegment, cached_completion
from common.wmt24pp_utils import Wmt24ppInstruction, INSTRUCTIONS_PATH


SYSTEM_PROMPT = """You are an expert at creating translation instructions. Given a text's domain and the English source text, generate a concise user instruction that a translator would use to properly translate this text.

The instruction should:
1. Specify the translation setting (e.g., social media post, news article, literary fiction, spoken transcript)
2. Define the target audience
3. Indicate the appropriate tone and style

Output ONLY the instruction, nothing else. Keep it to 1-3 sentences."""

FEW_SHOT_EXAMPLES = [
    {
        "domain": "social",
        "source": 'celeste is my favorite "tiny good freeware game to big bloated commercial game with stupidly detailed art" corruption arc',
        "instruction": "Translate as an informal social media post by a gamer. Use casual, opinionated language with internet-native phrasing suitable for a gaming community audience.",
    },
    {
        "domain": "news",
        "source": "Siso's depictions of land, water center new gallery exhibition",
        "instruction": "Translate as a news headline and article opening for a general readership. Use a neutral, informative journalistic tone appropriate for an arts and culture section.",
    },
    {
        "domain": "literary",
        "source": "The fruit trees in the small garden behind the house had long since stopped bearing, but the roses, roses and more roses still gave off a velvety warmth.",
        "instruction": "Translate as literary fiction with rich, evocative prose. Maintain the poetic imagery and rhythm of the original, targeting readers of literary novels.",
    },
]

DEFAULT_MODEL = "gemini/gemini-3-flash-preview"
MAX_WORKERS = 8


def build_messages(domain: str, source: str) -> list[dict]:
    """Build chat messages for instruction generation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": f"Domain: {ex['domain']}\nSource text: {ex['source']}\n\nGenerate a user instruction for translating this text.",
        })
        messages.append({"role": "assistant", "content": ex["instruction"]})

    messages.append({
        "role": "user",
        "content": f"Domain: {domain}\nSource text: {source}\n\nGenerate a user instruction for translating this text.",
    })
    return messages


def generate_instruction(segment: Wmt24ppSegment, model: str) -> Wmt24ppInstruction:
    """Generate an instruction for a single WMT24++ segment."""
    messages = build_messages(segment.domain, segment.source)
    response = cached_completion(model=model, messages=messages)

    return Wmt24ppInstruction(
        segment_key=f"{segment.document_id}_{segment.segment_id}",
        source=segment.source,
        domain=segment.domain,
        document_id=segment.document_id,
        user_instruction=response.strip(),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate WMT24++ translation instructions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of segments (for testing)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    # Load segments from any language (instructions are language-agnostic)
    segments = Wmt24ppSegment.load("en-fr_FR")
    segments = [s for s in segments if s.domain != "canary"]
    print(f"Loaded {len(segments)} non-canary WMT24++ segments")

    if args.limit:
        segments = segments[:args.limit]
        print(f"Limited to {len(segments)} segments")

    instructions = []
    errors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(generate_instruction, seg, args.model): seg
            for seg in segments
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            seg = futures[future]
            try:
                inst = future.result()
                instructions.append(inst)
            except Exception as e:
                key = f"{seg.document_id}_{seg.segment_id}"
                errors.append((key, str(e)))
                print(f"Error for {key}: {e}")

    if errors:
        print(f"\n{len(errors)} errors occurred")

    # Sort by segment_key for reproducibility
    instructions.sort(key=lambda x: x.segment_key)

    # Save
    INSTRUCTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INSTRUCTIONS_PATH, "w", encoding="utf-8") as f:
        for inst in instructions:
            f.write(inst.model_dump_json() + "\n")
    print(f"\nSaved {len(instructions)} instructions to {INSTRUCTIONS_PATH}")

    # Print samples
    import random
    random.seed(42)
    for inst in random.sample(instructions, min(5, len(instructions))):
        print(f"\n[{inst.domain}] {inst.source[:80]}...")
        print(f"  → {inst.user_instruction}")


if __name__ == "__main__":
    main()
