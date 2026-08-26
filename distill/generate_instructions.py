"""Generate translation instructions for SMOL English source sentences.

Uses Gemini via litellm to generate diverse instructions, biased toward
informal/casual registers to complement MADLAD's formal bias.

Usage:
    python -m distill.generate_instructions
    python -m distill.generate_instructions --limit 100
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from pydantic import BaseModel
from tqdm import tqdm

from distill.load_smol import SmolSourceRow
from common.trans_utils import cached_completion


INPUT_PATH = Path("datasets/smol/en_sources.jsonl")
OUTPUT_PATH = Path("datasets/smol/en_instructions.jsonl")

SYSTEM_PROMPT = """You are an expert at writing translation instructions.

You will be given an English sentence. Write a concise, language-agnostic instruction that describes HOW to translate it (register, tone, audience, style). Do NOT name or reference any specific target language in the instruction — the instruction should be reusable regardless of the target language.

Requirements:
1) Infer the appropriate setting/register from the text
2) Mention the intended audience or use case
3) Give style/tone guidance

The texts come from diverse domains including social media, blogs, conversations, web content, and narration. Match the instruction register to the text's natural register — casual texts should get casual instructions.

Output ONLY the instruction, nothing else. Keep it to 1-2 sentences."""

FEW_SHOT_EXAMPLES = [
    {
        "en": "Have errands to run and the kids won't let you run them?",
        "instruction": "Translate as a casual, engaging social media post aimed at a general audience, particularly parents. The tone should be relatable and conversational, using an impersonal 'you'.",
    },
    {
        "en": "You can book your appointment on our page here.",
        "instruction": "Translate as formal web content on an information page. Address the reader directly and politely, maintaining a helpful and informative tone for general website users.",
    },
    {
        "en": "She sat there thinking what could he have possibly meant?",
        "instruction": "Translate as a passage from a novel, conveying a character's internal thought process. Maintain a reflective and natural tone appropriate for literary fiction.",
    },
]


class SmolInstructionRow(SmolSourceRow):
    user_instruction: str
    instruction_model: str

    @classmethod
    def save(cls, rows: List["SmolInstructionRow"], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(row.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: Path) -> List["SmolInstructionRow"]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(cls.model_validate_json(line))
        return rows


def build_messages(en_text: str, system_prompt: str) -> List[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["en"]})
        messages.append({"role": "assistant", "content": ex["instruction"]})
    messages.append({"role": "user", "content": en_text})
    return messages


def generate_instruction(en_text: str, model: str, system_prompt: str) -> str:
    messages = build_messages(en_text, system_prompt)
    return cached_completion(model=model, messages=messages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate translation instructions for SMOL sources."
    )
    parser.add_argument(
        "--model", type=str, default="gemini/gemini-3-flash-preview",
        help="Gemini model for instruction generation",
    )
    parser.add_argument(
        "--lang", type=str, default="ukr_Cyrl",
        help="BOUQuET target language code (default: ukr_Cyrl)",
    )
    parser.add_argument(
        "--output-path", type=Path, default=None,
        help="Override output path (default: derived from --lang; ukr_Cyrl uses the legacy smol_en_instructions.jsonl path)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lang_name = get_lang_name(args.lang)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=lang_name)
    if args.output_path:
        output_path = args.output_path
    elif args.lang == "ukr_Cyrl":
        output_path = DEFAULT_OUTPUT_PATH
    else:
        output_path = Path(f"datasets/smol/en_instructions_{args.lang}.jsonl")

    if output_path.exists() and not args.overwrite:
        print(f"Output exists: {output_path}. Use --overwrite.")
        return

    print(f"Target language: {args.lang} ({lang_name})")
    print(f"Output: {output_path}")

    sources = SmolSourceRow.load(INPUT_PATH)
    if args.limit:
        sources = sources[:args.limit]
    print(f"Loaded {len(sources)} source sentences")

    results: List[SmolInstructionRow] = []
    failures = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_row = {
            executor.submit(generate_instruction, row.en_text, args.model, system_prompt): row
            for row in sources
        }
        for future in tqdm(as_completed(future_to_row), total=len(sources), desc="Generating instructions"):
            row = future_to_row[future]
            try:
                instruction = future.result()
                if instruction.strip():
                    results.append(SmolInstructionRow(
                        key=row.key,
                        en_text=row.en_text,
                        source=row.source,
                        user_instruction=instruction.strip(),
                        instruction_model=args.model,
                    ))
                else:
                    failures += 1
            except Exception as e:
                print(f"Error for {row.key}: {e}")
                failures += 1

    # Sort by key to maintain order
    results.sort(key=lambda r: r.key)

    SmolInstructionRow.save(results, output_path)
    print(f"Saved {len(results)} instructions to {output_path}")
    if failures:
        print(f"Failures: {failures}")


if __name__ == "__main__":
    main()
