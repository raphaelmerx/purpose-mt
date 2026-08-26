"""
Generate context-only and purpose-only instruction variants for the ablation study.

Both variants see the same input (domain, par_comment, tags, tgt_text) as the full
instruction generation. The difference is in the system prompt and few-shot examples,
which constrain the output type.

Usage:
    python -m instructions.generate_variants --split dev
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from pathlib import Path
from typing import List

from pydantic import BaseModel
from tqdm import tqdm

from translate.bouquet_vllm import BouquetInstruction
from common.trans_utils import cached_completion


# --- Few-shot examples (same 3 input rows as instructions/generate_bouquet.py) ---

FEW_SHOT_INPUTS = [
    {
        "domain": "web misc.",
        "par_comment": "This paragraph exemplifies a job posting on a social media platform or online bulletin board. It contains informal language identifiable by interrogative and exclamatory sentences. It also includes a common strategy for gender neutral language. The second person is generic and refers to anyone interested in the job post. The first person plural is exclusive and refers to the company",
        "tags": "<na>",
        "tgt_text": "Are you interested in the technology sector?"
    },
    {
        "domain": "social posts",
        "par_comment": "Blog of a movie review",
        "tags": "phrase:normalisation",
        "tgt_text": "However, as a film with a local Hong Kong theme, it has achieved a certain balance between niche characteristics and the market."
    },
    {
        "domain": "narration",
        "par_comment": "Imaginary description of a scene in a novel (probably set in the past) in which a young woman is compelled by her father to go to a party where she will meet a young man who wants to approach her. Diane is of the feminine grammatical gender; Aymeric of the masculine grammatical gender.",
        "tags": "clause:comparative, clause:preposition_phrase, phrase:verb_tense, phrase:adjective_order",
        "tgt_text": "The reception hall where Diane entered was bathed in cold light and had been decked with sparkling colors without any regard for harmony."
    },
]

# Context-only: describe the situation, no translation directives
CONTEXT_OUTPUTS = [
    "This is a job posting on a tech company's social media page.",
    "This is a blog post reviewing a Hong Kong movie.",
    "This is a passage from a historical novel, describing a young woman entering a reception hall.",
]

# Purpose-only: pragmatic directives only, no situational framing
PURPOSE_OUTPUTS = [
    "Use informal, engaging language with a friendly and inviting tone. Keep the language gender-neutral.",
    "Use a conversational yet analytical tone. Write for general readers.",
    "Use elegant, descriptive prose with attention to atmosphere and imagery. Maintain a formal third-person narrative voice.",
]

CONTEXT_SYSTEM_PROMPT = """You are an expert at describing translation contexts. Given metadata about a text segment (domain, context description, linguistic tags, and a reference translation), generate a brief description of the situational context ONLY.

Describe:
- What kind of text this is (e.g., job posting, novel, conversation)
- Who is speaking or writing, and to whom (if known)
- The medium or setting (e.g., social media, book, chat)

Do NOT include any translation directives about tone, style, register, or how to translate. Just describe the situation factually.

Output ONLY the context description, nothing else. Keep it to 1-2 sentences."""

PURPOSE_SYSTEM_PROMPT = """You are an expert at creating translation directives. Given metadata about a text segment (domain, context description, linguistic tags, and a reference translation), generate ONLY pragmatic translation directives.

Specify:
- What tone to use (e.g., formal, casual, sarcastic)
- What style or register to adopt (e.g., literary, conversational, technical)
- Any specific linguistic choices (e.g., gender-neutral language, specific pronouns)

Do NOT describe the situational context, genre, medium, or participants. Just give directives about how to translate.

Output ONLY the directives, nothing else. Keep it to 1-2 sentences."""

EVAL_MODEL = "gemini/gemini-3-flash-preview"


def format_user_message(domain: str, par_comment: str, tags: str, tgt_text: str) -> str:
    return f"""Domain: {domain}
Context: {par_comment}
Linguistic tags: {tags}
Reference translation: {tgt_text}

Generate the instruction for translating this text."""


def build_messages(
    inst: BouquetInstruction,
    system_prompt: str,
    few_shot_outputs: List[str],
) -> List[dict]:
    messages = [{"role": "system", "content": system_prompt}]

    for inp, out in zip(FEW_SHOT_INPUTS, few_shot_outputs):
        messages.append({
            "role": "user",
            "content": format_user_message(inp["domain"], inp["par_comment"], inp["tags"], inp["tgt_text"]),
        })
        messages.append({"role": "assistant", "content": out})

    messages.append({
        "role": "user",
        "content": format_user_message(inst.domain, inst.par_comment, inst.tags, inst.tgt_text),
    })
    return messages


def generate_variant(
    instructions: List[BouquetInstruction],
    variant: str,
) -> List[BouquetInstruction]:
    """Generate a variant (context or purpose) for all instructions."""
    if variant == "context":
        system_prompt = CONTEXT_SYSTEM_PROMPT
        few_shot_outputs = CONTEXT_OUTPUTS
    elif variant == "purpose":
        system_prompt = PURPOSE_SYSTEM_PROMPT
        few_shot_outputs = PURPOSE_OUTPUTS
    else:
        raise ValueError(f"Unknown variant: {variant}")

    results = []
    for inst in tqdm(instructions, desc=f"Generating {variant} variants"):
        messages = build_messages(inst, system_prompt, few_shot_outputs)
        response = cached_completion(model=EVAL_MODEL, messages=messages)

        results.append(BouquetInstruction(
            uniq_id=inst.uniq_id,
            tgt_text=inst.tgt_text,
            domain=inst.domain,
            par_comment=inst.par_comment,
            tags=inst.tags,
            register=inst.register,
            user_instruction=response.strip(),
        ))

    return results


def save_instructions(instructions: List[BouquetInstruction], split: str, variant: str):
    filepath = Path(f"datasets/bouquet_instructions_{split}_{variant}.jsonl")
    with open(filepath, "w", encoding="utf-8") as f:
        for inst in instructions:
            f.write(inst.model_dump_json() + "\n")
    print(f"Saved {len(instructions)} {variant} instructions to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate instruction variants for ablation study")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    # Load full instructions
    full_instructions = BouquetInstruction.load(args.split)
    instruction_list = list(full_instructions.values())
    print(f"Loaded {len(instruction_list)} instructions for {args.split} split")

    for variant in ["context", "purpose"]:
        print(f"\n{'='*50}")
        print(f"Generating {variant}-only variants...")
        results = generate_variant(instruction_list, variant)
        save_instructions(results, args.split, variant)

        # Print a few examples for spot-checking
        print(f"\nSample {variant}-only instructions:")
        for inst in results[:3]:
            print(f"  [{inst.domain}] {inst.tgt_text[:60]}...")
            print(f"    Full: {full_instructions[inst.uniq_id].user_instruction}")
            print(f"    {variant.title()}: {inst.user_instruction}")
            print()


if __name__ == "__main__":
    main()
