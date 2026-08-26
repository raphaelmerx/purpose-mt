"""Teacher translation step for PEFT distillation (paper Appendix I, step 3).

For each SMOL English sentence and its generated instruction, the Gemma-3-27B
teacher produces a translation conditioned on both. The resulting
(source, instruction, teacher-translation) triples are turned into SFT training
data by `distill/prepare_sft.py`.

The teacher run used for the paper generates a single greedy candidate:

    python -m distill.translate_teacher --lang ukr_Cyrl \
        --model google/gemma-3-27b-it --n-candidates 1

The `--n-candidates` / temperature knobs are retained from the sampling-based
variant of this script; `prepare_sft.py` consumes `candidates[0]`.
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from translate.bouquet_vllm import FewShotExample
from common.bouquet_utils import get_lang_name

DEFAULT_LANG = "ukr_Cyrl"
NUM_CANDIDATES = 5
TEMPERATURE = 0.8
DEFAULT_MODEL = "google/gemma-3-27b-it"


def build_system_prompt(lang_name: str) -> str:
    return (
        f"You are an expert translator. Translate the English text to {lang_name}.\n"
        "Follow the user's instruction about tone, style, and context.\n"
        "Give only the translation, no extra commentary."
    )


class TeacherInputRow(BaseModel):
    """One SMOL source sentence with its generated instruction."""
    key: str
    en_paragraph: str
    user_instruction: str


class TranslationCandidateRow(BaseModel):
    """One source sentence with its N teacher translations."""
    key: str
    en_paragraph: str
    user_instruction: str
    model: str
    candidates: List[str]

    # -- I/O helpers ----------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> List["TranslationCandidateRow"]:
        rows: List[TranslationCandidateRow] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(cls.model_validate_json(line))
        return rows

    @classmethod
    def save(cls, rows: List["TranslationCandidateRow"], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(row.model_dump_json() + "\n")

    @classmethod
    def get_output_path(cls, output_dir: Path, model: str) -> Path:
        model_name = model.split("/")[-1]
        return output_dir / f"candidates_{model_name}.jsonl"

    # -- Prompt building ------------------------------------------------------

    @classmethod
    def build_messages(
        cls,
        row: TeacherInputRow,
        few_shot_examples: List[FewShotExample],
        system_prompt: str,
    ) -> List[dict]:
        """Build the full chat message list for a single translation request."""
        messages: List[dict] = [{"role": "system", "content": system_prompt}]

        for ex in few_shot_examples:
            messages.append({
                "role": "user",
                "content": f"Instruction: {ex.instruction}\n\nText: {ex.en}",
            })
            messages.append({"role": "assistant", "content": ex.tgt})

        messages.append({
            "role": "user",
            "content": f"Instruction: {row.user_instruction}\n\nText: {row.en_paragraph}",
        })
        return messages

    # -- Construction from vLLM output ----------------------------------------

    @classmethod
    def from_vllm_output(
        cls,
        row: TeacherInputRow,
        output,  # vllm.RequestOutput
        model: str,
    ) -> "TranslationCandidateRow":
        return cls(
            key=row.key,
            en_paragraph=row.en_paragraph,
            user_instruction=row.user_instruction,
            model=model,
            candidates=[o.text.strip() for o in output.outputs],
        )

    # -- Pretty-printing ------------------------------------------------------

    def print_sample(self) -> None:
        print(f"\n  [{self.key}]")
        print(f"  EN:    {self.en_paragraph[:100]}...")
        print(f"  Instr: {self.user_instruction[:80]}...")
        for i, cand in enumerate(self.candidates):
            print(f"    candidate {i}: {cand[:100]}")


# -- CLI ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate teacher translations for SMOL sentences using vLLM."
    )
    parser.add_argument(
        "--lang", type=str, default=DEFAULT_LANG,
        help=f"Target language code (default: {DEFAULT_LANG})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"vLLM teacher model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of sentences to translate (default: all)",
    )
    parser.add_argument(
        "--ignore-cache", action="store_true",
        help="Ignore existing results and re-translate",
    )
    parser.add_argument(
        "--input-path", type=Path, default=None,
        help="Override input JSONL path (default: derived from --lang)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: derived from --lang)",
    )
    parser.add_argument(
        "--n-candidates", type=int, default=NUM_CANDIDATES,
        help=f"Candidates per sentence (default: {NUM_CANDIDATES}; the paper uses 1)",
    )
    return parser.parse_args()


# -- Main ---------------------------------------------------------------------


def main() -> None:
    from vllm import LLM, SamplingParams
    from distill.generate_instructions import SmolInstructionRow

    args = parse_args()
    lang_name = get_lang_name(args.lang)
    system_prompt = build_system_prompt(lang_name)

    input_path = args.input_path or Path("datasets/smol/en_instructions.jsonl")
    output_dir = args.output_dir or Path(f"datasets/smol/{args.lang}_candidates")

    print(f"Lang: {args.lang} ({lang_name})")
    print(f"Model: {args.model}")
    print(f"Input path: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Candidates per sentence: {args.n_candidates}")
    print(f"Limit: {args.limit}")

    output_path = TranslationCandidateRow.get_output_path(output_dir, args.model)

    # --- Cache check ---------------------------------------------------------
    if output_path.exists() and not args.ignore_cache:
        print(f"\nResults already exist at {output_path}")
        print("Use --ignore-cache to re-translate")
        rows = TranslationCandidateRow.load(output_path)
        print(f"Loaded {len(rows)} existing rows")
        for row in rows[:3]:
            row.print_sample()
        return

    # --- Load data -----------------------------------------------------------
    few_shot_examples = FewShotExample.build_examples_for_split(args.lang)
    print(f"\nFew-shot examples ({len(few_shot_examples)} from BOUQuET {args.lang}):")
    for ex in few_shot_examples:
        print(f"  EN:    {ex.en}")
        print(f"  TGT:   {ex.tgt}")
        print(f"  Instr: {ex.instruction}")
        print()

    smol_rows = SmolInstructionRow.load(input_path)
    if args.limit:
        smol_rows = smol_rows[:args.limit]
    instruction_rows = [
        TeacherInputRow(
            key=r.key,
            en_paragraph=r.en_text,
            user_instruction=r.user_instruction,
        )
        for r in smol_rows
    ]
    print(f"Sentences to translate: {len(instruction_rows)}")

    # --- Build prompts -------------------------------------------------------
    all_messages = [
        TranslationCandidateRow.build_messages(row, few_shot_examples, system_prompt)
        for row in instruction_rows
    ]

    print("\nSample prompt:")
    for msg in all_messages[0]:
        print(f"  [{msg['role']}]: {msg['content'][:120]}")

    # --- Translate with vLLM -------------------------------------------------
    print(f"\nLoading model {args.model}...")
    llm = LLM(args.model, max_model_len=4096)

    sampling_params = SamplingParams(
        n=args.n_candidates,
        temperature=TEMPERATURE if args.n_candidates > 1 else 0.0,
        max_tokens=512,
    )

    print(f"Translating {len(all_messages)} sentences x {args.n_candidates} candidates...")
    outputs = llm.chat(all_messages, sampling_params=sampling_params)

    # --- Build & save results ------------------------------------------------
    result_rows = [
        TranslationCandidateRow.from_vllm_output(row, output, args.model)
        for row, output in zip(instruction_rows, outputs)
    ]

    TranslationCandidateRow.save(result_rows, output_path)
    print(f"\nSaved {len(result_rows)} rows to {output_path}")

    print("\nSample outputs:")
    for result_row in result_rows[:3]:
        result_row.print_sample()


if __name__ == "__main__":
    main()
