"""Data models for WMT24++ instruction-following MT evaluation."""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from common.bouquet_utils import get_lang_name


INSTRUCTIONS_PATH = Path("datasets/wmt24pp_instructions.jsonl")


def get_instructions_path(instruction_type: str = "full") -> Path:
    """Path to the WMT24++ instructions file for a given instruction type.

    ``full`` -> the curated datasets/wmt24pp_instructions.jsonl.
    Anything else -> datasets/wmt24pp_instructions_{instruction_type}.jsonl
    (e.g. ``self-para`` -> wmt24pp_instructions_self-para.jsonl).
    """
    if instruction_type == "full":
        return INSTRUCTIONS_PATH
    return INSTRUCTIONS_PATH.with_name(f"wmt24pp_instructions_{instruction_type}.jsonl")


class Wmt24ppFewShotExample(BaseModel):
    en: str
    instruction: str
    tgt: str


class Wmt24ppInstruction(BaseModel):
    """Language-agnostic WMT24++ instruction (generated from English source + domain)."""
    segment_key: str  # "{document_id}_{segment_id}"
    source: str  # English source text
    domain: str
    document_id: str
    user_instruction: str

    @classmethod
    def load(cls, instruction_type: str = "full") -> dict[str, "Wmt24ppInstruction"]:
        """Load instructions and return a dict keyed by segment_key."""
        path = get_instructions_path(instruction_type)
        if not path.exists():
            raise FileNotFoundError(
                f"Instructions file not found: {path}\n"
                f"For 'full', run python -m instructions.generate_wmt24pp; "
                f"for 'self-*', run python -m self_instruct.wmt24pp."
            )
        instructions = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                inst = cls.model_validate_json(line)
                instructions[inst.segment_key] = inst
        return instructions


@dataclass
class Wmt24ppDocumentGroup:
    """A group of Wmt24ppInstructionRow sentences belonging to the same document."""
    document_id: str
    rows: list  # list[Wmt24ppInstructionRow], sorted by segment_id

    @staticmethod
    def group_by_document(rows: list) -> list["Wmt24ppDocumentGroup"]:
        groups: dict[str, list] = defaultdict(list)
        for row in rows:
            groups[row.document_id].append(row)
        result = []
        for doc_id in sorted(groups.keys()):
            sorted_rows = sorted(groups[doc_id], key=lambda r: r.segment_id)
            result.append(Wmt24ppDocumentGroup(document_id=doc_id, rows=sorted_rows))
        return result

    def get_marked_source(self, sentence_idx: int) -> str:
        """Build document text with [START_TRANSLATE]...[END_TRANSLATE] markers around the sentence at sentence_idx."""
        parts = []
        for i, row in enumerate(self.rows):
            if i == sentence_idx:
                parts.append(f"[START_TRANSLATE]{row.source}[END_TRANSLATE]")
            else:
                parts.append(row.source)
        return " ".join(parts)

    def get_numbered_source(self) -> str:
        """Build the document as an S1: .../S2: .../... numbered list (P2 strategy)."""
        return "\n".join(f"S{i+1}: {r.source}" for i, r in enumerate(self.rows))

    def assign_predictions(self, output: str):
        """Parse an S1:/S2: numbered translation and assign pred_text to each row."""
        parsed = {}
        for match in re.finditer(r"^\s*S(\d+)\s*:\s*(.*)", output, re.MULTILINE):
            parsed[int(match.group(1))] = match.group(2).strip()
        for i, row in enumerate(self.rows):
            num = i + 1
            if num in parsed:
                row.pred_text = parsed[num]
            else:
                print(f"Warning: No translation for S{num} in {self.document_id}")
                row.pred_text = ""


class Wmt24ppInstructionRow(BaseModel):
    """A WMT24++ segment with generated user instruction and optional prediction."""
    segment_key: str
    lang: str  # e.g. "en-fr_FR"
    domain: str
    document_id: str
    segment_id: int
    source: str  # English source
    target: str  # Reference translation
    user_instruction: str
    pred_text: Optional[str] = None

    @classmethod
    def load_for_lang(cls, lang: str, limit: Optional[int] = None,
                      instruction_type: str = "full") -> List["Wmt24ppInstructionRow"]:
        """Load WMT24++ segments for a language and merge with instructions."""
        from common.trans_utils import Wmt24ppSegment

        instructions = Wmt24ppInstruction.load(instruction_type)
        segments = Wmt24ppSegment.load(lang)

        rows = []
        for seg in segments:
            if seg.domain == "canary":
                continue
            key = f"{seg.document_id}_{seg.segment_id}"
            if key not in instructions:
                print(f"Warning: No instruction for {key}, skipping")
                continue
            inst = instructions[key]
            rows.append(cls(
                segment_key=key,
                lang=seg.lang,
                domain=seg.domain,
                document_id=seg.document_id,
                segment_id=seg.segment_id,
                source=seg.source,
                target=seg.target or "",
                user_instruction=inst.user_instruction,
            ))
            if limit is not None and len(rows) >= limit:
                break
        return rows

    @classmethod
    def get_savepath(
        cls, model: str, lang: str, use_instructions: bool,
        enable_thinking: bool = False, paragraph_level: bool = False,
        numbered_lists: bool = False, instruction_type: str = "full",
    ) -> Path:
        model_name = model.split("/")[-1]
        suffix = "with_instructions" if use_instructions else "without_instructions"
        if instruction_type != "full":
            suffix += f"_{instruction_type}"
        if paragraph_level:
            suffix += "_paragraph"
        if numbered_lists:
            suffix += "_numbered"
        if enable_thinking:
            suffix += "_thinking"
        return Path(f"translation_results/wmt24pp_vllm/wmt24pp_{lang}_{model_name}_{suffix}.jsonl")

    @classmethod
    def save(cls, rows: List["Wmt24ppInstructionRow"], filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(row.model_dump_json() + "\n")

    @classmethod
    def load(cls, filepath: Path) -> List["Wmt24ppInstructionRow"]:
        rows = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(cls.model_validate_json(line))
        return rows

    def get_messages_in_context(
        self, marked_source: str, use_instructions: bool, few_shot_examples: List,
    ) -> List[dict]:
        """Build chat messages for sentence-in-context (paragraph-level) translation.

        ``marked_source`` is the full document text with [START_TRANSLATE]...[END_TRANSLATE]
        markers around the sentence to translate. The model translates only the marked
        sentence; the rest is context. ``few_shot_examples`` are sentence-in-context
        examples whose ``.en`` field is itself a marked paragraph.
        """
        tgt_locale = self.lang.split("-", 1)[1]
        lang_code = tgt_locale.split("_")[0]
        import pycountry
        py_lang = pycountry.languages.get(alpha_2=lang_code)
        lang_name = py_lang.name if py_lang else tgt_locale

        if use_instructions:
            system = (
                f"You are an expert translator. You will be given a document in English "
                f"with one sentence wrapped in [START_TRANSLATE]...[END_TRANSLATE] markers. "
                f"Translate ONLY the marked sentence to {lang_name}. "
                f"Use the surrounding sentences as context but do not translate them. "
                f"Follow the user's instruction about tone, style, and context. "
                f"Give only the translation, no extra commentary."
            )
        else:
            system = (
                f"You will be given a document in English with one sentence wrapped "
                f"in [START_TRANSLATE]...[END_TRANSLATE] markers. Translate ONLY the "
                f"marked sentence to {lang_name}. Use the surrounding sentences as context "
                f"but do not translate them. Give only the translation, no extra commentary."
            )

        messages = [{"role": "system", "content": system}]

        for ex in few_shot_examples:
            if use_instructions and ex.instruction:
                messages.append({"role": "user", "content": f"Instruction: {ex.instruction}\n\n{ex.en}"})
            else:
                messages.append({"role": "user", "content": ex.en})
            messages.append({"role": "assistant", "content": ex.tgt})

        if use_instructions:
            messages.append({
                "role": "user",
                "content": f"Instruction: {self.user_instruction}\n\n{marked_source}",
            })
        else:
            messages.append({"role": "user", "content": marked_source})

        return messages

    def get_messages(self, use_instructions: bool, few_shot_examples: List) -> List[dict]:
        """Build chat messages for translation."""
        # Extract target language name from lang code (e.g. "en-fr_FR" -> "French")
        # WMT24++ uses locale codes like "fr_FR", map to language name
        tgt_locale = self.lang.split("-", 1)[1]  # "fr_FR"
        lang_code = tgt_locale.split("_")[0]  # "fr"
        import pycountry
        py_lang = pycountry.languages.get(alpha_2=lang_code)
        lang_name = py_lang.name if py_lang else tgt_locale

        if use_instructions:
            system = f"""You are an expert translator. Translate the English text to {lang_name}.
Follow the user's instruction about tone, style, and context.
Give only the translation, no extra commentary."""

            messages = [{"role": "system", "content": system}]

            for ex in few_shot_examples:
                messages.append({
                    "role": "user",
                    "content": f"Instruction: {ex.instruction}\n\nText: {ex.en}"
                })
                messages.append({"role": "assistant", "content": ex.tgt})

            messages.append({
                "role": "user",
                "content": f"Instruction: {self.user_instruction}\n\nText: {self.source}"
            })
        else:
            system = f"""Translate to {lang_name}. Give only the translation, no extra commentary."""

            messages = [{"role": "system", "content": system}]

            for ex in few_shot_examples:
                messages.append({"role": "user", "content": ex.en})
                messages.append({"role": "assistant", "content": ex.tgt})

            messages.append({"role": "user", "content": self.source})

        return messages
