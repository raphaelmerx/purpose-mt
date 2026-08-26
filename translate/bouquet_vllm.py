"""
Translate BOUQuET from English (tgt_text) to target language (src_text, default fra_Latn).
Explores whether LLM translation improves when given user_instructions about context.

Usage:
    python -m translate.bouquet_vllm --use-user-instructions --tgt-lang ind_Latn --model google/gemma-3-4b-it
    python -m translate.bouquet_vllm  --tgt-lang fra_Latn --model google/gemma-3-27b-it # without instructions
    python -m translate.bouquet_vllm --use-user-instructions --semantic-fewshot --tgt-lang fra_Latn  # semantic few-shot
    python -m translate.bouquet_vllm --paragraph-level --tgt-lang fra_Latn --model google/gemma-3-4b-it  # paragraph-level
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pycountry
from pydantic import BaseModel
from sacrebleu import corpus_chrf

from common.bouquet_utils import BouquetRow, get_lang_name


def parse_args():
    parser = argparse.ArgumentParser(description="Translate BOUQuET with vLLM")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it", 
                        help="LLM to use for translation")
    parser.add_argument("--tgt-lang", type=str, default="fra_Latn",
                        help="Target language code (matches BOUQuET src_lang)")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"],
                        help="Dataset split to use")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of rows to translate (None = all)")
    parser.add_argument("--use-user-instructions", action="store_true",
                        help="Include user_instructions in the prompt")
    parser.add_argument("--ignore-cache", action="store_true",
                        help="Ignore existing results and re-translate")
    parser.add_argument("--semantic-fewshot", action="store_true",
                        help="Use semantic similarity to select few-shot examples per row")
    parser.add_argument("--paragraph-level", action="store_true",
                        help="Translate full paragraphs at once instead of individual sentences")
    parser.add_argument("--numbered-lists", action="store_true",
                        help="Use S1:/S2: numbered sentence lists for paragraph-level translation (requires --paragraph-level)")
    parser.add_argument("--strict-prompt", action="store_true",
                        help="Add strict language constraint to system prompt")
    parser.add_argument("--instruction-type", type=str, default="full",
                        help="Instruction variant; loads datasets/bouquet_instructions_{split}_{instruction_type}.jsonl "
                             "(default: full -> bouquet_instructions_{split}.jsonl)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking/reasoning mode (Gemma 4). Saves reasoning traces alongside translations.")
    return parser.parse_args()


class BouquetInstruction(BaseModel):
    """Language-agnostic BOUQuET instruction generated from metadata."""
    uniq_id: str
    tgt_text: str  # English source text
    domain: str
    par_comment: str
    tags: str
    register: str
    user_instruction: str

    @classmethod
    def get_data_path(cls, split: str, instruction_type: str = "full") -> Path:
        if instruction_type == "full":
            return Path(f"datasets/bouquet_instructions_{split}.jsonl")
        return Path(f"datasets/bouquet_instructions_{split}_{instruction_type}.jsonl")

    @classmethod
    def load(cls, split: str, instruction_type: str = "full") -> dict[str, "BouquetInstruction"]:
        """Load instructions and return a dict keyed by uniq_id."""
        filepath = cls.get_data_path(split, instruction_type)
        if not filepath.exists():
            raise FileNotFoundError(
                f"Instructions file not found: {filepath}\n"
                f"Run `python -m instructions.generate_bouquet --split {split}` first, "
                f"or fetch the released instructions with `python -m data.download`."
            )
        instructions = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                inst = cls.model_validate_json(line)
                instructions[inst.uniq_id] = inst
        return instructions


class FewShotExample(BaseModel):
    en: str
    tgt: str
    instruction: str

    @classmethod
    def build_examples_for_split(cls, lang: str, split: str = "dev") -> List["FewShotExample"]:
        """Build 3 few-shot examples for a given split.

        To avoid sampling from the same split, we sample from `test` for `dev`, and from `dev` for `test`.
        
        Args:
            lang: Target language code (e.g., "fra_Latn")
            split: Dataset split (default "dev")
        Returns:
            List of example dicts with keys: "en", "tgt", "instruction"
        """
        other_split = "test" if split == "dev" else "dev"
        row_ids = []
        if split == "test":
            row_ids = ['P182-S3', 'P115-S1', 'P426-S3']
        elif split == "dev":
            row_ids = ['P131-S2', 'P164-S2', 'P427-S1']
        instructions = BouquetInstruction.load(other_split)
        bouquet_rows = [row for row in BouquetRow.load(lang, other_split) if row.uniq_id in row_ids]

        examples = []
        for row in bouquet_rows:
            inst = instructions[row.uniq_id]
            examples.append(cls(
                en=row.tgt_text,
                tgt=row.src_text,
                instruction=inst.user_instruction
            ))
        
        return examples


    @classmethod
    def build_numbered_list_examples_for_split(cls, lang: str, split: str = "dev") -> list["FewShotExample"]:
        """Build few-shot examples using S1:/S2: numbered paragraph format.

        Each example shows a full paragraph as numbered sentences, and the
        assistant reply is the full paragraph translated in the same format.
        """
        other_split = "test" if split == "dev" else "dev"
        if split == "test":
            par_ids = ["P001", "P020"]
        elif split == "dev":
            par_ids = ["P058", "P060"]

        bouquet_rows = BouquetRow.load(lang, other_split)
        par_groups: dict[str, list[BouquetRow]] = defaultdict(list)
        for row in bouquet_rows:
            par_groups[row.par_id].append(row)

        examples = []
        for par_id in par_ids:
            if par_id not in par_groups:
                continue
            par_rows = sorted(par_groups[par_id], key=lambda r: int(r.uniq_id.split("-S")[1]))
            en_numbered = "\n".join(f"S{i+1}: {r.tgt_text}" for i, r in enumerate(par_rows))
            tgt_numbered = "\n".join(f"S{i+1}: {r.src_text}" for i, r in enumerate(par_rows))
            examples.append(cls(en=en_numbered, tgt=tgt_numbered, instruction=""))

        return examples

    @classmethod
    def build_paragraph_examples_for_split(cls, lang: str, split: str = "dev") -> list["FewShotExample"]:
        """Build sentence-in-context few-shot examples from opposite split.

        Each example shows a full paragraph with <<<markers>>> around one sentence,
        and the assistant reply is just that sentence's translation. Picks 3 examples
        varying the marked position (first, middle, last).
        """
        other_split = "test" if split == "dev" else "dev"
        # Hardcoded (paragraph_id, sentence_number) for reproducibility
        if split == "test":
            examples_spec = [("P001", 1), ("P001", 3), ("P020", 2)]
        elif split == "dev":
            examples_spec = [("P058", 1), ("P058", 3), ("P060", 2)]

        instructions = BouquetInstruction.load(other_split)
        bouquet_rows = BouquetRow.load(lang, other_split)

        # Group rows by paragraph
        par_groups: dict[str, list[BouquetRow]] = defaultdict(list)
        for row in bouquet_rows:
            par_groups[row.par_id].append(row)

        examples = []
        for par_id, sent_num in examples_spec:
            if par_id not in par_groups:
                continue
            par_rows = sorted(par_groups[par_id], key=lambda r: int(r.uniq_id.split("-S")[1]))
            target_idx = sent_num - 1  # 0-indexed

            # Build marked paragraph
            parts = []
            for i, r in enumerate(par_rows):
                if i == target_idx:
                    parts.append(f"[START_TRANSLATE]{r.tgt_text}[END_TRANSLATE]")
                else:
                    parts.append(r.tgt_text)
            marked_en = " ".join(parts)

            target_row = par_rows[target_idx]
            inst = instructions.get(target_row.uniq_id)
            examples.append(cls(
                en=marked_en,
                tgt=target_row.src_text,
                instruction=inst.user_instruction if inst else "",
            ))

        return examples


@dataclass
class ParagraphGroup:
    """A group of BouquetInstructionRow sentences belonging to the same paragraph."""
    par_id: str
    rows: list  # list[BouquetInstructionRow], sorted by sentence number

    @staticmethod
    def group_by_paragraph(rows: list) -> list["ParagraphGroup"]:
        """Group BouquetInstructionRow objects by paragraph ID, sorted by sentence number."""
        groups: dict[str, list] = defaultdict(list)
        for row in rows:
            pid = row.uniq_id.split("-S")[0]
            groups[pid].append(row)
        result = []
        for pid in sorted(groups.keys()):
            sorted_rows = sorted(groups[pid], key=lambda r: int(r.uniq_id.split("-S")[1]))
            result.append(ParagraphGroup(par_id=pid, rows=sorted_rows))
        return result

    def get_numbered_source(self) -> str:
        """Build paragraph as S1: sent1\\nS2: sent2\\n... format."""
        return "\n".join(f"S{i+1}: {r.tgt_text}" for i, r in enumerate(self.rows))

    def get_numbered_target(self) -> str:
        """Build paragraph translations as S1: trans1\\nS2: trans2\\n... format."""
        return "\n".join(f"S{i+1}: {r.src_text}" for i, r in enumerate(self.rows))

    def assign_predictions(self, output: str):
        """Parse S1:/S2: numbered output and assign pred_text to each row."""
        parsed = {}
        for match in re.finditer(r"^\s*S(\d+)\s*:\s*(.*)", output, re.MULTILINE):
            num = int(match.group(1))
            text = match.group(2).strip()
            parsed[num] = text
        for i, row in enumerate(self.rows):
            sentence_num = i + 1
            if sentence_num in parsed:
                row.pred_text = parsed[sentence_num]
            else:
                print(f"Warning: No translation for S{sentence_num} in {self.par_id}")
                row.pred_text = ""

    def get_marked_source(self, sentence_idx: int) -> str:
        """Build paragraph text with [START_TRANSLATE]...[END_TRANSLATE] markers around the sentence at sentence_idx."""
        parts = []
        for i, row in enumerate(self.rows):
            if i == sentence_idx:
                parts.append(f"[START_TRANSLATE]{row.tgt_text}[END_TRANSLATE]")
            else:
                parts.append(row.tgt_text)
        return " ".join(parts)


class SemanticExampleSelector:
    """Selects few-shot examples based on semantic similarity of instructions."""
    
    def __init__(
        self,
        lang: str,
        split: str,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Path = Path("datasets/embeddings_cache"),
    ):
        """Initialize the selector with precomputed embeddings.
        
        Args:
            lang: Target language code (e.g., "fra_Latn")
            split: Dataset split we're translating ("dev" or "test")
            model_name: Sentence-transformers model name
            cache_dir: Directory to cache embeddings
        """
        from sentence_transformers import SentenceTransformer
        
        self.lang = lang
        self.split = split
        self.other_split = "test" if split == "dev" else "dev"
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load sentence transformer model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Load example pool from the other split
        self.instructions = BouquetInstruction.load(self.other_split)
        self.bouquet_rows = {
            row.uniq_id: row 
            for row in BouquetRow.load(lang, self.other_split)
        }
        
        # Filter to rows that have both instruction and translation
        self.valid_ids = [
            uid for uid in self.instructions.keys() 
            if uid in self.bouquet_rows
        ]
        self.instruction_texts = [
            self.instructions[uid].user_instruction for uid in self.valid_ids
        ]
        
        # Load or compute embeddings
        self.embeddings = self._load_or_compute_embeddings()
        print(f"Loaded {len(self.valid_ids)} example candidates from {self.other_split} split")
    
    def _get_cache_path(self) -> Path:
        """Get the cache file path for embeddings."""
        # Cache is per-split and model, not per-language (instructions are language-agnostic)
        return self.cache_dir / f"embeddings_{self.other_split}_{self.model_name.replace('/', '_')}.pkl"
    
    def _load_or_compute_embeddings(self) -> np.ndarray:
        """Load embeddings from cache or compute them."""
        cache_path = self._get_cache_path()
        
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            # Verify cache matches current data
            if cached.get("valid_ids") == self.valid_ids:
                return cached["embeddings"]
            print("Cache invalidated (different IDs), recomputing...")
        
        print(f"Computing embeddings for {len(self.instruction_texts)} instructions...")
        embeddings = self.model.encode(
            self.instruction_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        
        # Save to cache
        with open(cache_path, "wb") as f:
            pickle.dump({"valid_ids": self.valid_ids, "embeddings": embeddings}, f)
        print(f"Saved embeddings to {cache_path}")
        
        return embeddings
    
    def get_similar_examples(self, query_instruction: str, k: int = 3) -> List[FewShotExample]:
        """Find k examples with most similar instructions.
        
        Args:
            query_instruction: The instruction to find similar examples for
            k: Number of examples to return
            
        Returns:
            List of FewShotExample objects
        """
        # Compute query embedding
        query_embedding = self.model.encode([query_instruction], convert_to_numpy=True)[0]
        
        # Compute cosine similarities
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = embeddings_norm @ query_norm
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Build FewShotExample objects
        examples = []
        for idx in top_indices:
            uid = self.valid_ids[idx]
            inst = self.instructions[uid]
            row = self.bouquet_rows[uid]
            examples.append(FewShotExample(
                en=row.tgt_text,
                tgt=row.src_text,
                instruction=inst.user_instruction,
            ))
        
        return examples


class BouquetInstructionRow(BaseModel):
    """A BOUQuET row with generated user instruction and optional prediction."""
    uniq_id: str
    src_lang: str
    tgt_lang: str
    src_text: str  # Target text (e.g., French) - ground truth
    tgt_text: str  # Source text (English) - to translate
    domain: str
    par_comment: str
    tags: str
    register: str
    user_instruction: str
    pred_text: Optional[str] = None  # Model prediction

    @classmethod
    def load_for_lang(cls, lang: str, split: str, limit: Optional[int] = None, instruction_type: str = "full") -> List["BouquetInstructionRow"]:
        """Load BOUQuET data for a language and merge with language-generic instructions."""
        from common.bouquet_utils import BouquetRow

        # Load language-generic instructions
        instructions = BouquetInstruction.load(split, instruction_type)
        
        # Load language-specific BOUQuET data
        bouquet_rows = BouquetRow.load(lang, split)
        
        rows = []
        for row in bouquet_rows:
            if row.uniq_id not in instructions:
                print(f"Warning: No instruction for {row.uniq_id}, skipping")
                continue
            inst = instructions[row.uniq_id]
            rows.append(cls(
                uniq_id=row.uniq_id,
                src_lang=row.src_lang,
                tgt_lang=row.tgt_lang,
                src_text=row.src_text,
                tgt_text=row.tgt_text,
                domain=inst.domain,
                par_comment=inst.par_comment,
                tags=inst.tags,
                register=inst.register,
                user_instruction=inst.user_instruction,
            ))
            if limit is not None and len(rows) >= limit:
                break
        return rows

    @classmethod
    def get_savepath(
        cls, model: str, lang: str, split: str, use_instructions: bool,
        semantic_fewshot: bool = False, paragraph_level: bool = False,
        numbered_lists: bool = False, instruction_type: str = "full",
        enable_thinking: bool = False,
    ) -> Path:
        model_name = model.split("/")[-1]
        suffix = "with_instructions" if use_instructions else "without_instructions"
        if semantic_fewshot:
            suffix += "_semantic"
        if paragraph_level:
            suffix += "_paragraph"
        if numbered_lists:
            suffix += "_numbered"
        if instruction_type != "full":
            suffix += f"_{instruction_type}"
        if enable_thinking:
            suffix += "_thinking"
        return Path(f"translation_results/bouquet_vllm/bouquet_{lang}_{split}_{model_name}_{suffix}.jsonl")

    @classmethod
    def save(cls, rows: List["BouquetInstructionRow"], filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(row.model_dump_json() + "\n")

    @classmethod
    def load(cls, filepath: Path) -> List["BouquetInstructionRow"]:
        """Load saved translation results from a JSONL file."""
        rows = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(cls.model_validate_json(line))
        return rows

    def get_messages(self, use_instructions: bool, few_shot_examples: List[FewShotExample], strict_prompt: bool = False) -> List[dict]:
        """Build chat messages for translation."""
        lang_name = get_lang_name(self.src_lang)

        if use_instructions:
            system = f"""You are an expert translator. Translate the English text to {lang_name}.
Follow the user's instruction about tone, style, and context.
Give only the translation, no extra commentary."""
            if strict_prompt:
                system = f"""You are an expert translator. Translate the English text to {lang_name}.
You MUST output ONLY in {lang_name}. NEVER output English or any other language.
Follow the user's instruction about tone, style, and context.
Give only the translation, no extra commentary."""
            
            messages = [{"role": "system", "content": system}]
            
            # Few-shot examples with instructions
            for ex in few_shot_examples:
                messages.append({
                    "role": "user",
                    "content": f"Instruction: {ex.instruction}\n\nText: {ex.en}"
                })
                messages.append({"role": "assistant", "content": ex.tgt})
            
            # Actual query with instruction
            messages.append({
                "role": "user",
                "content": f"Instruction: {self.user_instruction}\n\nText: {self.tgt_text}"
            })
        else:
            system = f"""Translate to {lang_name}. Give only the translation, no extra commentary."""
            
            messages = [{"role": "system", "content": system}]
            
            # Few-shot examples without instructions
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": ex.en})
                messages.append({"role": "assistant", "content": ex.tgt})
            
            # Actual query
            messages.append({"role": "user", "content": self.tgt_text})
        
        return messages

def compute_chrf(predictions: List[str], references: List[str]) -> float:
    """Compute ChrF++ score."""
    return corpus_chrf(predictions, [references], word_order=2).score


def evaluate_translations(rows: List[BouquetInstructionRow]) -> dict:
    """Compute ChrF++ scores overall and per domain."""
    # Filter rows with predictions
    valid_rows = [r for r in rows if r.pred_text is not None]
    
    if not valid_rows:
        return {"overall": 0.0, "by_domain": {}}
    
    # Overall score
    predictions = [r.pred_text or "" for r in valid_rows]
    references = [r.src_text for r in valid_rows]
    overall_chrf = compute_chrf(predictions, references)
    
    # Per-domain scores
    domains = set(r.domain for r in valid_rows)
    by_domain = {}
    for domain in sorted(domains):
        domain_rows = [r for r in valid_rows if r.domain == domain]
        domain_preds = [r.pred_text or "" for r in domain_rows]
        domain_refs = [r.src_text for r in domain_rows]
        by_domain[domain] = {
            "chrf": compute_chrf(domain_preds, domain_refs),
            "count": len(domain_rows)
        }
    
    return {"overall": overall_chrf, "by_domain": by_domain}


def resolve_vllm_model_and_lora(model_arg: str) -> tuple[str, Optional[dict]]:
    """Resolve CLI model arg into vLLM model + optional LoRA adapter metadata."""
    model_path = Path(model_arg)
    adapter_config_path = model_path / "adapter_config.json"
    if not adapter_config_path.exists():
        return model_arg, None

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            f"LoRA adapter config at {adapter_config_path} is missing "
            "'base_model_name_or_path'."
        )

    lora_rank = int(adapter_config.get("r", 16))
    lora_path = str(model_path.resolve())
    lora_name = model_path.name
    return base_model, {
        "name": lora_name,
        "path": lora_path,
        "rank": lora_rank,
        "base_model": base_model,
    }


def _build_numbered_list_messages(
    numbered_source: str,
    lang_name: str,
    few_shot_examples: list[FewShotExample],
) -> list[dict]:
    """Build chat messages for numbered-list paragraph translation.

    The numbered_source contains the full paragraph as S1: .../S2: .../... lines.
    The model should return translations in the same S1:/S2: format.
    """
    system = (
        f"Translate each sentence to {lang_name}. "
        f"Output translations in the same S1:, S2:, ... format. "
        f"Give only the translations, no extra commentary."
    )

    messages = [{"role": "system", "content": system}]

    for ex in few_shot_examples:
        messages.append({"role": "user", "content": ex.en})
        messages.append({"role": "assistant", "content": ex.tgt})

    messages.append({"role": "user", "content": numbered_source})
    return messages


def _build_sentence_in_context_messages(
    marked_source: str,
    row: BouquetInstructionRow,
    lang_name: str,
    few_shot_examples: list[FewShotExample],
    use_instructions: bool = False,
) -> list[dict]:
    """Build chat messages for sentence-in-context translation.

    The marked_source contains the full paragraph with <<<markers>>> around the
    sentence to translate. The model should translate only the marked sentence.
    """
    if use_instructions:
        system = (
            f"You are an expert translator. You will be given a paragraph in English "
            f"with one sentence wrapped in [START_TRANSLATE]...[END_TRANSLATE] markers. "
            f"Translate ONLY the marked sentence to {lang_name}. "
            f"Use the surrounding sentences as context but do not translate them. "
            f"Follow the user's instruction about tone, style, and context. "
            f"Give only the translation, no extra commentary."
        )
    else:
        system = (
            f"You will be given a paragraph in English with one sentence wrapped "
            f"in [START_TRANSLATE]...[END_TRANSLATE] markers. Translate ONLY the marked sentence to {lang_name}. "
            f"Use the surrounding sentences as context but do not translate them. "
            f"Preserve any speaker markers (e.g., <A:>, <B:>) in your translation. "
            f"Give only the translation, no extra commentary."
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
            "content": f"Instruction: {row.user_instruction}\n\n{marked_source}"
        })
    else:
        messages.append({"role": "user", "content": marked_source})

    return messages


def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    
    args = parse_args()
    
    lang_name = get_lang_name(args.tgt_lang)

    # --instruction-type implies --use-user-instructions
    if args.instruction_type != "full":
        args.use_user_instructions = True

    print(f"Model: {args.model}")
    print(f"Target language: {args.tgt_lang} ({lang_name})")
    print(f"Split: {args.split}")
    print(f"Use instructions: {args.use_user_instructions}")
    print(f"Instruction type: {args.instruction_type}")
    print(f"Semantic few-shot: {args.semantic_fewshot}")
    print(f"Paragraph level: {args.paragraph_level}")
    print(f"Numbered lists: {args.numbered_lists}")
    print(f"Enable thinking: {args.enable_thinking}")
    print(f"Limit: {args.limit}")

    # Check if results already exist
    savepath = BouquetInstructionRow.get_savepath(
        args.model, args.tgt_lang, args.split, args.use_user_instructions,
        args.semantic_fewshot, args.paragraph_level, args.numbered_lists,
        args.instruction_type, args.enable_thinking,
    )
    if savepath.exists() and not args.ignore_cache:
        print(f"\nResults already exist at {savepath}")
        print("Use --ignore-cache to re-translate")
        # Load and evaluate existing results
        rows = BouquetInstructionRow.load(savepath)
        scores = evaluate_translations(rows)
        print(f"\nOverall ChrF++: {scores['overall']:.2f}")
        return
    
    # Load data
    rows = BouquetInstructionRow.load_for_lang(args.tgt_lang, args.split, args.limit, args.instruction_type)
    print(f"\nLoaded {len(rows)} rows from BOUQuET with instructions")

    # Build few-shot examples and messages
    use_numbered_lists = args.paragraph_level and args.numbered_lists
    if use_numbered_lists:
        paragraph_groups = ParagraphGroup.group_by_paragraph(rows)
        print(f"Grouped into {len(paragraph_groups)} paragraphs (numbered-list mode)")
        few_shot_examples = FewShotExample.build_numbered_list_examples_for_split(args.tgt_lang, args.split)
        print(f"Using {len(few_shot_examples)} numbered-list few-shot examples")
        # One message per paragraph group (not per sentence)
        messages = [
            _build_numbered_list_messages(pg.get_numbered_source(), lang_name, few_shot_examples)
            for pg in paragraph_groups
        ]
    elif args.paragraph_level:
        paragraph_groups = ParagraphGroup.group_by_paragraph(rows)
        print(f"Grouped into {len(paragraph_groups)} paragraphs")
        few_shot_examples = FewShotExample.build_paragraph_examples_for_split(args.tgt_lang, args.split)
        print(f"Using {len(few_shot_examples)} sentence-in-context few-shot examples")
        # Build a lookup: uniq_id -> (ParagraphGroup, sentence_idx)
        row_to_context = {}
        for pg in paragraph_groups:
            for i, r in enumerate(pg.rows):
                row_to_context[r.uniq_id] = (pg, i)
        messages = []
        for row in rows:
            pg, sent_idx = row_to_context[row.uniq_id]
            marked_source = pg.get_marked_source(sent_idx)
            messages.append(_build_sentence_in_context_messages(
                marked_source, row, lang_name, few_shot_examples,
                use_instructions=args.use_user_instructions,
            ))
    elif args.semantic_fewshot:
        print(f"\nInitializing semantic example selector...")
        selector = SemanticExampleSelector(args.tgt_lang, args.split)
        # Build messages with per-row semantic examples
        messages = [
            row.get_messages(
                args.use_user_instructions,
                selector.get_similar_examples(row.user_instruction, k=5),
                strict_prompt=args.strict_prompt,
            )
            for row in rows
        ]
        print(f"Built messages with semantically similar few-shot examples per row")
    else:
        # Fixed few-shot examples for all rows
        print(f"\nBuilding fixed few-shot examples from {args.split} set...")
        few_shot_examples = FewShotExample.build_examples_for_split(args.tgt_lang, args.split)
        print(f"Using {len(few_shot_examples)} fixed examples for few-shot learning")
        messages = [row.get_messages(args.use_user_instructions, few_shot_examples, strict_prompt=args.strict_prompt) for row in rows]
    print("Sample messages:")
    for row_messages in messages[:1]:
        for msg in row_messages:
            role = msg['role']
            content = msg['content']
            print(f"[{role}]: {content}")
    
    # Initialize vLLM and translate
    resolved_model, lora_meta = resolve_vllm_model_and_lora(args.model)
    if lora_meta is None:
        print(f"\nLoading model {resolved_model}...")
        model = LLM(resolved_model, max_model_len=8192)
        lora_request = None
    else:
        print(
            f"\nLoading base model {resolved_model} with LoRA adapter "
            f"{lora_meta['path']} (rank={lora_meta['rank']})..."
        )
        model = LLM(
            resolved_model,
            max_model_len=8192,
            enable_lora=True,
            max_lora_rank=max(16, lora_meta["rank"]),
        )
        lora_request = LoRARequest(
            lora_name=lora_meta["name"],
            lora_int_id=1,
            lora_path=lora_meta["path"],
            base_model_name=lora_meta["base_model"],
        )
    
    # TranslateGemma uses a special delimiter-based chat template format:
    #   <<<source>>>en<<<target>>>id<<<text>>>text_to_translate
    #   <<<custom>>>custom_prompt (for instruction-following)
    if "translategemma" in args.model.lower():
        # Map FLORES lang codes to ISO 639-1 codes for TranslateGemma
        _flores_to_iso = {
            "ind_Latn": "id", "fra_Latn": "fr", "ukr_Cyrl": "uk",
            "jav_Latn": "jv", "khm_Khmr": "km", "tet_Latn": "tet",
        }
        tgt_iso = _flores_to_iso.get(args.tgt_lang, args.tgt_lang[:2])
        lang_name = get_lang_name(args.tgt_lang)
        new_messages = []
        for row in rows:
            if args.use_user_instructions:
                prompt = (
                    f"Translate the following English text to {lang_name}.\n"
                    f"Instruction: {row.user_instruction}\n\n"
                    f"Text: {row.tgt_text}\n\n"
                    f"Give only the translation, no extra commentary."
                )
                content = f"<<<custom>>>{prompt}"
            else:
                content = f"<<<source>>>en<<<target>>>{tgt_iso}<<<text>>>{row.tgt_text}"
            new_messages.append([{"role": "user", "content": content}])
        messages = new_messages

    print("Translating...")
    chat_kwargs = {}
    sampling_kwargs = {"max_tokens": 1024, "temperature": 0.0}
    if args.enable_thinking:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        sampling_kwargs["skip_special_tokens"] = False
        sampling_kwargs["max_tokens"] = 4096  # more room for reasoning + translation
    elif "qwen" in args.model.lower():
        # Disable thinking for thinking models (e.g. Qwen3.5) to get clean translations
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    outputs = model.chat(
        messages,
        sampling_params=SamplingParams(**sampling_kwargs),
        lora_request=lora_request,
        **chat_kwargs,
    )

    # Extract predictions
    reasoning_traces = []  # populated only when --enable-thinking
    if use_numbered_lists:
        # One output per paragraph group — parse numbered output and assign to rows
        for pg, output in zip(paragraph_groups, outputs):
            text = output.outputs[0].text.strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            pg.assign_predictions(text)
    else:
        for row, output in zip(rows, outputs):
            text = output.outputs[0].text.strip()
            if args.enable_thinking and "<channel|>" in text:
                # Gemma 4 thinking: <|channel>thought\n...reasoning...<channel|>translation
                thinking_part, translation_part = text.split("<channel|>", 1)
                # Strip the <|channel>thought\n prefix and any trailing special tokens
                thinking_part = re.sub(r"^<\|channel>thought\n?", "", thinking_part).strip()
                translation_part = re.sub(r"<turn\|>$", "", translation_part).strip()
                reasoning_traces.append({"uniq_id": row.uniq_id, "reasoning": thinking_part})
                text = translation_part
            else:
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            text = text.replace("[START_TRANSLATE]", "").replace("[END_TRANSLATE]", "").strip()  # strip leaked markers
            row.pred_text = text
    
    # Save results
    BouquetInstructionRow.save(rows, savepath)
    print(f"\nSaved results to {savepath}")

    # Save reasoning traces if thinking was enabled
    if reasoning_traces:
        traces_path = savepath.with_suffix(".reasoning.jsonl")
        with open(traces_path, "w", encoding="utf-8") as f:
            for trace in reasoning_traces:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        print(f"Saved {len(reasoning_traces)} reasoning traces to {traces_path}")
    
    # Evaluate
    scores = evaluate_translations(rows)
    print(f"\n{'='*50}")
    print(f"Overall ChrF++: {scores['overall']:.2f}")
    print(f"\nPer-domain scores:")
    for domain, stats in scores['by_domain'].items():
        print(f"  {domain}: {stats['chrf']:.2f} (n={stats['count']})")
    
    # Show a few examples
    print(f"\n{'='*50}")
    print("Sample translations:")
    for row in rows[:3]:
        print(f"\n[{row.domain}]")
        print(f"  EN: {row.tgt_text[:80]}...")
        print(f"  Pred: {row.pred_text[:80]}..." if row.pred_text else "  Pred: None")
        print(f"  Ref:  {row.src_text[:80]}...")
        if args.use_user_instructions:
            print(f"  Instr: {row.user_instruction[:60]}...")


if __name__ == "__main__":
    main()