"""Load English source sentences from SMOL dataset.

Usage:
    python -m distill.load_smol
"""
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import List

from datasets import load_dataset
from pydantic import BaseModel


OUTPUT_PATH = Path("datasets/smol/en_sources.jsonl")


class SmolSourceRow(BaseModel):
    key: str        # "SMOL-S{idx:05d}"
    en_text: str
    source: str     # "smolsent" or "smoldoc"

    @classmethod
    def save(cls, rows: List["SmolSourceRow"], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(row.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: Path) -> List["SmolSourceRow"]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(cls.model_validate_json(line))
        return rows


def main() -> None:
    # smolsent: one src per row
    sent_ds = load_dataset("google/smol", "smolsent__en_xh", split="train", trust_remote_code=True)
    sent_texts = [(r["src"], "smolsent") for r in sent_ds]

    # smoldoc: flatten srcs list
    doc_ds = load_dataset("google/smol", "smoldoc__en_ha", split="train", trust_remote_code=True)
    doc_texts = [(s, "smoldoc") for r in doc_ds for s in r["srcs"]]

    # Deduplicate on text, keeping first occurrence
    seen = set()
    unique = []
    for text, source in sent_texts + doc_texts:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append((text, source))

    rows = [
        SmolSourceRow(key=f"SMOL-S{i:05d}", en_text=text, source=source)
        for i, (text, source) in enumerate(unique)
    ]

    SmolSourceRow.save(rows, OUTPUT_PATH)
    print(f"Saved {len(rows)} unique English sources to {OUTPUT_PATH}")
    print(f"  from smolsent: {sum(1 for r in rows if r.source == 'smolsent')}")
    print(f"  from smoldoc: {sum(1 for r in rows if r.source == 'smoldoc')}")


if __name__ == "__main__":
    main()
