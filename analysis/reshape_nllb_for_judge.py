"""Reshape NLLB-3.3B BOUQuET outputs into the BouquetInstructionRow schema
expected by judge.bouquet.

Reads:
  translation_results/nllb_3_3b_bouquet_<split>/eng_Latn-<lang>_<split>.jsonl

Writes (per language):
  translation_results/bouquet_vllm/bouquet_<lang>_<split>_nllb-200-3.3B_without_instructions.jsonl

Usage:
    PYTHONPATH=. .venv/bin/python reshape_nllb_for_judge.py --split dev
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from pathlib import Path

from translate.bouquet_vllm import BouquetInstructionRow

LANGS = ["fra_Latn", "ind_Latn", "ukr_Cyrl", "khm_Khmr", "jav_Latn"]
NLLB_MODEL_NAME = "facebook/nllb-200-3.3B"


def reshape_one(lang: str, split: str):
    nllb_path = Path(f"translation_results/nllb_3_3b_bouquet_{split}/eng_Latn-{lang}_{split}.jsonl")
    if not nllb_path.exists():
        print(f"  MISSING NLLB output: {nllb_path}")
        return None

    nllb_by_id = {}
    with nllb_path.open() as f:
        for line in f:
            r = json.loads(line)
            nllb_by_id[r["uniq_id"]] = r["hyp_text"]

    base_rows = BouquetInstructionRow.load_for_lang(lang, split)
    out_rows = []
    missing = 0
    for row in base_rows:
        hyp = nllb_by_id.get(row.uniq_id)
        if hyp is None:
            missing += 1
            continue
        row.pred_text = hyp
        out_rows.append(row)

    out_path = BouquetInstructionRow.get_savepath(NLLB_MODEL_NAME, lang, split, use_instructions=False)
    BouquetInstructionRow.save(out_rows, out_path)
    print(f"  wrote {len(out_rows)} rows ({missing} missing) -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--langs", nargs="*", default=LANGS)
    args = parser.parse_args()

    for lang in args.langs:
        print(f"=== {lang} ===")
        reshape_one(lang, args.split)


if __name__ == "__main__":
    main()
