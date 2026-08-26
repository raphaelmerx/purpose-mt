#!/usr/bin/env python3
"""
Score XCOMET-XL on exactly the rows that were human-rated in a Label Studio
export. The COMET analogue of human_eval/match_judge.py (LLM-as-judge).

Unlike matching human rows to the full test-set COMET files on uniq_id, this
scores the translation text *directly* from the export, so the COMET score is
guaranteed to be computed on precisely what the human annotators rated.
Reference-based COMET inputs are taken straight from the export:

    src = source_text, mt = model_translation, ref = reference_translation

(No langid gating -- matches metrics/comet.py, the paper pipeline.)

Usage:
    python -m human_eval.match_comet \
        --annotations annotation_exports/human/project_9_annotations.jsonl   # fra_Latn
    python -m human_eval.match_comet \
        --annotations annotation_exports/human/project_10_annotations.jsonl  # ind_Latn
    python -m human_eval.match_comet \
        --annotations annotation_exports/human/project_11_annotations.jsonl  # khm_Khmr
    python -m human_eval.match_comet \
        --annotations annotation_exports/human/project_13_annotations.jsonl  # ukr_Cyrl
    python -m human_eval.match_comet \
        --annotations annotation_exports/human/project_14_annotations.jsonl  # jav_Latn

    # All five at once (COMET model loaded only once):
    python -m human_eval.match_comet --annotations \
        annotation_exports/human/project_9_annotations.jsonl,annotation_exports/human/project_10_annotations.jsonl,annotation_exports/human/project_11_annotations.jsonl,annotation_exports/human/project_13_annotations.jsonl,annotation_exports/human/project_14_annotations.jsonl
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from pathlib import Path

from common.comet_utils import load_comet_model, comet_cached_predict


# Reference-based COMET needs these three text fields on every row.
REQUIRED_FIELDS = ("source_text", "model_translation", "reference_translation")


def load_annotation_rows(path: Path) -> list[dict]:
    """Load a Label Studio export JSONL."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def default_output_path(input_path: Path, comet_model: str) -> Path:
    comet_slug = comet_model.split("/")[-1]
    return Path("annotation_exports/comet") / f"{input_path.stem}_{comet_slug}.jsonl"


def score_file(
    annotation_path: Path,
    output_path: Path,
    comet_model,
    comet_model_name: str,
    batch_size: int,
    gpus: int,
) -> dict | None:
    """Score one export file and write per-row COMET scores. Returns a summary."""
    rows = load_annotation_rows(annotation_path)
    if not rows:
        print(f"WARNING: no rows in {annotation_path}, skipping")
        return None

    kept: list[dict] = []
    skipped = 0
    for row in rows:
        if all(row.get(f) for f in REQUIRED_FIELDS):
            kept.append(row)
        else:
            skipped += 1
    if not kept:
        print(f"WARNING: no scorable rows in {annotation_path}, skipping")
        return None

    data = [
        {
            "src": row["source_text"],            # original source shown to annotators
            "mt": row["model_translation"],       # the rated machine translation
            "ref": row["reference_translation"],  # gold reference
        }
        for row in kept
    ]
    scores = comet_cached_predict(
        comet_model, data,
        batch_size=batch_size, gpus=gpus, model_name=comet_model_name,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row, score in zip(kept, scores):
            meta = row.get("meta", {})
            f.write(json.dumps({
                "task_id": row.get("task_id"),
                "uniq_id": meta.get("uniq_id"),
                "model": meta.get("model"),
                "src_lang": meta.get("src_lang"),
                "instruction_variant": meta.get("instruction_variant"),
                "domain": meta.get("domain"),
                "comet_score": score,
            }, ensure_ascii=False) + "\n")

    overall = sum(scores) / len(scores)
    print(
        f"  {annotation_path.name}: scored {len(kept)}, skipped {skipped}, "
        f"overall COMET {overall:.4f} -> {output_path}"
    )
    return {
        "input": str(annotation_path),
        "output": str(output_path),
        "n": len(kept),
        "skipped": skipped,
        "overall": overall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XCOMET-XL scoring for Label Studio human-annotated exports"
    )
    parser.add_argument(
        "--annotations", type=str, required=True,
        help="Label Studio export JSONL path(s); comma-separated for multiple",
    )
    parser.add_argument(
        "--comet-model", type=str, default="Unbabel/XCOMET-XL",
        help="COMET model (default: Unbabel/XCOMET-XL)",
    )
    parser.add_argument(
        "--output", type=str,
        help="Output JSONL path (only honored when a single input file is given)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for COMET inference (default: 8)",
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs (default: 1)",
    )
    args = parser.parse_args()

    paths = [Path(p.strip()) for p in args.annotations.split(",") if p.strip()]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Annotation file not found: {p}")
    if args.output and len(paths) > 1:
        print("WARNING: --output ignored for multiple inputs; using default paths")

    comet_model = load_comet_model(args.comet_model)

    results: list[dict] = []
    for p in paths:
        out = (
            Path(args.output)
            if args.output and len(paths) == 1
            else default_output_path(p, args.comet_model)
        )
        res = score_file(p, out, comet_model, args.comet_model, args.batch_size, args.gpus)
        if res:
            results.append(res)

    print("\n## COMET Annotation Export\n")
    print(f"{'input':<34} {'n':>5} {'skipped':>8} {'overall':>9}")
    print("-" * 58)
    for r in results:
        print(
            f"{Path(r['input']).stem:<34} {r['n']:>5} {r['skipped']:>8} "
            f"{r['overall']:>9.4f}"
        )


if __name__ == "__main__":
    main()
