#!/usr/bin/env python3
"""
Compute correlation between human translation_rating_score and automatic
reference-based metrics (ChrF++ and COMET).

There are two ways to pair a human rating with a metric score:

  default    Join on uniq_id against the full test-set metric files under
             translation_results/. This is what the paper's human-agreement
             table reports. It assumes the translation now stored for a
             uniq_id is the one the annotator saw, which does not always
             hold: annotations were collected from an earlier generation, and
             38% of rated rows now carry different translation text. Those
             rows pair a human rating with a metric score computed on a
             different string.

  --matched  Score exactly the text in the export. COMET is read from
             annotation_exports/comet/ (produced by human_eval.match_comet)
             and ChrF++ is recomputed here from the export's own
             model_translation / reference_translation fields. This mirrors
             how the LLM judge is scored (human_eval.match_judge), so judge
             and metric are then measured on identical text.

--matched needs only the released annotation exports; the default mode needs
the item-level metric files, which are not released.

Usage:

    python -m human_eval.human_metric_correlation \
        --human-annotations annotation_exports/human/project_9_annotations.jsonl   # fra_Latn

    python -m human_eval.human_metric_correlation --matched \
        --human-annotations annotation_exports/human/project_11_annotations.jsonl  # khm_Khmr
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scipy.stats import kendalltau

from human_eval.human_llm_correlation import (
    krippendorff_alpha_interval,
    icc_two_way_mixed_consistency_average,
    load_annotation_rows,
    pearsonr,
)
from metrics.chrf import get_savepath as chrf_get_savepath
from metrics.comet import get_savepath as comet_get_savepath


def _load_metric_file(path: Path, score_field: str) -> dict[str, float]:
    """Load a metric JSONL file and return {uniq_id: score}."""
    index: dict[str, float] = {}
    if not path.exists():
        return index
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = obj.get("uniq_id")
            score = obj.get(score_field)
            if uid is not None and score is not None:
                index[uid] = float(score)
    return index


def _compute_and_print_correlations(
    metric_name: str,
    human_scores: list[float],
    metric_scores: list[float],
) -> dict:
    """Compute correlation metrics and print them. Returns a results dict."""
    n = len(human_scores)
    r = pearsonr(human_scores, metric_scores)
    if n >= 2:
        tau_result: Any = kendalltau(human_scores, metric_scores)
        tau = float(tau_result.statistic)
    else:
        tau = float("nan")
    alpha = krippendorff_alpha_interval(
        [[h, m] for h, m in zip(human_scores, metric_scores)]
    )
    icc = icc_two_way_mixed_consistency_average(zip(human_scores, metric_scores))

    r_str = f"{r:.3f}" if not math.isnan(r) else "nan"
    tau_str = f"{tau:.3f}" if not math.isnan(tau) else "nan"
    a_str = f"{alpha:.3f}" if not math.isnan(alpha) else "nan"
    icc_str = f"{icc:.3f}" if not math.isnan(icc) else "nan"

    print(f"\n{metric_name}:")
    print(f"  N = {n}")
    print(f"  Kendall tau-b = {tau_str}")
    print(f"  Pearson r = {r_str}")
    print(f"  Krippendorff alpha (interval) = {a_str}")
    print(f"  ICC(3,k) consistency = {icc_str}")

    return {
        "n": n,
        "pearson_r": r,
        "kendall_tau_b": tau,
        "krippendorff_alpha_interval": alpha,
        "icc_3_k_consistency": icc,
    }


def collect_uniqid_pairs(rows: list[dict], args: argparse.Namespace):
    """Join human rows to the full test-set metric files on uniq_id.

    This is the pairing behind the paper's human-agreement table. See the
    module docstring for its caveat.
    """
    # Group annotations by (model, src_lang, instruction_variant) to know which
    # metric files to load.
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        meta = row.get("meta", {})
        model = meta.get("model")
        src_lang = meta.get("src_lang")
        variant = meta.get("instruction_variant")
        if not model or not src_lang or not variant:
            continue
        key = (model, src_lang, variant)
        groups.setdefault(key, []).append(row)

    # Load metric indices per group.  The human annotations store model name
    # without the "google/" prefix, but get_savepath expects the full name.
    # We prepend "google/" to reconstruct the full model id.
    chrf_index: dict[tuple[str, str, str], dict[str, float]] = {}
    comet_index: dict[tuple[str, str, str], dict[str, float]] = {}
    use_instructions_map = {"with_instructions": True, "without_instructions": False}

    for (model, src_lang, variant) in groups:
        use_instr = use_instructions_map.get(variant)
        if use_instr is None:
            print(f"Warning: unknown instruction_variant '{variant}', skipping group")
            continue

        full_model = f"google/{model}"

        chrf_path = chrf_get_savepath(full_model, src_lang, args.split, use_instr)
        if chrf_path.exists():
            chrf_index[(model, src_lang, variant)] = _load_metric_file(chrf_path, "chrf_score")
        else:
            print(f"Warning: ChrF file not found: {chrf_path}")

        comet_path = comet_get_savepath(full_model, src_lang, args.split, use_instr, args.comet_model)
        if comet_path.exists():
            comet_index[(model, src_lang, variant)] = _load_metric_file(comet_path, "comet_score")
        else:
            print(f"Warning: COMET file not found: {comet_path}")

    # Collect paired scores.
    human_chrf_pairs: list[tuple[float, float]] = []
    human_comet_pairs: list[tuple[float, float]] = []
    skipped_no_human = 0
    skipped_no_chrf = 0
    skipped_no_comet = 0

    for (model, src_lang, variant), group_rows in groups.items():
        chrf_scores = chrf_index.get((model, src_lang, variant), {})
        comet_scores = comet_index.get((model, src_lang, variant), {})
        for row in group_rows:
            human_score = row.get("translation_rating_score")
            if human_score is None:
                skipped_no_human += 1
                continue
            human_val = float(human_score)

            uniq_id = row.get("meta", {}).get("uniq_id")
            if not uniq_id:
                continue

            chrf_val = chrf_scores.get(uniq_id)
            if chrf_val is not None:
                human_chrf_pairs.append((human_val, chrf_val))
            else:
                skipped_no_chrf += 1

            comet_val = comet_scores.get(uniq_id)
            if comet_val is not None:
                human_comet_pairs.append((human_val, comet_val))
            else:
                skipped_no_comet += 1
    return (human_chrf_pairs, human_comet_pairs,
            skipped_no_human, skipped_no_chrf, skipped_no_comet)


def collect_matched_pairs(rows: list[dict], annotation_path: Path, comet_model: str):
    """Pair human ratings with metric scores computed on the rated text itself.

    COMET is read from annotation_exports/comet/, keyed by task_id; ChrF++ is
    recomputed here with the same settings as metrics.chrf (word_order=2).
    """
    from sacrebleu import sentence_chrf

    comet_slug = comet_model.split("/")[-1]
    comet_path = Path("annotation_exports/comet") / f"{annotation_path.stem}_{comet_slug}.jsonl"
    comet_by_task: dict[Any, float] = {}
    if comet_path.exists():
        with open(comet_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    comet_by_task[row.get("task_id")] = row.get("comet_score")
    else:
        print(f"Warning: matched COMET file not found: {comet_path}")
        print("  Produce it with: python -m human_eval.match_comet "
              f"--annotations {annotation_path}")

    chrf_pairs: list[tuple[float, float]] = []
    comet_pairs: list[tuple[float, float]] = []
    skipped_no_human = skipped_no_chrf = skipped_no_comet = 0
    for row in rows:
        human_score = row.get("translation_rating_score")
        if human_score is None:
            skipped_no_human += 1
            continue
        human_val = float(human_score)

        hyp, ref = row.get("model_translation"), row.get("reference_translation")
        if hyp and ref:
            chrf_pairs.append((human_val, sentence_chrf(hyp, [ref], word_order=2).score))
        else:
            skipped_no_chrf += 1

        comet_val = comet_by_task.get(row.get("task_id"))
        if comet_val is None:
            skipped_no_comet += 1
        else:
            comet_pairs.append((human_val, float(comet_val)))

    return (chrf_pairs, comet_pairs, skipped_no_human, skipped_no_chrf, skipped_no_comet)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute human vs automatic metric (ChrF++, COMET) correlations"
    )
    parser.add_argument(
        "--human-annotations",
        type=str,
        required=True,
        help="Path to Label Studio exported annotations JSONL",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Dataset split (default: dev)",
    )
    parser.add_argument(
        "--comet-model",
        type=str,
        default="Unbabel/XCOMET-XL",
        help="COMET model name used for scoring (default: Unbabel/XCOMET-XL)",
    )
    parser.add_argument(
        "--matched",
        action="store_true",
        help="Score the exported text directly instead of joining on uniq_id "
             "(see the module docstring).",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        help="Optional path to write a JSON summary report",
    )
    args = parser.parse_args()

    annotation_path = Path(args.human_annotations)
    rows = load_annotation_rows(annotation_path)
    if not rows:
        raise ValueError(f"No annotation rows found in {annotation_path}")

    if args.matched:
        collected = collect_matched_pairs(rows, annotation_path, args.comet_model)
    else:
        collected = collect_uniqid_pairs(rows, args)
    (human_chrf_pairs, human_comet_pairs,
     skipped_no_human, skipped_no_chrf, skipped_no_comet) = collected

    # Print results.
    print("\n## Human vs Automatic Metric Correlation\n")
    print(f"Annotations: {annotation_path}")
    print(f"Matching: {'exported text' if args.matched else 'uniq_id join'}")
    print(f"Total annotation rows: {len(rows)}")
    print(f"Rows missing human translation_rating_score: {skipped_no_human}")
    print(f"Rows matched to ChrF++: {len(human_chrf_pairs)}")
    print(f"Rows matched to COMET: {len(human_comet_pairs)}")
    if skipped_no_chrf:
        print(f"Rows with no ChrF++ match: {skipped_no_chrf}")
    if skipped_no_comet:
        print(f"Rows with no COMET match: {skipped_no_comet}")

    report: dict = {
        "annotations_path": str(annotation_path),
        "split": args.split,
        "comet_model": args.comet_model,
        "matching": "matched" if args.matched else "uniq_id",
        "total_rows": len(rows),
        "metrics": {},
    }

    if human_chrf_pairs:
        hs, ms = zip(*human_chrf_pairs)
        result = _compute_and_print_correlations(
            "translation_rating_score vs ChrF++",
            list(hs),
            list(ms),
        )
        report["metrics"]["chrf"] = result
    else:
        print("\nNo ChrF++ pairs to compute correlation.")

    if human_comet_pairs:
        hs, ms = zip(*human_comet_pairs)
        result = _compute_and_print_correlations(
            "translation_rating_score vs COMET",
            list(hs),
            list(ms),
        )
        report["metrics"]["comet"] = result
    else:
        print("\nNo COMET pairs to compute correlation.")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nWrote report to {report_path}")


if __name__ == "__main__":
    main()
