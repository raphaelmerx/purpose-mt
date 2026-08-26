#!/usr/bin/env python3
"""
Compute correlation between Label Studio human scores and LLM eval scores.

Usage:

    python -m human_eval.human_llm_correlation \
        --human-annotations annotation_exports/human/project_9_annotations.jsonl  # fra_Latn
    python -m human_eval.human_llm_correlation \
        --human-annotations annotation_exports/human/project_10_annotations.jsonl  # ind_Latn
    python -m human_eval.human_llm_correlation \
        --human-annotations annotation_exports/human/project_11_annotations.jsonl  # khm_Khmr
    python -m human_eval.human_llm_correlation \
        --human-annotations annotation_exports/human/project_13_annotations.jsonl  # ukr_Cyrl
    python -m human_eval.human_llm_correlation \
        --human-annotations annotation_exports/human/project_14_annotations.jsonl  # jav_Latn
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Tuple, Any

from scipy.stats import kendalltau

from judge.bouquet import EvaluatedRow


SCORE_FIELDS = ("translation_rating_score", "adaptedness_score")


def pearsonr(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation. Returns NaN if undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(denom_x * denom_y)
    if denom == 0:
        return float("nan")
    return num / denom


def krippendorff_alpha_interval(units: Iterable[list[float]]) -> float:
    """Krippendorff's alpha for interval data, using pooled pairwise disagreement."""
    unit_list = [vals for vals in units if len(vals) >= 2]
    if not unit_list:
        return float("nan")

    do_sum = 0.0
    do_pairs = 0
    for vals in unit_list:
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                do_sum += (vals[i] - vals[j]) ** 2
                do_pairs += 1
    if do_pairs == 0:
        return float("nan")
    do = do_sum / do_pairs

    all_vals = [v for vals in unit_list for v in vals]
    if len(all_vals) < 2:
        return float("nan")
    de_sum = 0.0
    de_pairs = 0
    for i in range(len(all_vals)):
        for j in range(i + 1, len(all_vals)):
            de_sum += (all_vals[i] - all_vals[j]) ** 2
            de_pairs += 1
    if de_pairs == 0:
        return float("nan")
    de = de_sum / de_pairs
    if de == 0:
        return 1.0
    return 1.0 - (do / de)


def icc_two_way_mixed_consistency_average(pairs: Iterable[Tuple[float, float]]) -> float:
    """ICC(3,k) two-way mixed, consistency, average measures for two raters."""
    ratings = [(float(x), float(y)) for x, y in pairs]
    n = len(ratings)
    k = 2
    if n < 2:
        return float("nan")

    # Build n x k matrix
    matrix = [list(pair) for pair in ratings]
    grand_mean = sum(sum(row) for row in matrix) / (n * k)

    row_means = [sum(row) / k for row in matrix]
    col_means = [
        sum(matrix[i][j] for i in range(n)) / n
        for j in range(k)
    ]

    ss_total = sum(
        (matrix[i][j] - grand_mean) ** 2 for i in range(n) for j in range(k)
    )
    ss_rows = k * sum((mean - grand_mean) ** 2 for mean in row_means)
    ss_cols = n * sum((mean - grand_mean) ** 2 for mean in col_means)
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = df_rows * df_cols
    if df_rows == 0 or df_error == 0:
        return float("nan")

    ms_rows = ss_rows / df_rows
    ms_error = ss_error / df_error

    if ms_rows == 0:
        return float("nan")
    return (ms_rows - ms_error) / ms_rows


def load_annotation_rows(path: Path) -> list[dict]:
    """Load exported Label Studio annotation JSONL."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _match_key(user_instruction: str | None, source_text: str | None, model_translation: str | None) -> tuple[str, str, str]:
    return (
        user_instruction or "",
        source_text or "",
        model_translation or "",
    )


def build_eval_index(eval_path: Path) -> dict[tuple[str, str, str], dict]:
    """Load eval JSONL and index by (instruction, source, model_translation)."""
    index: dict[tuple[str, str, str], dict] = {}
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            evaluation = obj.get("evaluation", obj)
            row = obj.get("row", {})
            key = _match_key(
                row.get("user_instruction"),
                row.get("tgt_text"),
                row.get("pred_text"),
            )
            if not any(key):
                continue
            index[key] = {
                "translation_rating_score": evaluation.get("translation_rating_score"),
                "adaptedness_score": evaluation.get("adaptedness_score"),
            }
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute human vs LLM score correlations")
    parser.add_argument(
        "--human-annotations",
        type=str,
        required=True,
        help="Path to Label Studio exported annotations JSONL",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default="gemini/gemini-3-flash-preview",
        help="LLM-as-judge to correlate with (default: gemini/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        help="Optional path to write a JSON summary report",
    )
    parser.add_argument(
        "--reference-based",
        action="store_true",
        help="Look up the reference-based LLM evaluation file (suffix _refbased).",
    )
    args = parser.parse_args()

    annotation_path = Path(args.human_annotations)
    rows = load_annotation_rows(annotation_path)

    if not rows:
        raise ValueError(f"No annotation rows found in {annotation_path}")

    eval_slug = EvaluatedRow._get_eval_model_slug(args.eval_model)
    suffix = "_refbased" if args.reference_based else ""
    llm_path = Path("annotation_exports/llm") / f"{annotation_path.stem}_{eval_slug}{suffix}.jsonl"
    if not llm_path.exists():
        raise FileNotFoundError(
            f"LLM annotations not found at {llm_path}. "
            "Run human_eval.match_judge first."
            + (" (Pass --reference-based there too.)" if args.reference_based else "")
        )

    human_scores: dict[str, list[float]] = {field: [] for field in SCORE_FIELDS}
    llm_scores: dict[str, list[float]] = {field: [] for field in SCORE_FIELDS}
    matched = 0

    eval_index = build_eval_index(llm_path)
    for row in rows:
        key = _match_key(
            row.get("user_instruction"),
            row.get("source_text"),
            row.get("model_translation"),
        )
        if not any(key):
            continue
        eval_scores = eval_index.get(key)
        if not eval_scores:
            continue
        for field in SCORE_FIELDS:
            human_value = row.get(field)
            llm_value = eval_scores.get(field)
            if human_value is None or llm_value is None:
                continue
            human_scores[field].append(float(human_value))
            llm_scores[field].append(float(llm_value))
        matched += 1

    report = {
        "annotations_path": str(annotation_path),
        "llm_annotations_path": str(llm_path),
        "matched_rows": matched,
        "metrics": {},
    }

    print("\n## Correlation Results (Pooled)\n")
    print(f"Annotations: {annotation_path}")
    print(f"Matched rows: {matched}")
    for field in SCORE_FIELDS:
        xs = human_scores[field]
        ys = llm_scores[field]
        r = pearsonr(xs, ys)
        if len(xs) >= 2:
            tau_result: Any = kendalltau(xs, ys)
            tau = float(tau_result.statistic)
        else:
            tau = float("nan")
        alpha = krippendorff_alpha_interval([[x, y] for x, y in zip(xs, ys)])
        icc = icc_two_way_mixed_consistency_average(zip(xs, ys))
        report["metrics"][field] = {
            "n": len(xs),
            "pearson_r": r,
            "kendall_tau_b": tau,
            "krippendorff_alpha_interval": alpha,
            "icc_3_k_consistency": icc,
        }

        r_str = f"{r:.3f}" if not math.isnan(r) else "nan"
        tau_str = f"{tau:.3f}" if not math.isnan(tau) else "nan"
        a_str = f"{alpha:.3f}" if not math.isnan(alpha) else "nan"
        icc_str = f"{icc:.3f}" if not math.isnan(icc) else "nan"
        print(f"\n{field}:")
        print(f"  N = {len(xs)}")
        print(f"  Kendall tau-b = {tau_str}")
        print(f"  Pearson r = {r_str}")
        print(f"  Krippendorff alpha (interval) = {a_str}")
        print(f"  ICC(3,k) consistency = {icc_str}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nWrote report to {report_path}")


if __name__ == "__main__":
    main()
