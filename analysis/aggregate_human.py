#!/usr/bin/env python3
"""Reduce the human annotations to non-identifying score tables.

The item-level human exports carry an annotator id, timestamps and free-text
comments, and there is one annotator per language, so a row is attributable.
Annotators consented to aggregated, non-identifying use only. This script
produces that aggregate: means, correlations and divergence rates, with no
per-item rows, no annotator ids, no timestamps and no free text.

Reads:  annotation_exports/{human,llm,comet}/
Writes: data/scores/human_scores.csv
        data/scores/human_agreement.csv
        data/scores/human_divergence.csv

    python -m analysis.aggregate_human

COMET rows are emitted only if annotation_exports/comet/ is present; ChrF++ is
recomputed here from the export's own translation and reference, matching
metrics.chrf (word_order=2). Both are the "matched" pairing of
annotation.human_metric_correlation: scored on the text the annotator saw.
The paper's table instead joins on uniq_id against the unreleased item-level
metric files, so those two columns will not reproduce it exactly.
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from scipy.stats import kendalltau

from human_eval.human_llm_correlation import (
    SCORE_FIELDS,
    build_eval_index,
    icc_two_way_mixed_consistency_average,
    krippendorff_alpha_interval,
    load_annotation_rows,
    pearsonr,
    _match_key,
)

HUMAN_DIR = Path("annotation_exports/human")
LLM_DIR = Path("annotation_exports/llm")
COMET_DIR = Path("annotation_exports/comet")
OUT_DIR = Path("data/scores")

# The five projects behind the paper's human evaluation, one annotator each.
# Pilot projects (4, 6) are excluded: they used an earlier interface.
PROJECTS = [
    ("fra_Latn", "project_9_annotations.jsonl"),
    ("ind_Latn", "project_10_annotations.jsonl"),
    ("khm_Khmr", "project_11_annotations.jsonl"),
    ("ukr_Cyrl", "project_13_annotations.jsonl"),
    ("jav_Latn", "project_14_annotations.jsonl"),
]

# Items whose two human scores differ by more than this count as divergent.
DIVERGENCE_THRESHOLD = 20

COMET_MODEL = "Unbabel/XCOMET-XL"


def mean(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def rounded(x: Optional[float], places: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{places}f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  {path}  ({len(rows)} rows)")


def agreement_stats(xs: list[float], ys: list[float]) -> dict:
    """Pearson / Kendall / Krippendorff / ICC, as human_llm_correlation reports them."""
    if len(xs) >= 2:
        tau_result: Any = kendalltau(xs, ys)
        tau = float(tau_result.statistic)
    else:
        tau = float("nan")
    return {
        "n": len(xs),
        "pearson_r": pearsonr(xs, ys),
        "kendall_tau_b": tau,
        "krippendorff_alpha_interval": krippendorff_alpha_interval(
            [[x, y] for x, y in zip(xs, ys)]
        ),
        "icc_3k_consistency": icc_two_way_mixed_consistency_average(zip(xs, ys)),
    }


def load_projects() -> dict[str, list[dict]]:
    per_lang: dict[str, list[dict]] = {}
    for lang, fname in PROJECTS:
        path = HUMAN_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing human export: {path}")
        per_lang[lang] = load_annotation_rows(path)
    return per_lang


def build_scores(per_lang: dict[str, list[dict]]) -> list[dict]:
    """Per (lang, model, condition) means, plus a model='all' row per condition."""
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for lang, rows in per_lang.items():
        for r in rows:
            meta = r.get("meta") or {}
            model = meta.get("model") or "unknown"
            condition = meta.get("instruction_variant") or "unknown"
            buckets[(lang, model, condition)].append(r)
            buckets[(lang, "all", condition)].append(r)

    out = []
    for (lang, model, condition), rows in sorted(buckets.items()):
        crit = [r for r in rows if r.get("is_critical_failure") is True]
        ok = [r for r in rows if r.get("is_critical_failure") is not True]

        def vals(subset: list[dict], field: str) -> list[float]:
            return [float(r[field]) for r in subset if r.get(field) is not None]

        out.append({
            "lang": lang,
            "model": model,
            "condition": condition,
            "n": len(rows),
            "n_critical_failure": len(crit),
            "critical_failure_rate": rounded(len(crit) / len(rows) if rows else None),
            "translation_rating": rounded(mean(vals(rows, "translation_rating_score")), 2),
            "adaptedness": rounded(mean(vals(rows, "adaptedness_score")), 2),
            "translation_rating_excl_critical": rounded(
                mean(vals(ok, "translation_rating_score")), 2),
            "adaptedness_excl_critical": rounded(mean(vals(ok, "adaptedness_score")), 2),
        })
    return out


def _llm_comparators(stem: str) -> list[tuple[str, Path]]:
    """Every LLM-judge file for this project, named by its filename suffix."""
    found = []
    for path in sorted(LLM_DIR.glob(f"{stem}_*.jsonl")):
        found.append((path.stem[len(stem) + 1:], path))
    return found


def build_agreement(per_lang: dict[str, list[dict]]) -> list[dict]:
    """Human vs LLM judge, ChrF++ and COMET, per language and pooled."""
    from sacrebleu import sentence_chrf

    comet_slug = COMET_MODEL.split("/")[-1]
    # (comparator, score_field) -> pooled pairs
    pooled: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    rows_out: list[dict] = []

    for lang, fname in PROJECTS:
        rows = per_lang[lang]
        stem = Path(fname).stem

        # LLM judge: both score fields, matched on (instruction, source, translation).
        for comparator, llm_path in _llm_comparators(stem):
            index = build_eval_index(llm_path)
            pairs: dict[str, list[tuple[float, float]]] = {f: [] for f in SCORE_FIELDS}
            for r in rows:
                key = _match_key(
                    r.get("user_instruction"),
                    r.get("source_text"),
                    r.get("model_translation"),
                )
                if not any(key):
                    continue
                scores = index.get(key)
                if not scores:
                    continue
                for field in SCORE_FIELDS:
                    hv, lv = r.get(field), scores.get(field)
                    if hv is None or lv is None:
                        continue
                    pairs[field].append((float(hv), float(lv)))
            for field in SCORE_FIELDS:
                if not pairs[field]:
                    continue
                xs = [p[0] for p in pairs[field]]
                ys = [p[1] for p in pairs[field]]
                rows_out.append({
                    "lang": lang, "score_field": field,
                    "comparator": f"llm_{comparator}", "matching": "exported_text",
                    **agreement_stats(xs, ys),
                })
                pooled[(f"llm_{comparator}", field)] += pairs[field]

        # ChrF++, recomputed on the rated text.
        chrf_pairs: list[tuple[float, float]] = []
        for r in rows:
            human = r.get("translation_rating_score")
            hyp, ref = r.get("model_translation"), r.get("reference_translation")
            if human is None or not hyp or not ref:
                continue
            chrf_pairs.append(
                (float(human), sentence_chrf(hyp, [ref], word_order=2).score)
            )
        if chrf_pairs:
            rows_out.append({
                "lang": lang, "score_field": "translation_rating_score",
                "comparator": "chrf++", "matching": "exported_text",
                **agreement_stats([p[0] for p in chrf_pairs],
                                  [p[1] for p in chrf_pairs]),
            })
            pooled[("chrf++", "translation_rating_score")] += chrf_pairs

        # COMET, read from the matched exports if they were fetched.
        comet_path = COMET_DIR / f"{stem}_{comet_slug}.jsonl"
        if not comet_path.exists():
            print(f"  note: no COMET export for {lang} at {comet_path}, skipping")
            continue
        comet_by_task: dict[Any, float] = {}
        with open(comet_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    comet_by_task[obj.get("task_id")] = obj.get("comet_score")
        # COMET is 0-1 where the human rating and ChrF++ are 0-100. Rescale, or
        # Krippendorff's alpha and ICC compare incommensurable scales and come
        # out meaninglessly negative. Pearson and Kendall are unaffected.
        comet_pairs = []
        for r in rows:
            human = r.get("translation_rating_score")
            cv = comet_by_task.get(r.get("task_id"))
            if human is None or cv is None:
                continue
            comet_pairs.append((float(human), float(cv) * 100.0))
        if comet_pairs:
            rows_out.append({
                "lang": lang, "score_field": "translation_rating_score",
                "comparator": f"comet_{comet_slug}", "matching": "exported_text",
                **agreement_stats([p[0] for p in comet_pairs],
                                  [p[1] for p in comet_pairs]),
            })
            pooled[(f"comet_{comet_slug}", "translation_rating_score")] += comet_pairs

    for (comparator, field), pairs in sorted(pooled.items()):
        rows_out.append({
            "lang": "pooled", "score_field": field,
            "comparator": comparator, "matching": "exported_text",
            **agreement_stats([p[0] for p in pairs], [p[1] for p in pairs]),
        })

    for row in rows_out:
        for k in ("pearson_r", "kendall_tau_b",
                  "krippendorff_alpha_interval", "icc_3k_consistency"):
            row[k] = rounded(row[k])
    return rows_out


def build_divergence(per_lang: dict[str, list[dict]]) -> list[dict]:
    """How far the two human scores diverge, overall and by domain."""
    buckets: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for lang, rows in per_lang.items():
        for r in rows:
            rating, adapt = r.get("translation_rating_score"), r.get("adaptedness_score")
            if rating is None or adapt is None:
                continue
            domain = (r.get("meta") or {}).get("domain") or "unknown"
            pair = (float(rating), float(adapt))
            buckets[(lang, domain)].append(pair)
            buckets[(lang, "all")].append(pair)
            buckets[("pooled", domain)].append(pair)
            buckets[("pooled", "all")].append(pair)

    out = []
    for (lang, domain), pairs in sorted(buckets.items()):
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        n_div = sum(1 for x, y in pairs if abs(x - y) > DIVERGENCE_THRESHOLD)
        out.append({
            "lang": lang,
            "domain": domain,
            "n": len(pairs),
            "pearson_r_rating_vs_adaptedness": rounded(pearsonr(xs, ys)),
            "mean_abs_difference": rounded(mean([abs(x - y) for x, y in pairs]), 2),
            f"n_diff_gt_{DIVERGENCE_THRESHOLD}": n_div,
            f"frac_diff_gt_{DIVERGENCE_THRESHOLD}": rounded(n_div / len(pairs)),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    per_lang = load_projects()
    total = sum(len(v) for v in per_lang.values())
    print(f"Loaded {total} annotations across {len(per_lang)} languages\n")

    print("Writing:")
    scores = build_scores(per_lang)
    write_csv(args.out_dir / "human_scores.csv", list(scores[0].keys()), scores)

    divergence = build_divergence(per_lang)
    write_csv(args.out_dir / "human_divergence.csv",
              list(divergence[0].keys()), divergence)

    agreement = build_agreement(per_lang)
    write_csv(args.out_dir / "human_agreement.csv",
              list(agreement[0].keys()), agreement)


if __name__ == "__main__":
    main()
