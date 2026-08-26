"""Aggregate item-level results into the score tables behind the paper.

The item-level judge outputs are large (~770 MB) and embed benchmark source text, so
they are not released. This script reduces them to per-condition means, which carry no
benchmark text and are what every table and figure in the paper is computed from.

Reads:  translation_results/{qualitative_analysis,wmt24pp_qualitative,comet,chrf,win_rate}/
Writes: data/scores/*.csv

    python -m analysis.aggregate
    python -m analysis.aggregate --all-models   # include non-paper ablation runs

Emitted files:
    bouquet_scores.csv          per (split, lang, model, condition, judge)
    bouquet_domain_scores.csv   the above, split by domain
    wmt24pp_scores.csv          per (lang, model, condition, judge)
    wmt24pp_domain_scores.csv   the above, split by domain
    metric_scores.csv           XCOMET-XL and ChrF++ means per condition
    win_rates.csv               pairwise with- vs without-instruction win rates
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

RESULTS = Path("translation_results")
OUT_DIR = Path("data/scores")

# Models reported in the paper. Everything else (DPO ablations, TranslateGemma probes,
# judge-as-translator runs) is excluded unless --all-models is passed.
PAPER_MODELS = {
    "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it", "gemma-4-31b-it",
    "Qwen3.5-27B", "gpt-5.4",
    "gemma-3-4b-it-sft-smol-teacher-ukr_Cyrl",
    "gemma-3-4b-it-sft-smol-teacher-jav_Latn",
    "nllb-200-3.3B",
}

BOUQUET_RE = re.compile(
    r"^eval_bouquet_"
    r"(?P<lang>[a-z]{3}_[A-Za-z]{4})_"
    r"(?P<split>dev|test)_"
    r"(?P<model>.+?)_"
    r"(?P<condition>with_instructions|without_instructions)"
    r"(?P<variant>(?:_[a-z0-9][a-z0-9.-]*)*?)_"
    r"(?P<judge>(?:gemini|models|gpt|openai)-.+)"
    r"\.jsonl$"
)

WMT_RE = re.compile(
    r"^eval_wmt24pp_"
    r"(?P<lang>[a-z]{2,3}-[A-Za-z_]+)_"
    r"(?P<model>.+?)_"
    r"(?P<condition>with_instructions|without_instructions)"
    r"(?P<variant>(?:_[a-z0-9][a-z0-9.-]*)*?)_"
    r"(?P<judge>(?:gemini|models|gpt|openai)-.+)"
    r"\.jsonl$"
)

METRIC_RE = re.compile(
    r"^(?P<metric>comet_[A-Za-z0-9.-]+|chrf)_bouquet_"
    r"(?P<lang>[a-z]{3}_[A-Za-z]{4})_"
    r"(?P<split>dev|test)_"
    r"(?P<model>.+?)_"
    r"(?P<condition>with_instructions|without_instructions)"
    r"(?P<variant>(?:_[a-z0-9][a-z0-9.-]*)*?)"
    r"\.jsonl$"
)

WINRATE_RE = re.compile(
    r"^winrate_bouquet_"
    r"(?P<lang>[a-z]{3}_[A-Za-z]{4})_"
    r"(?P<split>dev|test)_"
    r"(?P<model>.+?)"
    r"\.jsonl$"
)


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def mean(xs: List[float]) -> Optional[float]:
    return round(statistics.mean(xs), 4) if xs else None


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: [str(r.get(k) or "") for k in fieldnames])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  {path}  ({len(rows)} rows)")


def summarise(evals: List[dict]) -> dict:
    """Reduce a list of judge evaluations to the paper's headline numbers.

    `translation_rating` and `adaptedness` average over ALL rows, with critical
    failures contributing their score of 0. This matches `EvaluatedRow.compute_metrics`
    in judge/bouquet.py, which is what the paper's tables are built from.

    The `*_excl_critical` columns average over non-failed rows only, for the
    instruction-failure tradeoff in §7. Note these are per-condition means: they do not
    reproduce the paper's paired "+14.7 among non-failed translations" figure, which is
    computed over items that did not fail in *either* condition and so needs item-level
    data rather than these aggregates.
    """
    n = len(evals)
    crit = [e for e in evals if e.get("is_critical_failure")]
    ok = [e for e in evals if not e.get("is_critical_failure")]

    def _mean(subset, field):
        return mean([e[field] for e in subset if e.get(field) is not None])

    return {
        "n": n,
        "n_critical_failure": len(crit),
        "critical_failure_rate": round(len(crit) / n, 4) if n else None,
        "translation_rating": _mean(evals, "translation_rating_score"),
        "adaptedness": _mean(evals, "adaptedness_score"),
        "translation_rating_excl_critical": _mean(ok, "translation_rating_score"),
        "adaptedness_excl_critical": _mean(ok, "adaptedness_score"),
    }


def _condition(condition: str, variant: str) -> str:
    """Fold the filename's condition + variant suffix into one readable label."""
    return condition + (variant or "")


def aggregate_judge(
    directory: Path, pattern: re.Pattern, has_split: bool, models: Optional[set]
) -> Tuple[List[dict], List[dict]]:
    """Return (overall rows, per-domain rows) for one judge output directory."""
    overall, by_domain = [], []
    if not directory.is_dir():
        print(f"  (skipped, not found: {directory})")
        return overall, by_domain

    for path in sorted(directory.glob("eval_*.jsonl")):
        m = pattern.match(path.name)
        if not m:
            continue
        g = m.groupdict()
        if models is not None and g["model"] not in models:
            continue

        rows = list(read_jsonl(path))
        if not rows:
            continue

        key = {"lang": g["lang"], "model": g["model"],
               "condition": _condition(g["condition"], g["variant"]),
               "judge": g["judge"]}
        if has_split:
            key["split"] = g["split"]

        overall.append({**key, **summarise([r["evaluation"] for r in rows])})

        per_domain: Dict[str, List[dict]] = defaultdict(list)
        for r in rows:
            per_domain[r.get("row", {}).get("domain") or "unknown"].append(r["evaluation"])
        for domain, evals in per_domain.items():
            by_domain.append({**key, "domain": domain, **summarise(evals)})

    return overall, by_domain


def aggregate_metrics(models: Optional[set]) -> List[dict]:
    out = []
    for directory, glob in [(RESULTS / "comet", "comet_*.jsonl"),
                            (RESULTS / "chrf", "chrf_*.jsonl")]:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(glob)):
            m = METRIC_RE.match(path.name)
            if not m:
                continue
            g = m.groupdict()
            if models is not None and g["model"] not in models:
                continue
            rows = list(read_jsonl(path))
            scores = [r["comet_score"] if "comet_score" in r else r.get("chrf_score")
                      for r in rows]
            scores = [s for s in scores if s is not None]
            if not scores:
                continue
            out.append({
                "metric": g["metric"], "lang": g["lang"], "split": g["split"],
                "model": g["model"],
                "condition": _condition(g["condition"], g["variant"]),
                "n": len(scores), "score": mean(scores),
            })
    return out


def aggregate_win_rates(models: Optional[set]) -> List[dict]:
    directory = RESULTS / "win_rate"
    out = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("winrate_*.jsonl")):
        m = WINRATE_RE.match(path.name)
        if not m:
            continue
        g = m.groupdict()
        if models is not None and g["model"] not in models:
            continue
        rows = list(read_jsonl(path))
        if not rows:
            continue
        counts = defaultdict(int)
        for r in rows:
            counts[r.get("winner") or "unknown"] += 1
        n = len(rows)
        out.append({
            "lang": g["lang"], "split": g["split"], "model": g["model"], "n": n,
            "with_instructions": counts.get("with_instructions", 0),
            "without_instructions": counts.get("without_instructions", 0),
            "tie": counts.get("tie", 0),
            "win_rate_with_instructions": round(
                counts.get("with_instructions", 0) / n, 4) if n else None,
        })
    return out


SCORE_COLS = ["n", "n_critical_failure", "critical_failure_rate",
              "translation_rating", "adaptedness",
              "translation_rating_excl_critical", "adaptedness_excl_critical"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--all-models", action="store_true",
                   help="Include runs for models not reported in the paper.")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    models = None if args.all_models else PAPER_MODELS

    print("Aggregating BOUQuET judge outputs...")
    bq, bq_dom = aggregate_judge(
        RESULTS / "qualitative_analysis", BOUQUET_RE, has_split=True, models=models)
    write_csv(args.out_dir / "bouquet_scores.csv",
              ["split", "lang", "model", "condition", "judge"] + SCORE_COLS, bq)
    write_csv(args.out_dir / "bouquet_domain_scores.csv",
              ["split", "lang", "model", "condition", "judge", "domain"] + SCORE_COLS, bq_dom)

    print("Aggregating WMT24++ judge outputs...")
    wm, wm_dom = aggregate_judge(
        RESULTS / "wmt24pp_qualitative", WMT_RE, has_split=False, models=models)
    write_csv(args.out_dir / "wmt24pp_scores.csv",
              ["lang", "model", "condition", "judge"] + SCORE_COLS, wm)
    write_csv(args.out_dir / "wmt24pp_domain_scores.csv",
              ["lang", "model", "condition", "judge", "domain"] + SCORE_COLS, wm_dom)

    print("Aggregating COMET / ChrF++...")
    write_csv(args.out_dir / "metric_scores.csv",
              ["metric", "split", "lang", "model", "condition", "n", "score"],
              aggregate_metrics(models))

    print("Aggregating win rates...")
    write_csv(args.out_dir / "win_rates.csv",
              ["split", "lang", "model", "n", "with_instructions",
               "without_instructions", "tie", "win_rate_with_instructions"],
              aggregate_win_rates(models))

    print(f"\nDone. Score tables in {args.out_dir}/")


if __name__ == "__main__":
    main()
