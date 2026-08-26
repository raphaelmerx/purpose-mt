"""Extract all key results for the ACL paper."""
from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path
from collections import defaultdict
import statistics

RESULTS_DIR = Path("translation_results")
QUAL_DIR = RESULTS_DIR / "qualitative_analysis"
COMET_DIR = RESULTS_DIR / "comet"
CHRF_DIR = RESULTS_DIR / "chrf"
WINRATE_DIR = RESULTS_DIR / "win_rate"

LANGUAGES = ["fra_Latn", "ind_Latn", "ukr_Cyrl", "khm_Khmr", "jav_Latn"]
MODELS = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]
SPLIT = "dev"

LANG_NAMES = {
    "fra_Latn": "French",
    "ind_Latn": "Indonesian",
    "ukr_Cyrl": "Ukrainian",
    "khm_Khmr": "Khmer",
    "jav_Latn": "Javanese",
}

def load_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]

def extract_qual_metrics(lang, model, variant, split=SPLIT):
    """Extract qualitative analysis metrics."""
    fname = f"eval_bouquet_{lang}_{split}_{model}_{variant}_gemini-gemini-3-flash-preview.jsonl"
    rows = load_jsonl(QUAL_DIR / fname)
    if not rows:
        return None

    evals = [r["evaluation"] for r in rows]
    n = len(evals)
    critical = sum(1 for e in evals if e.get("is_critical_failure", False))
    ratings = [e["translation_rating_score"] for e in evals]
    adaptedness = [e["adaptedness_score"] for e in evals]

    # Per-domain breakdown
    domain_ratings = defaultdict(list)
    domain_adaptedness = defaultdict(list)
    domain_critical = defaultdict(lambda: [0, 0])  # [count, total]
    for r in rows:
        d = r["row"]["domain"]
        e = r["evaluation"]
        domain_ratings[d].append(e["translation_rating_score"])
        domain_adaptedness[d].append(e["adaptedness_score"])
        domain_critical[d][1] += 1
        if e.get("is_critical_failure", False):
            domain_critical[d][0] += 1

    return {
        "n": n,
        "critical_pct": critical / n * 100,
        "critical_count": critical,
        "avg_rating": statistics.mean(ratings),
        "avg_adaptedness": statistics.mean(adaptedness),
        "domain_ratings": {d: statistics.mean(v) for d, v in domain_ratings.items()},
        "domain_adaptedness": {d: statistics.mean(v) for d, v in domain_adaptedness.items()},
        "domain_critical_pct": {d: v[0] / v[1] * 100 for d, v in domain_critical.items()},
    }

def extract_comet(lang, model, variant, split=SPLIT):
    fname = f"comet_XCOMET-XL_bouquet_{lang}_{split}_{model}_{variant}.jsonl"
    rows = load_jsonl(COMET_DIR / fname)
    if not rows:
        return None
    scores = [r["comet_score"] for r in rows]
    domain_scores = defaultdict(list)
    for r in rows:
        domain_scores[r["domain"]].append(r["comet_score"])
    return {
        "avg": statistics.mean(scores),
        "n": len(scores),
        "domain_avg": {d: statistics.mean(v) for d, v in domain_scores.items()},
    }

def extract_chrf(lang, model, variant, split=SPLIT):
    fname = f"chrf_bouquet_{lang}_{split}_{model}_{variant}.jsonl"
    rows = load_jsonl(CHRF_DIR / fname)
    if not rows:
        return None
    scores = [r["chrf_score"] for r in rows]
    return {"avg": statistics.mean(scores), "n": len(scores)}

def extract_winrate(lang, model, split=SPLIT):
    fname = f"winrate_bouquet_{lang}_{split}_{model}.jsonl"
    rows = load_jsonl(WINRATE_DIR / fname)
    if not rows:
        return None
    n = len(rows)
    instr_wins = sum(1 for r in rows if r["winner"] == r["with_instr_position"])
    ties = sum(1 for r in rows if r["winner"] == "tie")
    return {
        "n": n,
        "instr_win_pct": instr_wins / n * 100,
        "tie_pct": ties / n * 100,
        "no_instr_win_pct": (n - instr_wins - ties) / n * 100 if "tie" in str([r.get("winner") for r in rows]) else (n - instr_wins) / n * 100,
    }

# ============================================================
# MAIN EXTRACTION
# ============================================================

print("=" * 80)
print("QUALITATIVE ANALYSIS: Critical Failure Rate, Avg Rating, Avg Adaptedness")
print("=" * 80)

for variant in ["with_instructions", "without_instructions"]:
    print(f"\n--- {variant.upper()} ---")
    for model in MODELS:
        print(f"\nModel: {model}")
        for lang in LANGUAGES:
            m = extract_qual_metrics(lang, model, variant)
            if m:
                print(f"  {LANG_NAMES.get(lang, lang):12s}: n={m['n']:3d}  crit={m['critical_pct']:5.1f}%  rating={m['avg_rating']:5.1f}  adapted={m['avg_adaptedness']:5.1f}")

# Also check semantic fewshot
print("\n\n--- SEMANTIC FEWSHOT ---")
for model in MODELS:
    print(f"\nModel: {model}")
    for lang in LANGUAGES:
        fname = f"eval_bouquet_{lang}_{SPLIT}_{model}_semantic_gemini-gemini-3-flash-preview.jsonl"
        rows = load_jsonl(QUAL_DIR / fname)
        if rows:
            evals = [r["evaluation"] for r in rows]
            n = len(evals)
            critical = sum(1 for e in evals if e.get("is_critical_failure", False))
            ratings = [e["translation_rating_score"] for e in evals]
            adaptedness = [e["adaptedness_score"] for e in evals]
            print(f"  {LANG_NAMES.get(lang, lang):12s}: n={n:3d}  crit={critical/n*100:5.1f}%  rating={statistics.mean(ratings):5.1f}  adapted={statistics.mean(adaptedness):5.1f}")

print("\n\n" + "=" * 80)
print("COMET SCORES")
print("=" * 80)

for variant in ["with_instructions", "without_instructions"]:
    print(f"\n--- {variant.upper()} ---")
    for model in MODELS:
        row_parts = []
        for lang in LANGUAGES:
            c = extract_comet(lang, model, variant)
            if c:
                row_parts.append(f"{LANG_NAMES.get(lang, lang)[:3]}={c['avg']:.4f}")
        if row_parts:
            print(f"  {model:35s}: {', '.join(row_parts)}")

print("\n\n" + "=" * 80)
print("CHRF SCORES")
print("=" * 80)

for variant in ["with_instructions", "without_instructions"]:
    print(f"\n--- {variant.upper()} ---")
    for model in MODELS:
        row_parts = []
        for lang in LANGUAGES:
            c = extract_chrf(lang, model, variant)
            if c:
                row_parts.append(f"{LANG_NAMES.get(lang, lang)[:3]}={c['avg']:.1f}")
        if row_parts:
            print(f"  {model:35s}: {', '.join(row_parts)}")

print("\n\n" + "=" * 80)
print("WIN RATES (instructions vs no-instructions)")
print("=" * 80)

for model in MODELS:
    row_parts = []
    for lang in LANGUAGES:
        w = extract_winrate(lang, model)
        if w:
            row_parts.append(f"{LANG_NAMES.get(lang, lang)[:3]}={w['instr_win_pct']:.1f}%")
    if row_parts:
        print(f"  {model:35s}: {', '.join(row_parts)}")

print("\n\n" + "=" * 80)
print("PER-DOMAIN BREAKDOWN (with_instructions, gemma-3-27b-it)")
print("=" * 80)

for lang in LANGUAGES:
    m = extract_qual_metrics(lang, "gemma-3-27b-it", "with_instructions")
    if m:
        print(f"\n{LANG_NAMES.get(lang, lang)}:")
        for d in sorted(m["domain_ratings"].keys()):
            print(f"  {d:40s}: rating={m['domain_ratings'][d]:5.1f}  adapted={m['domain_adaptedness'][d]:5.1f}  crit={m['domain_critical_pct'][d]:5.1f}%")

print("\n\n" + "=" * 80)
print("PER-DOMAIN BREAKDOWN (without_instructions, gemma-3-27b-it)")
print("=" * 80)

for lang in LANGUAGES:
    m = extract_qual_metrics(lang, "gemma-3-27b-it", "without_instructions")
    if m:
        print(f"\n{LANG_NAMES.get(lang, lang)}:")
        for d in sorted(m["domain_ratings"].keys()):
            print(f"  {d:40s}: rating={m['domain_ratings'][d]:5.1f}  adapted={m['domain_adaptedness'][d]:5.1f}  crit={m['domain_critical_pct'][d]:5.1f}%")


# Check Gemini Flash results
print("\n\n" + "=" * 80)
print("GEMINI FLASH RESULTS")
print("=" * 80)

for variant in ["with_instructions", "without_instructions"]:
    print(f"\n--- {variant.upper()} ---")
    for lang in LANGUAGES:
        m = extract_qual_metrics(lang, "gemini-3-flash-preview", variant)
        if m:
            print(f"  {LANG_NAMES.get(lang, lang):12s}: n={m['n']:3d}  crit={m['critical_pct']:5.1f}%  rating={m['avg_rating']:5.1f}  adapted={m['avg_adaptedness']:5.1f}")

# Win rates for Gemini Flash
print("\n--- GEMINI FLASH WIN RATES ---")
for lang in LANGUAGES:
    w = extract_winrate(lang, "gemini-3-flash-preview")
    if w:
        print(f"  {LANG_NAMES.get(lang, lang):12s}: instr_win={w['instr_win_pct']:.1f}%")

# COMET for semantic fewshot
print("\n\n" + "=" * 80)
print("COMET - SEMANTIC FEWSHOT")
print("=" * 80)
for model in MODELS:
    row_parts = []
    for lang in LANGUAGES:
        fname = f"comet_XCOMET-XL_bouquet_{lang}_{SPLIT}_{model}_semantic.jsonl"
        rows = load_jsonl(COMET_DIR / fname)
        if rows:
            scores = [r["comet_score"] for r in rows]
            row_parts.append(f"{LANG_NAMES.get(lang, lang)[:3]}={statistics.mean(scores):.4f}")
    if row_parts:
        print(f"  {model:35s}: {', '.join(row_parts)}")

# Dataset sizes
print("\n\n" + "=" * 80)
print("DATASET SIZES (n per lang/model)")
print("=" * 80)
for lang in LANGUAGES:
    for model in MODELS[:1]:
        m = extract_qual_metrics(lang, model, "with_instructions")
        if m:
            print(f"  {LANG_NAMES.get(lang, lang):12s}: n={m['n']}")

# Count domains
print("\n\n--- DOMAINS ---")
for lang in LANGUAGES:
    m = extract_qual_metrics(lang, "gemma-3-27b-it", "with_instructions")
    if m:
        print(f"  {LANG_NAMES.get(lang, lang):12s}: {len(m['domain_ratings'])} domains: {list(m['domain_ratings'].keys())}")
