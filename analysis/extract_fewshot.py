"""Extract few-shot vs instructions comparison data for all languages."""
import json
from pathlib import Path

LANGS = ["fra_Latn", "ind_Latn", "ukr_Cyrl", "khm_Khmr", "jav_Latn"]
LANG_NAMES = {"fra_Latn": "French", "ind_Latn": "Indonesian", "ukr_Cyrl": "Ukrainian", "khm_Khmr": "Khmer", "jav_Latn": "Javanese"}
MODELS = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]
EVAL_MODEL = "gemini-gemini-3-flash-preview"

def load_eval(lang, model, suffix):
    path = Path(f"translation_results/qualitative_analysis/eval_bouquet_{lang}_dev_{model}_{suffix}_{EVAL_MODEL}.jsonl")
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # Handle nested schema: evaluation.translation_rating_score
    ratings, adapt, crits = [], [], []
    for r in rows:
        ev = r.get("evaluation", r)  # try nested, fall back to flat
        if ev.get("translation_rating_score") is not None:
            ratings.append(ev["translation_rating_score"])
        if ev.get("adaptedness_score") is not None:
            adapt.append(ev["adaptedness_score"])
        if ev.get("is_critical_failure"):
            crits.append(1)
    return {
        "rating": sum(ratings) / len(ratings) if ratings else 0,
        "adaptedness": sum(adapt) / len(adapt) if adapt else 0,
        "critical": len(crits) / len(rows) * 100 if rows else 0,
        "n": len(rows)
    }

print("=" * 100)
print("FEW-SHOT vs INSTRUCTIONS vs BASELINE — ALL LANGUAGES")
print("=" * 100)

for model in MODELS:
    print(f"\n{'='*80}")
    print(f"Model: {model}")
    print(f"{'='*80}")
    print(f"{'Lang':<12} {'Base Adp':>9} {'FS Adp':>9} {'Instr Adp':>10} | {'FS-Base':>8} {'Instr-Base':>10} {'Instr-FS':>9}")
    print("-" * 80)

    for lang in LANGS:
        baseline = load_eval(lang, model, "without_instructions")
        fewshot = load_eval(lang, model, "without_instructions_semantic")
        instruct = load_eval(lang, model, "with_instructions")

        b_adp = baseline["adaptedness"] if baseline else None
        f_adp = fewshot["adaptedness"] if fewshot else None
        i_adp = instruct["adaptedness"] if instruct else None

        b_str = f"{b_adp:>9.1f}" if b_adp else f"{'N/A':>9}"
        f_str = f"{f_adp:>9.1f}" if f_adp else f"{'N/A':>9}"
        i_str = f"{i_adp:>10.1f}" if i_adp else f"{'N/A':>10}"

        fs_d = f"{f_adp - b_adp:>+8.1f}" if (f_adp and b_adp) else f"{'N/A':>8}"
        ib_d = f"{i_adp - b_adp:>+10.1f}" if (i_adp and b_adp) else f"{'N/A':>10}"
        if_d = f"{i_adp - f_adp:>+9.1f}" if (i_adp and f_adp) else f"{'N/A':>9}"

        print(f"{LANG_NAMES[lang]:<12} {b_str} {f_str} {i_str} | {fs_d} {ib_d} {if_d}")

print("\n\n")
print("=" * 100)
print("FOR LATEX TABLE: Rating and Adaptedness per condition")
print("=" * 100)

for model in MODELS:
    print(f"\nModel: {model}")
    print(f"{'Lang':<12} | {'Condition':<15} | {'Rating':>7} | {'Adapt':>7} | {'Crit%':>6}")
    print("-" * 60)
    for lang in LANGS:
        for suffix, label in [("without_instructions", "Baseline"),
                               ("without_instructions_semantic", "Few-shot"),
                               ("with_instructions", "Instructions")]:
            data = load_eval(lang, model, suffix)
            if data:
                print(f"{LANG_NAMES[lang]:<12} | {label:<15} | {data['rating']:>7.1f} | {data['adaptedness']:>7.1f} | {data['critical']:>5.1f}%")
            else:
                print(f"{LANG_NAMES[lang]:<12} | {label:<15} | {'N/A':>7} | {'N/A':>7} | {'N/A':>6}")
    print()
