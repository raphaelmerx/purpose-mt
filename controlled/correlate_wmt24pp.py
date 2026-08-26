"""Mechanism test: does the model's default register explain its WMT24++ adaptedness baseline?

The paper's Discussion asserts, without testing it, that "LLMs default to the written register of
their pretraining distribution; instructions help most on conversational data because they nudge
the model off that default."

Two facts make that testable. First, across the languages that overlap CoCoA-MT, WMT24++
*instructed* adaptedness is a near-constant ceiling (sd < 1 point) while the *baseline* varies by
15 points, and that baseline variance is concentrated in the informal domains (social, speech)
while news is uniformly high. So ΔAdaptedness is, arithmetically, a measure of the un-instructed
default. Second, CoCoA-MT's no-control condition measures that default register directly, against
gold phrase-level annotations, and its M-Acc denominator conditions on segments that *carry* a
register marker. The predictor is therefore "given that a register must be chosen, how often does
the model choose formal", which is not confounded with whether a language grammaticalises register
at all.

Prediction: the formal-default rate correlates negatively with baseline adaptedness on social and
speech, and approximately zero on news, where a formal default is the correct choice.

Placebo predictor: MT-GenEval's masculine-default rate. Gender is not a register, so it should not
predict adaptedness on any domain.

Because the WMT24++ judge was run through two pipelines that disagree on identical inputs
(``gemini-...`` from judge/wmt24pp.py, ``models-...`` from the batch script),
``--judge-tag`` is explicit and a missing language is a hard error rather than a silent fallback.

Usage:
    python -m controlled.correlate_wmt24pp --model google/gemma-3-27b-it
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import statistics as st
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

from controlled import cocoa_utils as C
from controlled import geneval_utils as G
from controlled import score_cocoa, score_geneval

JUDGE_DIR = Path("translation_results/wmt24pp_qualitative")
DOMAINS = ["social", "speech", "literary", "news"]


def load_wmt24pp_adaptedness(locale: str, judge_tag: str, model: str) -> Dict[str, Dict[str, float]]:
    """Per-domain mean adaptedness for the baseline and instructed conditions."""
    model_name = model.split("/")[-1]
    out: Dict[str, Dict[str, float]] = {}
    for cond, instr in (("baseline", "without"), ("instructed", "with")):
        path = JUDGE_DIR / f"eval_wmt24pp_{locale}_{model_name}_{instr}_instructions_{judge_tag}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing judge output: {path}\n"
                f"Refusing to fall back to the other judge run: the two disagree by ~4 points "
                f"per item on identical inputs."
            )
        per_domain = defaultdict(list)
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                per_domain[rec["row"]["domain"]].append(rec["evaluation"]["adaptedness_score"])
        scores = {d: st.mean(v) for d, v in per_domain.items()}
        scores["all"] = st.mean(x for v in per_domain.values() for x in v)
        out[cond] = scores
    return out


def cocoa_formal_default_rate(model: str, condition: str,
                              cocoa_domain: Optional[str] = None) -> Dict[str, float]:
    """Fraction of decidable no-control outputs rendered in the formal register, per language.

    ``cocoa_domain`` restricts the measurement to one CoCoA domain. The model's default register is
    strongly domain-conditioned (Gemma-3-27B on German: 77.3% formal on call_center, 12.9% on
    telephony, 12.2% on topical_chat), so the aggregate averages over a 65-point within-language
    spread and is the wrong quantity to correlate against a per-domain outcome.
    """
    rates = {}
    for lang_pair in C.LANG_PAIRS:
        path = C.get_savepath(model, lang_pair, condition)
        if not path.exists():
            continue
        rows = C.load_predictions(path)
        if cocoa_domain:
            rows = [r for r in rows if r.domain == cocoa_domain]
        if not rows:
            continue
        rates[lang_pair] = score_cocoa.score(rows, condition, target_register=None).formal_rate
    return rates


# CoCoA domain -> the WMT24++ domain with the closest register expectations. topical_chat is
# written informal conversation (WMT24++ social); telephony is spoken conversation (speech).
# call_center is a formal service interaction with no WMT24++ analogue, so it serves as the
# placebo: a formal default there should not predict a *deficit* on informal WMT24++ text.
DOMAIN_PAIRS = [
    ("topical_chat", "baseline_social", "matched: written informal conversation"),
    ("telephony", "baseline_speech", "matched: spoken conversation"),
    ("call_center", "baseline_news", "placebo: formal domain both sides"),
]


def geneval_masculine_default_rate(model: str) -> Dict[str, float]:
    """Fraction of committed sentence-only outputs rendered with masculine forms, per language.

    Uses the X0 condition, where the model has no context and must guess the referent's gender.
    """
    rates = {}
    for lang in G.LANGS:
        path = G.get_savepath(model, lang, "contextual", "X0_sentence")
        if not path.exists():
            continue
        masc = committed = 0
        for r in G.load_context_predictions(path):
            correct_is_masc = r.gender == "masculine"
            # decide() against the masculine reference, whichever of the two it is
            masc_ref = r.ref_original if correct_is_masc else r.ref_flipped
            fem_ref = r.ref_flipped if correct_is_masc else r.ref_original
            used_masc, com, strict_masc = score_geneval.decide(r.pred_text or "", masc_ref, fem_ref)
            if com:
                committed += 1
                masc += strict_masc
        if committed:
            rates[lang] = masc / committed
    return rates


def _exact_or_sampled_perm_p(x: np.ndarray, y: np.ndarray, observed: float, n_iter: int = 100_000) -> float:
    """Two-sided permutation p-value for Spearman rho. Exact when n! is small."""
    n = len(x)
    count = total = 0
    if n <= 8:
        for perm in permutations(range(n)):
            rho = spearmanr(x, y[list(perm)]).statistic
            count += abs(rho) >= abs(observed) - 1e-12
            total += 1
    else:
        rng = np.random.default_rng(0)
        for _ in range(n_iter):
            rho = spearmanr(x, rng.permutation(y)).statistic
            count += abs(rho) >= abs(observed) - 1e-12
            total += 1
    return count / total


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10_000) -> tuple:
    rng = np.random.default_rng(0)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(x[idx])) < 3 or len(set(y[idx])) < 3:
            continue
        rhos.append(spearmanr(x[idx], y[idx]).statistic)
    if not rhos:
        return (float("nan"), float("nan"))
    return (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)))


def report(name: str, predictor: Dict[str, float], outcomes: Dict[str, Dict[str, float]],
           langs: List[str]) -> None:
    print(f"\n### {name}   (n={len(langs)} languages: {', '.join(langs)})\n")
    print(f"{'outcome':28s} {'rho':>7s} {'95% CI':>18s} {'perm p':>9s}")
    for key, label in [
        ("baseline_social", "baseline adaptedness: social"),
        ("baseline_speech", "baseline adaptedness: speech"),
        ("baseline_literary", "baseline adaptedness: literary"),
        ("baseline_news", "baseline adaptedness: news  [placebo]"),
        ("baseline_all", "baseline adaptedness: all"),
        ("delta_all", "delta adaptedness: all"),
    ]:
        y = np.array([outcomes[l][key] for l in langs], dtype=float)
        x = np.array([predictor[l] for l in langs], dtype=float)
        rho = spearmanr(x, y).statistic
        lo, hi = _bootstrap_ci(x, y)
        p = _exact_or_sampled_perm_p(x, y, rho)
        print(f"{label:28s} {rho:7.3f}  [{lo:6.3f}, {hi:6.3f}] {p:9.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="CoCoA default register vs WMT24++ adaptedness")
    p.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    p.add_argument("--judge-tag", type=str, default="models-gemini-3-flash-preview",
                   choices=["models-gemini-3-flash-preview", "gemini-gemini-3-flash-preview"])
    p.add_argument("--keep-all-langs", action="store_true",
                   help="do not apply the C1 manipulation-check validity gate")
    p.add_argument("--cocoa-condition", type=str, default="C0_3shot",
                   help="C0_3shot is prompt-matched to the WMT24++ baseline; C0_zeroshot is the robustness check")
    args = p.parse_args()

    predictor = cocoa_formal_default_rate(args.model, args.cocoa_condition)
    if not predictor:
        print(f"No CoCoA {args.cocoa_condition} predictions found. Run trans_controlled_vllm first.")
        return

    outcomes: Dict[str, Dict[str, float]] = {}
    for lang_pair in predictor:
        adapt = load_wmt24pp_adaptedness(C.WMT24PP_LOCALE[lang_pair], args.judge_tag, args.model)
        row = {f"baseline_{d}": adapt["baseline"].get(d, float("nan")) for d in DOMAINS}
        row["baseline_all"] = adapt["baseline"]["all"]
        row["delta_all"] = adapt["instructed"]["all"] - adapt["baseline"]["all"]
        outcomes[lang_pair] = row

    # Validity gate, applied before any correlation is computed and without reference to any
    # outcome: if the explicit tag cannot move M-Acc, M-Acc is not measuring register here.
    excluded = {}
    for lang_pair in sorted(predictor):
        # Deliberately the default (unmarked-prompt) run, not the ptPT variant that report_tables
        # uses. The outcome side, WMT24++ pt_PT adaptedness, was generated with the same
        # region-stripped prompt, so it measures a Brazilian-writing model; scoring the predictor
        # from a European-prompt run would pair two different systems. pt stays excluded here until
        # WMT24++ pt_PT is re-translated and re-judged with a region-matched prompt.
        check = score_cocoa.manipulation_check(args.model, lang_pair)
        if check and not check[1]:
            excluded[lang_pair] = check[0]
    if excluded and not args.keep_all_langs:
        for lp, acc in excluded.items():
            print(f"EXCLUDED {lp}: manipulation check {acc:.1f}% "
                  f"(< {score_cocoa.MANIPULATION_CHECK_THRESHOLD}%); M-Acc cannot see register for "
                  f"this model-language pair. Pass --keep-all-langs to override.")
            predictor.pop(lp)
        print()

    langs = sorted(predictor)
    print(f"Judge run: {args.judge_tag}   CoCoA condition: {args.cocoa_condition}\n")

    # Per-domain default register: the predictor actually used by the matched tests.
    by_domain = {d: {k: v for k, v in cocoa_formal_default_rate(args.model, args.cocoa_condition, d).items()
                     if k in predictor} for d in C.DOMAINS}
    print("Default formal% by CoCoA domain, and the WMT24++ baseline it is matched against:\n")
    print(f"{'lang':7s} {'chat':>7s} {'phone':>7s} {'call':>7s} | {'b_social':>9s} {'b_speech':>9s} {'b_news':>7s} {'dAdapt':>7s}")
    for l in langs:
        o = outcomes[l]
        print(f"{l:7s} {by_domain['topical_chat'].get(l, float('nan')):7.1%} "
              f"{by_domain['telephony'].get(l, float('nan')):7.1%} {by_domain['call_center'].get(l, float('nan')):7.1%} | "
              f"{o['baseline_social']:9.1f} {o['baseline_speech']:9.1f} {o['baseline_news']:7.1f} {o['delta_all']:+7.1f}")

    print("\n" + "=" * 78)
    print("PRIMARY: domain-matched tests. Prediction: negative on the matched pairs, null on the placebo.")
    print("=" * 78)
    for cocoa_dom, outcome_key, note in DOMAIN_PAIRS:
        pred = by_domain[cocoa_dom]
        ls = [l for l in langs if l in pred]
        x = np.array([pred[l] for l in ls])
        y = np.array([outcomes[l][outcome_key] for l in ls])
        rho = spearmanr(x, y).statistic
        lo, hi = _bootstrap_ci(x, y)
        p = _exact_or_sampled_perm_p(x, y, rho)
        print(f"\n  CoCoA {cocoa_dom:13s} -> WMT24++ {outcome_key.replace('baseline_',''):8s}  ({note})")
        print(f"    n={len(ls)}  rho={rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  permutation p={p:.4f}")

    print("\n" + "=" * 78)
    print("SECONDARY: aggregate predictor, as pre-specified in the analysis plan")
    print("=" * 78)
    report("Formal-default rate, averaged over CoCoA domains", predictor, outcomes, langs)

    placebo = geneval_masculine_default_rate(args.model)
    shared = sorted(set(C.WMT24PP_LOCALE) & {f"en-{l}" for l in placebo})
    if shared:
        pl = {lp: placebo[lp.split("-")[1]] for lp in shared}
        oc = {lp: outcomes[lp] for lp in shared if lp in outcomes}
        if len(oc) >= 5:
            report("Masculine-default rate (MT-GenEval, no context) [placebo predictor]",
                   pl, oc, sorted(oc))


if __name__ == "__main__":
    main()
