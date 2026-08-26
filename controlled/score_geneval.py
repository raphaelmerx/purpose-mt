"""Gender-accuracy scoring for MT-GenEval, with a guard against degenerate systems.

The official metric (github.com/amazon-science/machine-translation-gender-eval,
``accuracy_metric.py``) marks a hypothesis Correct whenever it contains no word unique to the
*wrong*-gender reference. It never checks that the hypothesis committed to the right gender, so
a system that fails to translate scores near-perfectly. Measured on en-fr counterfactual:

    oracle (correct reference)       official 1.000   strict 1.000
    empty output                     official 1.000   strict 0.000
    English source passed through    official 0.993   strict 0.143

This matters for any model with a non-trivial critical-failure rate, and for any model that
hedges by avoiding gendered forms. We therefore report three numbers:

``official_accuracy``  no wrong-gender word. Replicates Currey et al. (2022); comparable to prior work.
``commit_rate``        the hypothesis contains at least one gender-marked word from either reference.
``strict_accuracy``    committed to the correct gender AND used no wrong-gender word. Headline.

Usage:
    python -m controlled.score_geneval --self-test
    python -m controlled.score_geneval --model google/gemma-3-27b-it
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import string
from pathlib import Path
from typing import List, Optional, Set, Tuple

from pydantic import BaseModel

from controlled import geneval_utils as G

_STRIP_PUNCT = str.maketrans(string.punctuation, " " * len(string.punctuation))


def clean_line(line: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace (official preprocessing)."""
    return line.lower().translate(_STRIP_PUNCT).strip()


def _words(line: str) -> Set[str]:
    return set(clean_line(line).split())


def decide(hypothesis: str, correct_ref: str, counterfactual_ref: str) -> Tuple[bool, bool, bool]:
    """Return (official_correct, committed, strict_correct) for one hypothesis."""
    hyp, cor, ctf = _words(hypothesis), _words(correct_ref), _words(counterfactual_ref)
    cor_unique, ctf_unique = cor - ctf, ctf - cor

    used_correct = bool(hyp & cor_unique)
    used_wrong = bool(hyp & ctf_unique)

    official = not used_wrong
    committed = used_correct or used_wrong
    strict = used_correct and not used_wrong
    return official, committed, strict


class GenevalScore(BaseModel):
    lang: str
    subset: str
    condition: str
    n: int
    official_accuracy: float
    strict_accuracy: float
    commit_rate: float
    empty_rate: float


def score_counterfactual(rows: List[G.GenevalSentenceRow], condition: str) -> GenevalScore:
    """Official semantics: an item counts as Correct only if both gender variants are Correct."""
    official = strict = commit = empty = 0
    for r in rows:
        hm, hf = (r.pred_masculine or ""), (r.pred_feminine or "")
        om, cm, sm = decide(hm, r.ref_masculine, r.ref_feminine)
        of, cf, sf = decide(hf, r.ref_feminine, r.ref_masculine)
        official += om and of
        strict += sm and sf
        commit += cm and cf
        empty += (not hm.strip()) or (not hf.strip())
    n = len(rows)
    return GenevalScore(
        lang=rows[0].lang, subset="counterfactual", condition=condition, n=n,
        official_accuracy=official / n, strict_accuracy=strict / n,
        commit_rate=commit / n, empty_rate=empty / n,
    )


def score_contextual(rows: List[G.GenevalContextRow], condition: str) -> GenevalScore:
    official = strict = commit = empty = 0
    for r in rows:
        hyp = r.pred_text or ""
        o, c, s = decide(hyp, r.ref_original, r.ref_flipped)
        official += o
        strict += s
        commit += c
        empty += not hyp.strip()
    n = len(rows)
    return GenevalScore(
        lang=rows[0].lang, subset="contextual", condition=condition, n=n,
        official_accuracy=official / n, strict_accuracy=strict / n,
        commit_rate=commit / n, empty_rate=empty / n,
    )


def _self_test() -> None:
    """Degenerate systems must be caught by strict_accuracy but not by official_accuracy."""
    lang = "fr"
    rows = G.load_sentences(lang)
    en_masc = [r.source_masculine for r in rows]

    systems = {
        "oracle (correct refs)": ([r.ref_masculine for r in rows], [r.ref_feminine for r in rows]),
        "empty output": ([""] * len(rows), [""] * len(rows)),
        "English passthrough": (en_masc, [r.source_feminine for r in rows]),
        "always masculine": ([r.ref_masculine for r in rows], [r.ref_masculine for r in rows]),
    }
    print(f"{'system':24s} {'official':>9s} {'commit':>8s} {'strict':>8s}")
    results = {}
    for name, (pm, pf) in systems.items():
        for r, m, f in zip(rows, pm, pf):
            r.pred_masculine, r.pred_feminine = m, f
        s = score_counterfactual(rows, "self-test")
        results[name] = s
        print(f"{name:24s} {s.official_accuracy:9.3f} {s.commit_rate:8.3f} {s.strict_accuracy:8.3f}")

    assert results["oracle (correct refs)"].strict_accuracy > 0.95, "oracle should pass strict"
    assert results["empty output"].official_accuracy == 1.0, "official metric should be fooled by empty"
    assert results["empty output"].strict_accuracy == 0.0, "strict must reject empty"
    assert results["English passthrough"].official_accuracy > 0.9, "official metric should be fooled"
    assert results["English passthrough"].strict_accuracy < 0.3, "strict must reject passthrough"
    assert results["always masculine"].strict_accuracy < 0.1, "strict must reject gender-blind system"
    print("\nOK: strict_accuracy rejects empty, passthrough and gender-blind systems;")
    print("    official_accuracy is fooled by all three, as documented.")


def main() -> None:
    p = argparse.ArgumentParser(description="Score MT-GenEval gender accuracy")
    p.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out", type=Path, default=Path("translation_results/controlled_mt/geneval_scores.jsonl"))
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    from controlled.prompts import GENEVAL_CONTEXT_CONDITIONS, GENEVAL_SENTENCE_CONDITIONS

    results: List[GenevalScore] = []
    for lang in G.LANGS:
        for cond in GENEVAL_SENTENCE_CONDITIONS:
            path = G.get_savepath(args.model, lang, "counterfactual", cond)
            if path.exists():
                results.append(score_counterfactual(G.load_sentence_predictions(path), cond))
        for cond in GENEVAL_CONTEXT_CONDITIONS:
            path = G.get_savepath(args.model, lang, "contextual", cond)
            if path.exists():
                results.append(score_contextual(G.load_context_predictions(path), cond))

    if not results:
        print("No MT-GenEval predictions found. Run controlled.translate first.")
        return

    print(f"{'lang':5s} {'subset':14s} {'condition':18s} {'n':>5s} {'official':>9s} {'strict':>8s} {'commit':>8s}")
    for s in results:
        print(f"{s.lang:5s} {s.subset:14s} {s.condition:18s} {s.n:5d} "
              f"{s.official_accuracy:9.3f} {s.strict_accuracy:8.3f} {s.commit_rate:8.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in results:
            f.write(s.model_dump_json() + "\n")
    print(f"\nWrote {len(results)} scores to {args.out}")


if __name__ == "__main__":
    main()
