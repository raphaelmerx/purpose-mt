"""Matched-accuracy (M-Acc) scoring for CoCoA-MT formality control.

Two matchers are provided:

``official``
    Byte-for-byte replication of the IWSLT2022 shared-task scorer
    (github.com/amazon-science/contrastive-controlled-mt, IWSLT2022/scorer.py): annotated
    phrases match a hypothesis if their whitespace tokens are a subset of the hypothesis
    tokens (or, for non-whitespace-delimited targets, if the phrase is a substring).
    Reported for comparability with prior work.

``robust``
    Same, plus two corrections. (1) Punctuation and non-breaking spaces are stripped before
    token matching: the official matcher fails on markers adjacent to punctuation, e.g. the
    French marker ``vous`` does not match the hypothesis token ``vous!``. (2) A matched phrase
    that is a substring of an opposing-register match is dropped: in Japanese the informal
    marker ``教えて`` is a prefix of the formal ``教えてください``, so a formal hypothesis
    always also "matches" informal.

Both matchers are validated against an oracle (the reference itself as the hypothesis) by
``--self-test``. Under ``official`` the oracle only reaches 85.6% coverage; under ``robust``
it reaches 96-99.5% for every language except Japanese (78.8%), and both score 100% accuracy
on decidable segments in both register directions.

M-Acc's denominator excludes NEUTRAL (no marker matched) and OTHER (both matched) segments, so
a model that hedges on hard segments scores well on a small denominator. Always read ``coverage``
next to ``m_acc``.

Usage:
    python -m controlled.score_cocoa --self-test
    python -m controlled.score_cocoa --model google/gemma-3-27b-it
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import string
from collections import Counter
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from controlled import cocoa_utils as C
from controlled.cocoa_utils import (FORMALITY_PHRASE, CocoaRow, get_savepath,
                                       load_predictions)

_PUNCT = "".join(set(string.punctuation) | set("¡¿«»„“”‘’—–…、。！？「」『』：；，"))
_SPACES = {"\xa0": " ", " ": " ", " ": " ", " ": " "}

FORMAL, INFORMAL, NEUTRAL, OTHER = "FORMAL", "INFORMAL", "NEUTRAL", "OTHER"


def _normalize(text: str) -> str:
    for src, dst in _SPACES.items():
        text = text.replace(src, dst)
    return text


def _tokens(text: str) -> List[str]:
    return [t for t in (tok.strip(_PUNCT) for tok in _normalize(text).split()) if t]


def matched_phrases(
    hypothesis: str, annotated_ref: str, whitespace_delimited: bool, robust: bool
) -> List[str]:
    """Annotated formality phrases from ``annotated_ref`` that occur in ``hypothesis``."""
    phrases = FORMALITY_PHRASE.findall(annotated_ref)
    if whitespace_delimited:
        if robust:
            hyp_tokens = set(_tokens(hypothesis))
            return [p for p in phrases if set(_tokens(p)).issubset(hyp_tokens)]
        hyp_tokens = set(hypothesis.split(" "))
        return [p for p in phrases if set(p.split(" ")).issubset(hyp_tokens)]
    if robust:
        hyp = _normalize(hypothesis)
        return [p for p in phrases if _normalize(p) in hyp]
    return [p for p in phrases if p in hypothesis]


def _drop_opposing_substrings(formal: List[str], informal: List[str]) -> tuple[list, list]:
    """Drop matches that are a substring of an opposing-register match."""
    keep_f = [f for f in formal if not any(f != i and f in i for i in informal)]
    keep_i = [i for i in informal if not any(i != f and i in f for f in formal)]
    return keep_f, keep_i


def predict_label(formal_matches: List[str], informal_matches: List[str]) -> str:
    if formal_matches and not informal_matches:
        return FORMAL
    if informal_matches and not formal_matches:
        return INFORMAL
    if not formal_matches and not informal_matches:
        return NEUTRAL
    return OTHER


def label_row(row: CocoaRow, hypothesis: str, robust: bool = True) -> str:
    """Formality label of ``hypothesis`` against this row's two annotated references."""
    ws = row.lang_pair not in C.NON_WHITESPACE_DELIMITED
    formal = matched_phrases(hypothesis, row.formal_annotated, ws, robust)
    informal = matched_phrases(hypothesis, row.informal_annotated, ws, robust)
    if robust:
        formal, informal = _drop_opposing_substrings(formal, informal)
    return predict_label(formal, informal)


class CocoaScore(BaseModel):
    lang_pair: str
    condition: str
    domain: str = "all"
    n: int
    decidable: int
    coverage: float
    labels: dict
    # accuracy among decidable segments w.r.t. the requested register; None when no register
    # was requested (the no-control condition C0)
    m_acc: Optional[float] = None
    # fraction of decidable segments rendered formally; the mechanism-test predictor
    formal_rate: float
    empty_rate: float


def score(
    rows: List[CocoaRow], condition: str, target_register: Optional[str], robust: bool = True,
    domain: str = "all",
) -> CocoaScore:
    """Score predictions already attached to ``rows`` as ``pred_text``."""
    labels = Counter()
    empty = 0
    for row in rows:
        hyp = (row.pred_text or "").strip()
        if not hyp:
            empty += 1
        labels[label_row(row, hyp, robust=robust)] += 1

    decidable = labels[FORMAL] + labels[INFORMAL]
    formal_rate = labels[FORMAL] / decidable if decidable else 0.0
    m_acc = None
    if target_register == "formal":
        m_acc = formal_rate
    elif target_register == "informal":
        m_acc = 1.0 - formal_rate if decidable else 0.0

    return CocoaScore(
        lang_pair=rows[0].lang_pair, condition=condition, domain=domain,
        n=len(rows), decidable=decidable,
        coverage=decidable / len(rows) if rows else 0.0,
        labels=dict(labels), m_acc=m_acc, formal_rate=formal_rate,
        empty_rate=empty / len(rows) if rows else 0.0,
    )


MANIPULATION_CHECK_THRESHOLD = 75.0


def manipulation_check(model: str, lang_pair: str, variant: str = "") -> Optional[tuple]:
    """Is M-Acc able to see this model's register choice in this language at all?

    Under the explicit categorical tag (C1) the model is *told* which register to use, so control
    accuracy should approach 100. If it sits near 50 (chance), M-Acc is not measuring register for
    this model-language pair and the language must be excluded from every formality analysis.

    Gemma-3-27B fails this for en-pt (55.0%): prompted with "Portuguese" it writes Brazilian
    Portuguese, where ``você`` + 3rd person is the everyday neutral form, while CoCoA's *European*
    Portuguese annotation marks exactly that conjugation as the formal variant. The model emits
    ``tu`` in 0/599 segments even when told to be informal. The variant mismatch was documented in
    the analysis plan before the run; this check detects it without reference to any outcome.

    The failure is an artifact of our prompt, not a limit of the model. Re-running en-pt with
    ``--lang-name 'Portuguese (from Portugal)' --fewshot-key pt_PT --variant ptPT`` lifts control
    accuracy 55.0 -> 97.6 and the C0 default formal rate 98.3 -> 52.9, with ``tu`` appearing in
    20 segments and ``você`` falling from 359 to 7. Pass ``variant="ptPT"`` to score that run.

    Note the fixed run may be used for the CoCoA control table but NOT for the mechanism test in
    correlate_wmt24pp.py: that test's outcome variable is WMT24++ pt_PT adaptedness, generated with
    the same region-stripped prompt (wmt24pp_utils.py drops the locale's region), so the predictor
    would measure a European-writing model and the outcome a Brazilian-writing one.

    Returns (control_accuracy, passed), or None when the C1 predictions are absent.
    """
    formal_path = get_savepath(model, lang_pair, "C1_tag_formal", variant)
    informal_path = get_savepath(model, lang_pair, "C1_tag_informal", variant)
    if not (formal_path.exists() and informal_path.exists()):
        return None
    f = score(load_predictions(formal_path), "C1_tag_formal", "formal")
    i = score(load_predictions(informal_path), "C1_tag_informal", "informal")
    if f.m_acc is None or i.m_acc is None:
        return None
    control_accuracy = 100 * (f.m_acc + i.m_acc) / 2
    return control_accuracy, control_accuracy >= MANIPULATION_CHECK_THRESHOLD


def _self_test() -> None:
    """Score each reference against itself. A correct matcher must return 100% accuracy."""
    print(f"{'lp':7s} {'matcher':9s} {'oracle cov':>10s} {'formal_acc':>11s} {'informal_acc':>13s}")
    failures = []
    for lang_pair in C.LANG_PAIRS:
        rows = C.load(lang_pair)
        for robust in (False, True):
            for reg, ref in (("formal", "formal_ref"), ("informal", "informal_ref")):
                for r in rows:
                    r.pred_text = getattr(r, ref)
                s = score(rows, "oracle", reg, robust=robust)
                if reg == "formal":
                    cov, facc = s.coverage, s.m_acc
                else:
                    iacc = s.m_acc
            name = "robust" if robust else "official"
            print(f"{lang_pair:7s} {name:9s} {cov:10.1%} {facc:11.1%} {iacc:13.1%}")
            if robust and (facc < 1.0 or iacc < 1.0):
                failures.append((lang_pair, facc, iacc))
    for r in C.load("en-fr"):
        r.pred_text = None
    assert not failures, f"robust matcher mislabels the oracle: {failures}"
    print("\nOK: robust matcher scores 100% on both oracle directions for every language.")


def main() -> None:
    p = argparse.ArgumentParser(description="Score CoCoA-MT formality control")
    p.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    p.add_argument("--self-test", action="store_true", help="validate the matchers against the oracle")
    p.add_argument("--matcher", choices=["official", "robust", "both"], default="both")
    p.add_argument("--per-domain", action="store_true")
    p.add_argument("--out", type=Path, default=Path("translation_results/controlled_mt/cocoa_scores.jsonl"))
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    from controlled.prompts import COCOA_CONDITIONS

    results: List[CocoaScore] = []
    for lang_pair in C.LANG_PAIRS:
        for cond in COCOA_CONDITIONS:
            path = C.get_savepath(args.model, lang_pair, cond.name)
            if not path.exists():
                continue
            rows = C.load_predictions(path)
            groups = {"all": rows}
            if args.per_domain:
                for d in C.DOMAINS:
                    groups[d] = [r for r in rows if r.domain == d]
            for dom, grp in groups.items():
                if grp:
                    results.append(score(grp, cond.name, cond.target_register, robust=True, domain=dom))

    if not results:
        print("No CoCoA-MT predictions found. Run controlled.translate first.")
        return

    print(f"{'lang':7s} {'condition':22s} {'n':>5s} {'cov':>7s} {'M-Acc':>7s} {'formal%':>8s} {'empty%':>7s}")
    for s in results:
        if s.domain != "all":
            continue
        macc = f"{s.m_acc:7.1%}" if s.m_acc is not None else "      -"
        print(f"{s.lang_pair:7s} {s.condition:22s} {s.n:5d} {s.coverage:7.1%} {macc} "
              f"{s.formal_rate:8.1%} {s.empty_rate:7.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in results:
            f.write(s.model_dump_json() + "\n")
    print(f"\nWrote {len(results)} scores to {args.out}")


if __name__ == "__main__":
    main()
