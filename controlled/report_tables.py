"""Emit the LaTeX tables for the controlled-MT appendix.

Usage:
    PYTHONPATH=. .venv/bin/python -m controlled.report_tables --model google/gemma-3-27b-it
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
from typing import Dict, List

from controlled import cocoa_utils as C
from controlled import geneval_utils as G
from controlled import score_cocoa, score_geneval
from controlled.prompts import (
    COCOA_CONDITIONS,
    GENEVAL_CONTEXT_CONDITIONS,
    GENEVAL_SENTENCE_CONDITIONS,
)

COCOA_LANG_LABEL = {lp: lp.split("-")[1] for lp in C.LANG_PAIRS}

# CoCoA annotates *European* Portuguese, but the prompt "Portuguese" makes Gemma-3-27B write
# Brazilian, whose você + 3rd person is exactly CoCoA's formal marker. The model then looks
# unsteerable (C1 control accuracy 55.0, chance 50) when it is simply writing another variant.
# We therefore report pt from the run whose prompt and few-shot examples match the annotation
# ('Portuguese (from Portugal)', pt_PT examples), where C1 reaches 97.6. Every other language is
# unaffected. See score_cocoa.manipulation_check.
COCOA_VARIANT = {"en-pt": "ptPT"}


def _cocoa_scores(model: str) -> Dict[str, Dict[str, score_cocoa.CocoaScore]]:
    out: Dict[str, Dict[str, score_cocoa.CocoaScore]] = {}
    for lang_pair in C.LANG_PAIRS:
        variant = COCOA_VARIANT.get(lang_pair, "")
        for cond in COCOA_CONDITIONS:
            path = C.get_savepath(model, lang_pair, cond.name, variant)
            if not path.exists():
                continue
            rows = C.load_predictions(path)
            out.setdefault(lang_pair, {})[cond.name] = score_cocoa.score(
                rows, cond.name, cond.target_register, robust=True
            )
    return out


def table_cocoa_control(scores, model: str) -> str:
    """Control accuracy per condition: can the model be steered into a target register?"""
    langs = [lp for lp in C.LANG_PAIRS if lp in scores]

    # The manipulation check is a hard gate: under the explicit tag C1 the model is *told* the
    # register, so a language sitting near chance means M-Acc cannot see register there and the
    # column is meaningless. Fail loudly rather than print it.
    failed = {lp for lp in langs
              if (chk := score_cocoa.manipulation_check(model, lp, COCOA_VARIANT.get(lp, "")))
              and not chk[1]}
    if failed:
        raise SystemExit(
            f"manipulation check failed for {sorted(failed)}; M-Acc cannot see register for "
            f"these languages. Fix the prompt or drop the column before reporting."
        )

    header = " & ".join(COCOA_LANG_LABEL[lp] for lp in langs)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{adjustbox}{max width=\columnwidth}",
        r"\begin{tabular}{l " + "r" * len(langs) + "}",
        r"\toprule",
        f"\\textbf{{Condition}} & {header} \\\\",
        r"\midrule",
    ]

    def control_acc(lp: str, formal_cond: str, informal_cond: str) -> float:
        f, i = scores[lp].get(formal_cond), scores[lp].get(informal_cond)
        if not f or not i or f.m_acc is None or i.m_acc is None:
            return float("nan")
        return 100 * (f.m_acc + i.m_acc) / 2

    def fmt(v):
        return "--" if v != v else f"{v:.1f}"

    # C3b reuses C3's informal direction: only the topical_chat formal cell differs.
    rows = [
        ("C1 categorical tag", "C1_tag_formal", "C1_tag_informal"),
        ("C2 explicit brief", "C2_brief_formal", "C2_brief_informal"),
        ("C3 implicit scenario", "C3_scenario_formal", "C3_scenario_informal"),
        ("C3b implicit scenario\\smark", "C3b_scenario_formal", "C3_scenario_informal"),
    ]
    for label, fc, ic in rows:
        cells = " & ".join(fmt(control_acc(lp, fc, ic)) for lp in langs)
        lines.append(f"{label} & {cells} \\\\")

    lines.append(r"\midrule")
    for label, cond in [("C0 formal default (0-shot)", "C0_zeroshot"),
                        ("C0 formal default (3-shot)", "C0_3shot")]:
        cells = " & ".join(
            f"{100 * scores[lp][cond].formal_rate:.1f}" if cond in scores[lp] else "--"
            for lp in langs
        )
        lines.append(f"{label} & {cells} \\\\")

    lines.append(r"\midrule")
    cells = " & ".join(
        f"{100 * scores[lp]['C3_scenario_formal'].coverage:.0f}"
        if "C3_scenario_formal" in scores[lp] else "--" for lp in langs
    )
    lines.append(f"Coverage (\\%) & {cells} \\\\")

    lines += [
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
        r"\caption{"
        r"CoCoA-MT formality accuracy, Gemma-3-27B. Mean of M-Acc under a formal and an informal target "
        r"(50 = chance). C3 and C3b never name the register; they describe the social situation only. "
        r"The C0 rows give the fraction of un-instructed outputs rendered formally, i.e.\ the model's "
        r"default register. Coverage is the share of segments that M-Acc can score, i.e.\ those whose "
        r"output contains one of CoCoA's annotated formality markers; on the rest the output commits to "
        r"neither register and is dropped from the M-Acc denominator (reported for C3). \smark~C3b is "
        r"C3 with a single prompt changed. C3's formal \textit{topical\_chat} scenario, a public "
        r"discussion forum between strangers, does not imply formality in most of these languages, so "
        r"C3b substitutes a professional-conference setting; every other scenario, including all three "
        r"informal ones, is unchanged; see \S\ref{par:lang-specific-norms}." r"}",
        r"\label{tab:cocoa-control}", r"\end{table}",
    ]
    return "\n".join(lines)


def table_geneval_context(model: str) -> str:
    """Instruction vs paragraph context vs self-instruction, on the contextual subset."""
    langs, data = [], {}
    for lang in G.LANGS:
        cell = {}
        for cond in GENEVAL_CONTEXT_CONDITIONS:
            path = G.get_savepath(model, lang, "contextual", cond)
            if path.exists():
                cell[cond] = score_geneval.score_contextual(G.load_context_predictions(path), cond)
        if cell:
            langs.append(lang)
            data[lang] = cell

    header = " & ".join(langs)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{adjustbox}{max width=\columnwidth}",
        r"\begin{tabular}{l " + "r" * len(langs) + "r}",
        r"\toprule",
        f"\\textbf{{Condition}} & {header} & \\textbf{{Avg.}} \\\\",
        r"\midrule",
    ]
    labels = {
        "X0_sentence": "Sentence only",
        "X1_paragraph": "Paragraph context ($P_1$)",
        "X2_brief": r"\textbf{Instruction}",
        "X3_self_instruct": "Self-instruction",
    }
    for cond, label in labels.items():
        vals = [100 * data[l][cond].strict_accuracy for l in langs if cond in data[l]]
        if not vals:
            continue
        cells = " & ".join(f"{100 * data[l][cond].strict_accuracy:.1f}" for l in langs)
        lines.append(f"{label} & {cells} & {sum(vals) / len(vals):.1f} \\\\")

    lines += [
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
        r"\caption{MT-GenEval contextual subset, strict gender accuracy. The gender cue appears "
        r"only in the preceding sentence, so the sentence-only condition must guess. Strict "
        r"accuracy requires the hypothesis to commit to the correct gender; the official metric "
        r"scores a non-translation as correct.}",
        r"\label{tab:geneval-context}", r"\end{table}",
    ]
    return "\n".join(lines)


def table_geneval_counterfactual(model: str) -> str:
    langs, data = [], {}
    for lang in G.LANGS:
        cell = {}
        for cond in GENEVAL_SENTENCE_CONDITIONS:
            path = G.get_savepath(model, lang, "counterfactual", cond)
            if path.exists():
                cell[cond] = score_geneval.score_counterfactual(G.load_sentence_predictions(path), cond)
        if cell:
            langs.append(lang)
            data[lang] = cell

    header = " & ".join(langs)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{adjustbox}{max width=\columnwidth}",
        r"\begin{tabular}{ll " + "r" * len(langs) + "}",
        r"\toprule",
        f"\\textbf{{Condition}} & \\textbf{{Metric}} & {header} \\\\",
        r"\midrule",
    ]
    for cond, label in [("G0_baseline", "Baseline"), ("G1_brief", "Gender brief")]:
        for attr, mlabel in [("strict_accuracy", "Strict acc."),
                             ("official_accuracy", "Official acc."),
                             ("commit_rate", "Commit rate")]:
            cells = " & ".join(
                f"{100 * getattr(data[l][cond], attr):.1f}" if cond in data[l] else "--"
                for l in langs
            )
            lines.append(f"{label if attr == 'strict_accuracy' else ''} & {mlabel} & {cells} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"

    lines += [
        r"\end{tabular}", r"\end{adjustbox}",
        r"\caption{MT-GenEval counterfactual subset. Official accuracy is the metric of "
        r"\citet{currey-etal-2022-mtgeneval}, which marks any hypothesis without a wrong-gender "
        r"word as correct and therefore rewards non-translation. Strict accuracy additionally "
        r"requires a correct-gender word.}",
        r"\label{tab:geneval-counterfactual}", r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    args = p.parse_args()

    scores = _cocoa_scores(args.model)
    if scores:
        print(table_cocoa_control(scores, args.model))
        print()
    print(table_geneval_counterfactual(args.model))
    print()
    print(table_geneval_context(args.model))


if __name__ == "__main__":
    main()
