"""Build the WMT24++ self-instruction table (LaTeX) for the appendix.

Reads eval files produced by judge/wmt24pp.py and emits the
LaTeX block for ``tab:self-instr-wmt-27b``: per-language None / Self-Instr / Full
+ % Gap closed, for Adaptedness and Translation rating, mirroring the BOUQuET
``tab:self-instr-gemma3``.

  None      = baseline, no instruction (use_instructions=False)
  Self-Instr= translated with model-drafted instruction (instruction_type=self-para)
  Full      = curated instruction (use_instructions=True)
  % Gap     = (Self-Instr - None) / (Full - None)

Usage:
    PYTHONPATH=. .venv/bin/python paper/extract_wmt24pp_self_instr.py
"""
from dotenv import load_dotenv
load_dotenv()

from judge.wmt24pp import Wmt24ppEvaluatedRow

MODEL = "gemma-3-27b-it"
EVAL_MODEL = "gemini/gemini-3-flash-preview"
LANGS = [("en-fr_FR", "French"), ("en-id_ID", "Indonesian"), ("en-uk_UA", "Ukrainian")]

CONDITIONS = [
    ("None",       dict(use_instructions=False)),
    ("Self-Instr", dict(use_instructions=True, instruction_type="self-para")),
    ("Full",       dict(use_instructions=True)),
]


def load_metrics(lang_code: str, **kwargs):
    path = Wmt24ppEvaluatedRow.get_savepath(
        model=MODEL, lang=lang_code, eval_model=EVAL_MODEL, **kwargs,
    )
    if not path.exists():
        return None
    rows = Wmt24ppEvaluatedRow.load(path)
    if not rows:
        return None
    m = Wmt24ppEvaluatedRow.compute_metrics(rows)
    return m.adaptedness_avg, m.translation_rating_avg


def fmt(x):
    return f"{x:.1f}" if x is not None else "--"


def gap_cell(none_v, self_v, full_v):
    """% of the None-to-Full gap closed by Self-Instr, colored like tab:self-instr-gemma3."""
    if None in (none_v, self_v, full_v):
        return "--"
    denom = full_v - none_v
    if denom <= 0:
        return "--"  # curated did not improve over None: ratio undefined
    pct = (self_v - none_v) / denom * 100
    if pct < 0:
        color = "dneg4"
    elif pct >= 70:
        color = "dpos8"
    elif pct >= 50:
        color = "dpos4"
    else:
        color = "dpos0"
    return f"\\cellcolor{{{color}}}{pct:.0f}\\%"


def render():
    # metrics[lang][cond] = (adapt, rating)
    metrics = {}
    for lang_code, _ in LANGS:
        metrics[lang_code] = {}
        for cond, kwargs in CONDITIONS:
            metrics[lang_code][cond] = load_metrics(lang_code, **kwargs)

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l rrrr rrrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{4}{c}{\textbf{Adaptedness}} & \multicolumn{4}{c}{\textbf{Translation rating}} \\")
    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(r"\textbf{Language} & \textbf{None} & \textbf{Self-Instr} & \textbf{Full} & \textbf{\% Gap} "
                 r"& \textbf{None} & \textbf{Self-Instr} & \textbf{Full} & \textbf{\% Gap} \\")
    lines.append(r"\midrule")

    for lang_code, lang_name in LANGS:
        m = metrics[lang_code]
        none_a, none_r = m["None"] or (None, None)
        self_a, self_r = m["Self-Instr"] or (None, None)
        full_a, full_r = m["Full"] or (None, None)
        lines.append(
            f"{lang_name} & {fmt(none_a)} & {fmt(self_a)} & {fmt(full_a)} & {gap_cell(none_a, self_a, full_a)} "
            f"& {fmt(none_r)} & {fmt(self_r)} & {fmt(full_r)} & {gap_cell(none_r, self_r, full_r)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Per-language self-instruction results with \textbf{Gemma-3-27B} on WMT24++ "
        r"(fra/ind/ukr), under \textbf{None} (no instruction), \textbf{Self-Instr} (self-generated "
        r"from the surrounding document), and \textbf{Full} (curated). \textbf{\% Gap} is the "
        r"fraction of the None-to-Full gap closed by Self-Instr. Translations were produced with "
        r"the self-instruction; adaptedness is judged against the curated instruction.}"
    )
    lines.append(r"\label{tab:self-instr-wmt-27b}")
    lines.append(r"\end{table*}")

    # Cross-language averages, for the prose (not part of the table).
    def avg(metric_idx, cond):
        xs = [metrics[lc][cond][metric_idx] for lc, _ in LANGS if metrics[lc][cond] is not None]
        return sum(xs) / len(xs) if xs else None

    summary = []
    for label, idx in [("Adaptedness", 0), ("Translation rating", 1)]:
        n, s, f = avg(idx, "None"), avg(idx, "Self-Instr"), avg(idx, "Full")
        if None not in (n, s, f) and f - n > 0:
            pct = (s - n) / (f - n) * 100
            summary.append(f"% {label}: None {n:.1f} / Self-Instr {s:.1f} / Full {f:.1f} "
                           f"-> {pct:.0f}% of gap closed")
        else:
            summary.append(f"% {label}: None {fmt(n)} / Self-Instr {fmt(s)} / Full {fmt(f)}")

    return "\n".join(lines) + "\n" + "\n".join(summary)


def main():
    print(render())


if __name__ == "__main__":
    main()
