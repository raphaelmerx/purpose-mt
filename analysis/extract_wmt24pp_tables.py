"""Build the WMT24++ Gemma-3-27B per-language table (LaTeX) for the appendix.

Reads eval files produced by judge/wmt24pp.py and emits the
LaTeX block for ``tab:full-wmt-27b`` (Baseline / Paragraph / Instructions across
fra, ind, ukr + Avg., for both Translation Rating and Adaptedness, with delta
rows colored using the same dpos/dneg color tokens as ``tab:full-27b``).

Usage:
    PYTHONPATH=. .venv/bin/python paper/extract_wmt24pp_full_results.py
"""
from dotenv import load_dotenv
load_dotenv()

from judge.wmt24pp import Wmt24ppEvaluatedRow

MODEL = "gemma-3-27b-it"
EVAL_MODEL = "gemini/gemini-3-flash-preview"
LANGS = [("en-fr_FR", "fra"), ("en-id_ID", "ind"), ("en-uk_UA", "ukr")]

CONDITIONS = [
    ("Baseline",      dict(use_instructions=False)),
    ("Paragraph",     dict(use_instructions=False, paragraph_level=True)),
    ("Paragraph-P2",  dict(use_instructions=False, paragraph_level=True, numbered_lists=True)),
    ("Instructions",  dict(use_instructions=True)),
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
    return m.translation_rating_avg, m.adaptedness_avg


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return f"{x:.1f}" if x is not None else "--"


def color_for_delta(d: float) -> str:
    """Match the dpos/dneg buckets used in tab:full-27b."""
    if d is None:
        return ""
    a = abs(d)
    if d >= 0:
        if a < 2:
            return "dpos0"
        if a < 5:
            return "dpos4"
        return "dpos8"
    else:
        if a < 2:
            return "dneg0"
        if a < 5:
            return "dneg4"
        return "dneg8"


def fmt_delta_cell(d):
    if d is None:
        return "--"
    color = color_for_delta(d)
    sign = "+" if d >= 0 else "--"
    val = f"{sign}{abs(d):.1f}"
    return f"\\cellcolor{{{color}}}{val}"


def bold_max(values, formatted):
    """Bold the formatted entry that holds the max numeric value (ignoring None)."""
    nums = [v for v in values if v is not None]
    if not nums:
        return formatted
    top = max(nums)
    out = []
    for v, f in zip(values, formatted):
        if v is not None and abs(v - top) < 1e-9:
            out.append(f"\\textbf{{{f}}}")
        else:
            out.append(f)
    return out


def render():
    # scores[cond] = (rating_per_lang, adapt_per_lang)
    scores = {}
    for cond, kwargs in CONDITIONS:
        rating, adapt = [], []
        for lang_code, _ in LANGS:
            res = load_metrics(lang_code, **kwargs)
            if res is None:
                rating.append(None)
                adapt.append(None)
            else:
                rating.append(res[0])
                adapt.append(res[1])
        scores[cond] = (rating, adapt)

    # Header / opening
    lines = []
    lines.append(r"\begin{table*}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l rrr r rrr r}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{4}{c}{\textbf{Translation Rating}} & \multicolumn{4}{c}{\textbf{Adaptedness}} \\")
    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    header_langs = " & ".join(rf"\textbf{{{short}}}" for _, short in LANGS)
    lines.append(rf"& {header_langs} & \textbf{{Avg.}} & {header_langs} & \textbf{{Avg.}} \\")
    lines.append(r"\midrule")

    # Compute column-wise max per metric for bolding
    rating_cols = list(zip(*(scores[c][0] for c, _ in CONDITIONS)))  # per-lang tuples across conditions
    adapt_cols = list(zip(*(scores[c][1] for c, _ in CONDITIONS)))
    rating_avgs = [avg(scores[c][0]) for c, _ in CONDITIONS]
    adapt_avgs = [avg(scores[c][1]) for c, _ in CONDITIONS]
    rating_max_per_lang = [max([v for v in col if v is not None], default=None) for col in rating_cols]
    adapt_max_per_lang = [max([v for v in col if v is not None], default=None) for col in adapt_cols]
    rating_max_avg = max([v for v in rating_avgs if v is not None], default=None)
    adapt_max_avg = max([v for v in adapt_avgs if v is not None], default=None)

    for cond_idx, (cond, _) in enumerate(CONDITIONS):
        rating, adapt = scores[cond]
        r_avg = avg(rating)
        a_avg = avg(adapt)

        def cell(v, is_max):
            s = fmt(v)
            return f"\\textbf{{{s}}}" if (v is not None and is_max is not None and abs(v - is_max) < 1e-9) else s

        rating_cells = [cell(v, rating_max_per_lang[i]) for i, v in enumerate(rating)]
        rating_avg_cell = cell(r_avg, rating_max_avg)
        adapt_cells = [cell(v, adapt_max_per_lang[i]) for i, v in enumerate(adapt)]
        adapt_avg_cell = cell(a_avg, adapt_max_avg)

        label = rf"\textbf{{{cond}}}" if cond == "Instructions" else cond
        lines.append(
            f"{label} & " + " & ".join(rating_cells) + f" & {rating_avg_cell} & "
            + " & ".join(adapt_cells) + f" & {adapt_avg_cell} \\\\"
        )

    lines.append(r"\midrule")

    base_rating, base_adapt = scores["Baseline"]
    short_map = {"Paragraph": "Para", "Paragraph-P2": "P2", "Instructions": "Instr"}
    for cond, _ in CONDITIONS:
        if cond == "Baseline":
            continue
        cur_r, cur_a = scores[cond]
        d_r = [(c - b) if (c is not None and b is not None) else None for c, b in zip(cur_r, base_rating)]
        d_a = [(c - b) if (c is not None and b is not None) else None for c, b in zip(cur_a, base_adapt)]
        d_r_avg = avg(d_r)
        d_a_avg = avg(d_a)
        short = short_map.get(cond, cond)
        lines.append(
            rf"$\Delta$ {short}$-$Base & "
            + " & ".join(fmt_delta_cell(v) for v in d_r) + f" & {fmt_delta_cell(d_r_avg)} & "
            + " & ".join(fmt_delta_cell(v) for v in d_a) + f" & {fmt_delta_cell(d_a_avg)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Gemma-3-27B on WMT24++ (997 items per cell). "
        r"Paragraph context is sentence-in-context with [START\_TRANSLATE] markers; "
        r"Instructions uses the gold per-sentence instruction. "
        r"Baseline uses a fixed 3-shot sentence-level prompt without instructions.}"
    )
    lines.append(r"\label{tab:full-wmt-27b}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    print(render())


if __name__ == "__main__":
    main()
