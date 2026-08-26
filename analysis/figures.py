"""Generate all figures for the ACL paper."""
import json
import statistics
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("translation_results")
QUAL_DIR = RESULTS_DIR / "qualitative_analysis"
COMET_DIR = RESULTS_DIR / "comet"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["fra_Latn", "ind_Latn", "ukr_Cyrl", "khm_Khmr", "jav_Latn"]
LANG_LABELS = {"fra_Latn": "French", "ind_Latn": "Indonesian", "ukr_Cyrl": "Ukrainian",
               "khm_Khmr": "Khmer", "jav_Latn": "Javanese"}
MODELS = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it", "gemma-4-31b-it"]
MODEL_LABELS = {"gemma-3-4b-it": "4B", "gemma-3-12b-it": "12B", "gemma-3-27b-it": "27B", "gemma-4-31b-it": "Gemma 4 31B"}
SPLIT = "dev"

# ACL-friendly colors
COLORS = {"4B": "#a6cee3", "12B": "#4a90d9", "27B": "#1f4e8c", "Gemma 4 31B": "#0a2647"}
COLOR_POS = "#2ca02c"  # green for positive delta
COLOR_NEG = "#d62728"  # red for negative delta

# Serif font (Nimbus Roman is a Times clone) so figures blend with the ACL paper body text.
PAPER_FONT_RC = {
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Liberation Serif', 'Times', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}
plt.rcParams.update(PAPER_FONT_RC)

def load_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]

def get_qual(lang, model, variant):
    fname = f"eval_bouquet_{lang}_{SPLIT}_{model}_{variant}_gemini-gemini-3-flash-preview.jsonl"
    rows = load_jsonl(QUAL_DIR / fname)
    if not rows:
        return None
    evals = [r["evaluation"] for r in rows]
    return {
        "rating": statistics.mean([e["translation_rating_score"] for e in evals]),
        "adaptedness": statistics.mean([e["adaptedness_score"] for e in evals]),
        "critical_pct": sum(1 for e in evals if e.get("is_critical_failure", False)) / len(evals) * 100,
        "domain_adaptedness": {r["row"]["domain"]: [] for r in rows},
    }

# Recompute with domain info
def get_qual_domains(lang, model, variant):
    fname = f"eval_bouquet_{lang}_{SPLIT}_{model}_{variant}_gemini-gemini-3-flash-preview.jsonl"
    rows = load_jsonl(QUAL_DIR / fname)
    if not rows:
        return {}
    domain_adapt = defaultdict(list)
    for r in rows:
        domain_adapt[r["row"]["domain"]].append(r["evaluation"]["adaptedness_score"])
    return {d: statistics.mean(v) for d, v in domain_adapt.items()}


# WMT24++ data loading
WMT24PP_DIR = RESULTS_DIR / "wmt24pp_qualitative"
WMT24PP_LANGUAGES = [
    "en-fr_FR", "en-id_ID", "en-uk_UA",
    "en-zh_CN", "en-ko_KR", "en-hi_IN", "en-ar_EG", "en-vi_VN",
    "en-th_TH", "en-zu_ZA", "en-gu_IN",
    "en-ja_JP", "en-de_DE", "en-pt_BR", "en-tr_TR", "en-sw_KE",
    "en-ru_RU", "en-es_MX", "en-it_IT", "en-pl_PL", "en-bn_IN",
    "en-fa_IR", "en-fil_PH", "en-he_IL", "en-ta_IN", "en-el_GR",
    "en-cs_CZ", "en-nl_NL", "en-ro_RO", "en-hu_HU", "en-sv_SE",
    "en-te_IN", "en-ur_PK", "en-sr_RS", "en-ca_ES", "en-no_NO",
    "en-bg_BG", "en-da_DK", "en-fi_FI", "en-kn_IN", "en-ml_IN",
    "en-mr_IN", "en-pa_IN", "en-pt_PT", "en-sk_SK", "en-sl_SI",
    "en-ar_SA", "en-et_EE", "en-fr_CA", "en-hr_HR", "en-is_IS",
    "en-lt_LT", "en-lv_LV", "en-sw_TZ",
]
WMT24PP_LANG_LABELS = {
    "en-fr_FR": "French", "en-id_ID": "Indonesian", "en-uk_UA": "Ukrainian",
    "en-zh_CN": "Chinese", "en-ko_KR": "Korean", "en-hi_IN": "Hindi",
    "en-ar_EG": "Arabic", "en-vi_VN": "Vietnamese", "en-th_TH": "Thai",
    "en-zu_ZA": "Zulu", "en-gu_IN": "Gujarati",
    "en-ja_JP": "Japanese", "en-de_DE": "German", "en-pt_BR": "Portuguese (BR)",
    "en-tr_TR": "Turkish", "en-sw_KE": "Swahili",
    "en-ru_RU": "Russian", "en-es_MX": "Spanish (MX)", "en-it_IT": "Italian",
    "en-pl_PL": "Polish", "en-bn_IN": "Bengali", "en-fa_IR": "Persian",
    "en-fil_PH": "Filipino", "en-he_IL": "Hebrew", "en-ta_IN": "Tamil",
    "en-el_GR": "Greek",
    "en-cs_CZ": "Czech", "en-nl_NL": "Dutch", "en-ro_RO": "Romanian",
    "en-hu_HU": "Hungarian", "en-sv_SE": "Swedish", "en-te_IN": "Telugu",
    "en-ur_PK": "Urdu", "en-sr_RS": "Serbian", "en-ca_ES": "Catalan",
    "en-no_NO": "Norwegian",
    "en-bg_BG": "Bulgarian", "en-da_DK": "Danish", "en-fi_FI": "Finnish",
    "en-kn_IN": "Kannada", "en-ml_IN": "Malayalam", "en-mr_IN": "Marathi",
    "en-pa_IN": "Punjabi", "en-pt_PT": "Portuguese (PT)", "en-sk_SK": "Slovak",
    "en-sl_SI": "Slovenian",
    "en-ar_SA": "Arabic (SA)", "en-et_EE": "Estonian", "en-fr_CA": "French (CA)",
    "en-hr_HR": "Croatian", "en-is_IS": "Icelandic", "en-lt_LT": "Lithuanian",
    "en-lv_LV": "Latvian", "en-sw_TZ": "Swahili (TZ)",
}

def get_qual_wmt24pp(lang, model, variant):
    # Try both eval model slugs, prefer the one with more rows
    slugs = ["models-gemini-3-flash-preview", "gemini-gemini-3-flash-preview"]
    rows = []
    for slug in slugs:
        fname = f"eval_wmt24pp_{lang}_{model}_{variant}_{slug}.jsonl"
        candidate = load_jsonl(WMT24PP_DIR / fname)
        if len(candidate) > len(rows):
            rows = candidate
    if not rows:
        return None
    evals = [r["evaluation"] for r in rows]
    return {
        "rating": statistics.mean([e["translation_rating_score"] for e in evals]),
        "adaptedness": statistics.mean([e["adaptedness_score"] for e in evals]),
        "critical_pct": sum(1 for e in evals if e.get("is_critical_failure", False)) / len(evals) * 100,
    }


# ============================================================
# Figure 1: Combined butterfly dumbbell plot (adaptedness + rating)
# ============================================================
print("Generating Figure 1: Combined dumbbell plot...")

from matplotlib.lines import Line2D

# Collect data: rows are (language, model) pairs
rows_data = []
for lang in LANGUAGES:
    for model in MODELS:
        wi = get_qual(lang, model, "with_instructions")
        wo = get_qual(lang, model, "without_instructions")
        if wi and wo:
            rows_data.append({
                "lang": LANG_LABELS[lang],
                "model": MODEL_LABELS[model],
                "without_adapt": wo["adaptedness"],
                "with_adapt": wi["adaptedness"],
                "delta_adapt": wi["adaptedness"] - wo["adaptedness"],
                "without_rating": wo["rating"],
                "with_rating": wi["rating"],
                "delta_rating": wi["rating"] - wo["rating"],
            })

n_rows = len(rows_data)
# Group models within a language (spacing 0.7) with larger gaps between languages (1.4)
n_models = len(MODELS)
y_positions = []
y = 0
for i in range(n_rows):
    y_positions.append(y)
    if (i + 1) % n_models == 0:  # end of a language group
        y += 1.4
    else:
        y += 0.7
y_positions = np.array(y_positions)

fig = plt.figure(figsize=(11, 5.5))
gs = fig.add_gridspec(1, 2, wspace=0.40)
ax_left = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)

def draw_dumbbell(ax, rows, y_pos, wo_key, wi_key, delta_key, label_outward='right'):
    """label_outward: visual direction for positive-delta labels ('right' or 'left')."""
    for i, row in enumerate(rows):
        y = y_pos[i]
        color = COLORS[row["model"]]
        wo_val, wi_val, delta = row[wo_key], row[wi_key], row[delta_key]
        text_color = COLOR_POS if delta >= 0 else COLOR_NEG
        ax.plot([wo_val, wi_val], [y, y], color=color, linewidth=1.5, zorder=1)
        ax.scatter(wo_val, y, s=40, facecolors='white', edgecolors=color, linewidths=1.2, zorder=2)
        ax.scatter(wi_val, y, s=40, facecolors=color, edgecolors=color, linewidths=1.2, zorder=2)
        # For positive delta, label goes outward; for negative, inward (opposite)
        go_right = (delta >= 0) == (label_outward == 'right')
        if go_right:
            ax.annotate(f'{delta:+.1f}', xy=(wi_val, y), xytext=(8, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=7.5, color=text_color)
        else:
            ax.annotate(f'{delta:+.1f}', xy=(wi_val, y), xytext=(-8, 0),
                        textcoords='offset points', ha='right', va='center',
                        fontsize=7.5, color=text_color)

# Left panel: adaptedness (x-axis reversed so high scores near center)
draw_dumbbell(ax_left, rows_data, y_positions, "without_adapt", "with_adapt", "delta_adapt", label_outward='left')
ax_left.set_xlim(108, 15)  # reversed
ax_left.set_xlabel('Adaptedness score', fontsize=9)
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)
ax_left.spines['left'].set_visible(False)
ax_left.invert_yaxis()

# Hide y-axis ticks on left panel; we'll place labels manually at the figure center
ax_left.set_yticks(y_positions)
ax_left.set_yticklabels([])
ax_left.tick_params(axis='y', length=0)

# Right panel: translation rating
draw_dumbbell(ax_right, rows_data, y_positions, "without_rating", "with_rating", "delta_rating")
ax_right.set_xlim(15, 108)
ax_right.set_xlabel('Translation rating', fontsize=9)
ax_right.spines['top'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.tick_params(axis='y', left=False, labelleft=False)

# Single legend in the bottom-left of the left panel
legend_elements = [
    Line2D([0], [0], marker='o', color='gray', markerfacecolor='white', markeredgecolor='gray',
           markersize=7.5, linewidth=0, label='Without instructions'),
    Line2D([0], [0], marker='o', color='gray', markerfacecolor='gray', markeredgecolor='gray',
           markersize=7.5, linewidth=0, label='With instructions'),
]
ax_left.legend(handles=legend_elements, fontsize=9, loc='upper right')

# Gutter labels: one bold language name per model group (vertically centred on
# its rows) plus the model size on each row. 31B is Gemma 4 (see footnote).
fig.canvas.draw()  # force layout so transforms are valid
left_right_edge = ax_left.get_position().x1
right_left_edge = ax_right.get_position().x0
x_center = (left_right_edge + right_left_edge) / 2

def _y_fig_norm(y_data):
    _, y_disp = ax_left.transData.transform((0, y_data))
    _, y_norm = fig.transFigure.inverted().transform((0, y_disp))
    return y_norm

# Model size per row, left-aligned just right of centre
for i, row in enumerate(rows_data):
    model_disp = row["model"].replace("Gemma 4 ", "")
    fig.text(x_center + 0.010, _y_fig_norm(y_positions[i]), model_disp,
             ha='left', va='center', fontsize=9, color='#444444')

# Language name once per group, bold, right-aligned just left of centre
for g in range(n_rows // n_models):
    idxs = list(range(g * n_models, (g + 1) * n_models))
    y_mid = float(np.mean(y_positions[idxs]))
    fig.text(x_center - 0.010, _y_fig_norm(y_mid), rows_data[idxs[0]]["lang"],
             ha='right', va='center', fontsize=11)

plt.savefig(FIGURES_DIR / "dumbbell.pdf", bbox_inches='tight', dpi=300)
plt.savefig(FIGURES_DIR / "dumbbell.png", bbox_inches='tight', dpi=300)
plt.close()
print("  Saved dumbbell.pdf")

# ============================================================
# Figure 2: Per-domain adaptedness improvement (27B, averaged across langs)
# ============================================================
print("Generating Figure 2: Domain breakdown...")

domains_all = set()
for lang in LANGUAGES:
    d = get_qual_domains(lang, "gemma-3-27b-it", "with_instructions")
    domains_all.update(d.keys())

domain_deltas = defaultdict(list)
domain_deltas_labeled = defaultdict(list)
for lang in LANGUAGES:
    wi = get_qual_domains(lang, "gemma-3-27b-it", "with_instructions")
    wo = get_qual_domains(lang, "gemma-3-27b-it", "without_instructions")
    for d in wi:
        if d in wo:
            delta = wi[d] - wo[d]
            domain_deltas[d].append(delta)
            domain_deltas_labeled[d].append((LANG_LABELS[lang], delta))

# Average across languages
domain_avg = {d: statistics.mean(v) for d, v in domain_deltas.items()}
sorted_domains = sorted(domain_avg, key=lambda x: domain_avg[x], reverse=True)

LANG_LETTERS = {"French": "F", "Indonesian": "I", "Ukrainian": "U", "Khmer": "K", "Javanese": "J"}
LANG_COLORS  = {"French": "#1f77b4", "Indonesian": "#ff7f0e", "Ukrainian": "#2ca02c", "Khmer": "#9467bd", "Javanese": "#d62728"}
lang_order = list(LANG_LETTERS.keys())

with plt.rc_context(PAPER_FONT_RC):
    fig, ax = plt.subplots(figsize=(7, 4.0))
    y_pos = np.arange(len(sorted_domains))
    vals = [domain_avg[d] for d in sorted_domains]
    # Single fill colour for all bars: Material Green 700, same hue family as
    # the paper's table palette and dark enough for the white letter-circles to pop.
    BAR_COLOR = '#388E3C'
    ax.barh(y_pos, vals, color=BAR_COLOR, edgecolor='white', linewidth=0.5)

    # Per-language letter labels on top of bars
    for i, d in enumerate(sorted_domains):
        for lang_label, delta in domain_deltas_labeled[d]:
            ax.scatter(delta, i, color='white', s=200, zorder=3, edgecolors='#cccccc', linewidths=0.3)
            ax.text(delta, i, LANG_LETTERS[lang_label], ha='center', va='center',
                    fontsize=11, fontweight='bold', color=LANG_COLORS[lang_label], zorder=4)

    # Language legend
    legend_patches = []
    for l in lang_order:
        legend_patches.append(ax.scatter([], [], color='white', s=120, edgecolors='#cccccc',
                                         linewidths=0.3, label=l))
    from matplotlib.offsetbox import AnnotationBbox, TextArea, HPacker, DrawingArea
    from matplotlib.patches import Circle
    import matplotlib.patches as mpatches

    class LetterCircleHandler:
        def __init__(self, letter, color):
            self.letter = letter
            self.color = color
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height
            cx, cy = x0 + width / 2, y0 + height / 2
            # Match the on-chart markers: scatter s=200 -> diameter sqrt(200) pt;
            # handlebox coordinates are in points, so radius is set directly.
            circle = mpatches.Circle((cx, cy), radius=(200 ** 0.5) / 2,
                                      facecolor='white', edgecolor='#cccccc', linewidth=0.5,
                                      transform=handlebox.get_transform())
            handlebox.add_artist(circle)
            txt = plt.Text(cx, cy, self.letter, ha='center', va='center',
                           fontsize=11, fontweight='bold', color=self.color,
                           transform=handlebox.get_transform())
            handlebox.add_artist(txt)
            return circle

    for p in legend_patches:
        p.remove()

    from matplotlib.lines import Line2D
    dummy_handles = [Line2D([], [], linestyle='') for l in lang_order]
    handler_map = {h: LetterCircleHandler(LANG_LETTERS[l], LANG_COLORS[l])
                   for h, l in zip(dummy_handles, lang_order)}
    ax.legend(handles=dummy_handles, labels=[l for l in lang_order],
              handler_map=handler_map, fontsize=16, loc='lower right', framealpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_domains, fontsize=16, color='#222222')
    ax.set_xlabel('')
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=14, colors='#222222')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "domain_adaptedness.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(FIGURES_DIR / "domain_adaptedness.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved domain_adaptedness.pdf")


# ============================================================
# Figure 5: WMT24++ dumbbell plot (adaptedness + rating)
# Grouped by register-richness; within-group order by informal resource rank.
# ============================================================
print("Generating Figure 3: WMT24++ dumbbell plot...")

# Register-rich: grammaticalised honorifics, diglossia, or elaborate pronoun/register systems
# Register-light: European (T-V or simpler, no honorific morphology)
REGISTER_GROUP = {
    # Register-rich, ordered by informal resource rank (higher resource first)
    "en-zh_CN": ("rich", 1),
    "en-ja_JP": ("rich", 2),
    "en-ko_KR": ("rich", 3),
    "en-ar_EG": ("rich", 4),
    "en-ar_SA": ("rich", 5),
    "en-hi_IN": ("rich", 6),
    "en-tr_TR": ("rich", 7),
    "en-vi_VN": ("rich", 8),
    "en-id_ID": ("rich", 9),
    "en-th_TH": ("rich", 10),
    "en-fa_IR": ("rich", 11),
    "en-he_IL": ("rich", 12),
    "en-bn_IN": ("rich", 13),
    "en-fil_PH": ("rich", 14),
    "en-ta_IN": ("rich", 15),
    "en-ur_PK": ("rich", 16),
    "en-te_IN": ("rich", 17),
    "en-sw_KE": ("rich", 18),
    "en-sw_TZ": ("rich", 19),
    "en-mr_IN": ("rich", 20),
    "en-ml_IN": ("rich", 21),
    "en-kn_IN": ("rich", 22),
    "en-pa_IN": ("rich", 23),
    "en-gu_IN": ("rich", 24),
    "en-zu_ZA": ("rich", 25),
    # Register-light (European), ordered by informal resource rank
    "en-fr_FR": ("light", 1),
    "en-es_MX": ("light", 2),
    "en-de_DE": ("light", 3),
    "en-ru_RU": ("light", 4),
    "en-pt_BR": ("light", 5),
    "en-it_IT": ("light", 6),
    "en-nl_NL": ("light", 7),
    "en-pl_PL": ("light", 8),
    "en-pt_PT": ("light", 9),
    "en-fr_CA": ("light", 10),
    "en-sv_SE": ("light", 11),
    "en-cs_CZ": ("light", 12),
    "en-ro_RO": ("light", 13),
    "en-el_GR": ("light", 14),
    "en-uk_UA": ("light", 15),
    "en-hu_HU": ("light", 16),
    "en-da_DK": ("light", 17),
    "en-no_NO": ("light", 18),
    "en-fi_FI": ("light", 19),
    "en-bg_BG": ("light", 20),
    "en-ca_ES": ("light", 21),
    "en-sr_RS": ("light", 22),
    "en-sk_SK": ("light", 23),
    "en-hr_HR": ("light", 24),
    "en-sl_SI": ("light", 25),
    "en-et_EE": ("light", 26),
    "en-lt_LT": ("light", 27),
    "en-lv_LV": ("light", 28),
    "en-is_IS": ("light", 29),
}

rows_wmt = []
WMT_MODEL = "gemma-3-27b-it"
for lang in WMT24PP_LANGUAGES:
    wi = get_qual_wmt24pp(lang, WMT_MODEL, "with_instructions")
    wo = get_qual_wmt24pp(lang, WMT_MODEL, "without_instructions")
    if wi and wo:
        grp, rank = REGISTER_GROUP[lang]
        rows_wmt.append({
            "lang": WMT24PP_LANG_LABELS[lang],
            "group": grp,
            "rank": rank,
            "model": MODEL_LABELS[WMT_MODEL],
            "without_adapt": wo["adaptedness"],
            "with_adapt": wi["adaptedness"],
            "delta_adapt": wi["adaptedness"] - wo["adaptedness"],
            "without_rating": wo["rating"],
            "with_rating": wi["rating"],
            "delta_rating": wi["rating"] - wo["rating"],
        })

# Sort: group (rich first), then resource rank (high -> low)
group_order = {"rich": 0, "light": 1}
rows_wmt.sort(key=lambda r: (group_order[r["group"]], r["rank"]))

# Build y-positions with a gap between groups
y_pos_wmt = []
y = 0.0
step = 0.75
group_gap = 1.1  # extra spacing between the two register groups
prev_group = None
group_bounds = {}  # group -> (y_min, y_max)
for r in rows_wmt:
    if prev_group is not None and r["group"] != prev_group:
        y += group_gap
    y_pos_wmt.append(y)
    lo, hi = group_bounds.get(r["group"], (y, y))
    group_bounds[r["group"]] = (min(lo, y), max(hi, y))
    prev_group = r["group"]
    y += step
y_pos_wmt = np.array(y_pos_wmt)

n_rows_wmt = len(rows_wmt)
fig_height = max(3.2, 0.22 * n_rows_wmt + 1.2)
fig = plt.figure(figsize=(11, fig_height))
gs = fig.add_gridspec(1, 2, wspace=0.40)
ax_left = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)

# Register-group background bands (drawn first so dumbbells sit on top)
GROUP_BAND = {"rich": "#e6efff", "light": "#f0f0f0"}
GROUP_LABEL = {"rich": "Register-rich (honorifics / diglossia)",
               "light": "Register-light (T-V distinction or simpler)"}
for grp, (lo, hi) in group_bounds.items():
    pad = 0.45
    for ax in (ax_left, ax_right):
        ax.axhspan(lo - pad, hi + pad, facecolor=GROUP_BAND[grp], edgecolor='none', zorder=0)

# Left panel: adaptedness (x-axis reversed)
draw_dumbbell(ax_left, rows_wmt, y_pos_wmt, "without_adapt", "with_adapt", "delta_adapt", label_outward='left')
ax_left.set_xlim(108, 15)
ax_left.set_xlabel('Adaptedness score', fontsize=9)
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)
ax_left.spines['left'].set_visible(False)
ax_left.invert_yaxis()
ax_left.set_yticks(y_pos_wmt)
ax_left.set_yticklabels([])
ax_left.tick_params(axis='y', length=0)

# Right panel: translation rating
draw_dumbbell(ax_right, rows_wmt, y_pos_wmt, "without_rating", "with_rating", "delta_rating")
ax_right.set_xlim(15, 108)
ax_right.set_xlabel('Translation rating', fontsize=9)
ax_right.spines['top'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.tick_params(axis='y', left=False, labelleft=False)

# Legend
_wmt_color = COLORS[MODEL_LABELS[WMT_MODEL]]
legend_elements = [
    Line2D([0], [0], marker='o', color=_wmt_color, markerfacecolor='white', markeredgecolor=_wmt_color,
           markersize=6, linewidth=0, label='Without instructions'),
    Line2D([0], [0], marker='o', color=_wmt_color, markerfacecolor=_wmt_color, markeredgecolor=_wmt_color,
           markersize=6, linewidth=0, label='With instructions'),
]
# Anchor legend to the top-left of the left panel, vertically aligned with
# the start of the first (register-rich) band.
ax_left.legend(handles=legend_elements, fontsize=7, loc='upper left',
               bbox_to_anchor=(0.0, 1.0), frameon=False,
               borderaxespad=0.2, handletextpad=0.4)

# Centered y-axis labels (language names between the two panels)
y_labels_wmt = [r['lang'] for r in rows_wmt]
fig.canvas.draw()
left_right_edge = ax_left.get_position().x1
right_left_edge = ax_right.get_position().x0
x_center = (left_right_edge + right_left_edge) / 2
for i, label in enumerate(y_labels_wmt):
    _, y_fig = ax_left.transData.transform((0, y_pos_wmt[i]))
    _, y_fig_norm = fig.transFigure.inverted().transform((0, y_fig))
    fig.text(x_center, y_fig_norm, label, ha='center', va='center', fontsize=7.5)

# Group labels at the top of each band, placed in the centre gutter
for grp, (lo, hi) in group_bounds.items():
    _, y_fig = ax_left.transData.transform((0, lo - 0.45))
    _, y_fig_norm = fig.transFigure.inverted().transform((0, y_fig))
    fig.text(x_center, y_fig_norm, GROUP_LABEL[grp],
             ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333333')

plt.savefig(FIGURES_DIR / "dumbbell_wmt24pp.pdf", bbox_inches='tight', dpi=300)
plt.savefig(FIGURES_DIR / "dumbbell_wmt24pp.png", bbox_inches='tight', dpi=300)
plt.close()
print("  Saved dumbbell_wmt24pp.pdf")

# ============================================================
# Figure 6: Instruction ablation box plot (27B, adaptedness by variant)
# ============================================================
print("Generating Figure 4: Instruction ablation box plot...")

ABLATION_VARIANTS = {
    "none": "without_instructions",
    "context": "with_instructions_context",
    "purpose": "with_instructions_purpose",
    "full": "with_instructions",
}
VARIANT_LABELS = ["None", "Context", "Purpose", "Full"]
VARIANT_COLORS = ["#bdbdbd", "#a6cee3", "#fdbf6f", "#1f4e8c"]

ablation_model = "gemma-3-27b-it"
fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(12, 3.2), sharey=True)

for ax_i, lang in enumerate(LANGUAGES):
    ax = axes[ax_i]
    box_data = []
    for vkey in ["none", "context", "purpose", "full"]:
        suffix = ABLATION_VARIANTS[vkey]
        fname = f"eval_bouquet_{lang}_{SPLIT}_{ablation_model}_{suffix}_gemini-gemini-3-flash-preview.jsonl"
        rows = load_jsonl(QUAL_DIR / fname)
        scores = [r["evaluation"]["adaptedness_score"] for r in rows] if rows else []
        box_data.append(scores)

    positions = np.arange(len(VARIANT_LABELS))
    bp = ax.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', linewidth=1.2))
    for patch, color in zip(bp['boxes'], VARIANT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Mean markers
    means = [np.mean(d) if d else 0 for d in box_data]
    ax.scatter(positions, means, marker='D', color='white', edgecolors='black',
               s=20, zorder=5, linewidths=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(VARIANT_LABELS, fontsize=7, rotation=30, ha='right')
    ax.set_title(LANG_LABELS[lang], fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ax_i == 0:
        ax.set_ylabel('Adaptedness score', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "ablation_boxplot.pdf", bbox_inches='tight', dpi=300)
plt.savefig(FIGURES_DIR / "ablation_boxplot.png", bbox_inches='tight', dpi=300)
plt.close()
print("  Saved ablation_boxplot.pdf")

print("\nAll figures generated!")
