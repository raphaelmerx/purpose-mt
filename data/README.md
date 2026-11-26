# Data

No data is stored in this repository. Everything lives in a [gated HuggingFace dataset](https://huggingface.co/datasets/raphaelmerx/purpose-mt) and is fetched with:

```bash
python -m data.download
```

## What lands where

`download.py` mirrors the scripts layout:

| Group | Destination | Contents |
|---|---|---|
| `instructions` | `datasets/` | Generated user instructions for BOUQuET, WMT24++ and SMOL |
| `annotations` | `annotation_exports/` | LLM-judge and XCOMET-XL scores over the 160 hand-rated items × 5 languages. The human ratings themselves are released in aggregate only, as the `human_*.csv` files under `scores/` |
| `scores` | `data/scores/` | Aggregated per-condition means behind every table and figure |
| `controlled` | `translation_results/controlled_mt/` | CoCoA-MT and MT-GenEval aggregate scores |

Fetch a subset with `--only`, e.g. `python -m data.download --only scores`, which is
enough to check any number in the paper.

All four destinations are `.gitignore`d.

## Score table columns

`data/scores/bouquet_scores.csv` and `wmt24pp_scores.csv` are keyed by
`(split, lang, model, condition, judge)`; the `*_domain_scores.csv` variants add `domain`.

| Column | Meaning |
|---|---|
| `n` | items scored in this condition |
| `n_critical_failure`, `critical_failure_rate` | outputs that are not a translation at all (wrong language, or no translation) |
| `translation_rating` | 0-100 ESA meaning-preservation score, averaged over **all** items, critical failures contributing 0 |
| `adaptedness` | 0-100 match to the instruction's purpose, same averaging |
| `*_excl_critical` | the same means over non-failed items only |

## Human evaluation columns

The human ratings are released in aggregate, as annotators consented to aggregated,
non-identifying use of their judgements. `python -m analysis.aggregate_human` rebuilds
these three from the annotation exports:

| File | Keyed by | Contents |
|---|---|---|
| `human_scores.csv` | `(lang, model, condition)` | mean translation rating and adaptedness, with and without critical failures |
| `human_agreement.csv` | `(lang, score_field, comparator)` | human against LLM judge, ChrF++ and XCOMET-XL: Pearson *r*, Kendall tau-b, Krippendorff alpha, ICC(3,k) |
| `human_divergence.csv` | `(lang, domain)` | rating against adaptedness: Pearson *r*, mean absolute difference, share of items differing by more than 20 points |

Together these carry every human-evaluation number the paper reports. `human_agreement.csv`
pairs scores on the exported text the annotator saw, whereas the paper's table joins on
`uniq_id` against item-level metric files that are not released, so its ChrF++ and
XCOMET-XL rows differ slightly from the published ones.

`translation_rating` and `adaptedness` use the paper's convention and are what its tables
report. The `condition` column encodes the experimental arm: `with_instructions`,
`without_instructions`, `without_instructions_paragraph`, `without_instructions_semantic`
(few-shot), `with_instructions_self-para-<model>` (§5), `with_instructions_{context,purpose}`
(Appendix G), and `*_{sourceonly,instructiononly}` (Appendix J).
