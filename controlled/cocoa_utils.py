"""Data models and loaders for the CoCoA-MT formality-control test set.

CoCoA-MT (Nadejde et al., 2022) provides, for each English source segment, a formal and an
informal reference translation, each annotated with phrase-level formality markers wrapped in
``[F]...[/F]``. Scoring is matched accuracy (M-Acc); see ``controlled/score_cocoa.py``.

The release uses three different filename schemes across language pairs, which this module hides.
Note that ``en-pt`` is *European* Portuguese (the files are named ``en-ptpt``).
"""
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

DATA_ROOT = Path("datasets/cocoa_mt/test")

LANG_PAIRS = ["en-de", "en-es", "en-fr", "en-hi", "en-it", "en-ja", "en-nl", "en-pt"]
DOMAINS = ["call_center", "telephony", "topical_chat"]

# Target languages whose tokens are not whitespace-delimited: phrase matching must not split
# on spaces (see the -nd flag of the official IWSLT2022 scorer).
NON_WHITESPACE_DELIMITED = {"en-ja"}

LANG_NAMES = {
    "en-de": "German", "en-es": "Spanish", "en-fr": "French", "en-hi": "Hindi",
    "en-it": "Italian", "en-ja": "Japanese", "en-nl": "Dutch", "en-pt": "Portuguese",
}

# WMT24++ locale used for the mechanism analysis in controlled/correlate_wmt24pp.py.
# CoCoA's Portuguese is European, so it pairs with pt_PT rather than pt_BR. Nadejde et al. (2022)
# fn. 7 states the French and Spanish references are the France and *Spain* variants, but WMT24++
# only ships es_MX, so en-es pairs a European-Spanish annotation with a Mexican-Spanish outcome.
# Unlike pt this is not fatal: usted/tu is shared across variants and en-es passes the
# manipulation check at 97.4 (see score_cocoa.manipulation_check).
WMT24PP_LOCALE = {
    "en-de": "en-de_DE", "en-es": "en-es_MX", "en-fr": "en-fr_FR", "en-hi": "en-hi_IN",
    "en-it": "en-it_IT", "en-ja": "en-ja_JP", "en-nl": "en-nl_NL", "en-pt": "en-pt_PT",
}

FORMALITY_PHRASE = re.compile(r"\[F\](.*?)\[/F\]")


def _stem(lang_pair: str, domain: str) -> Path:
    """Filename stem, which differs per language pair."""
    if lang_pair == "en-pt":
        return DATA_ROOT / lang_pair / f"formality-control.test.{domain}.en-ptpt"
    return DATA_ROOT / lang_pair / f"formality-control.test.{domain}.{lang_pair}"


def _paths(lang_pair: str, domain: str) -> tuple[Path, Path, Path]:
    """(source, formal annotated ref, informal annotated ref) for a language pair and domain."""
    stem = _stem(lang_pair, domain)
    if lang_pair in ("en-nl", "en-pt"):
        tgt = lang_pair.split("-")[1] if lang_pair == "en-nl" else "pt"
        return (
            stem.with_name(stem.name + ".en"),
            stem.with_name(f"{stem.name}.formal.annotated.{tgt}"),
            stem.with_name(f"{stem.name}.informal.annotated.{tgt}"),
        )
    return (
        stem,
        stem.with_name(stem.name + ".formal.annotated"),
        stem.with_name(stem.name + ".informal.annotated"),
    )


def strip_markers(text: str) -> str:
    """Remove [F]...[/F] annotation markers, leaving the plain reference translation."""
    return re.sub(r"\[/?F\]", "", text)


def _read(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


class CocoaRow(BaseModel):
    """One CoCoA-MT test segment with both annotated references."""
    lang_pair: str
    domain: str
    idx: int
    source: str                 # English
    formal_annotated: str       # target, with [F]...[/F] markers
    informal_annotated: str
    pred_text: Optional[str] = None

    @property
    def formal_ref(self) -> str:
        return strip_markers(self.formal_annotated)

    @property
    def informal_ref(self) -> str:
        return strip_markers(self.informal_annotated)

    @property
    def has_formality_marker(self) -> bool:
        """False for segments where neither reference carries a marker: M-Acc cannot decide them."""
        return bool(
            FORMALITY_PHRASE.search(self.formal_annotated)
            or FORMALITY_PHRASE.search(self.informal_annotated)
        )


def load(lang_pair: str, domains: Optional[List[str]] = None) -> List[CocoaRow]:
    """Load all test segments for a language pair."""
    if lang_pair not in LANG_PAIRS:
        raise ValueError(f"Unknown CoCoA-MT language pair: {lang_pair}")

    rows: List[CocoaRow] = []
    for domain in domains or DOMAINS:
        src_p, fml_p, inf_p = _paths(lang_pair, domain)
        src, fml, inf = _read(src_p), _read(fml_p), _read(inf_p)
        if not len(src) == len(fml) == len(inf):
            raise ValueError(
                f"Line count mismatch for {lang_pair}/{domain}: "
                f"src={len(src)} formal={len(fml)} informal={len(inf)}"
            )
        for i, (s, f, n) in enumerate(zip(src, fml, inf)):
            rows.append(CocoaRow(
                lang_pair=lang_pair, domain=domain, idx=i,
                source=s, formal_annotated=f, informal_annotated=n,
            ))
    return rows


def get_savepath(model: str, lang_pair: str, condition: str, variant: str = "") -> Path:
    """``variant`` tags a re-run under a different prompt, e.g. ``ptPT`` for a run that asks for
    European rather than unmarked Portuguese. Empty keeps the original filename."""
    model_name = model.split("/")[-1]
    suffix = f"__{variant}" if variant else ""
    return Path(
        f"translation_results/controlled_mt/cocoa_{lang_pair}_{model_name}_{condition}{suffix}.jsonl"
    )


def save(rows: List[CocoaRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def load_predictions(path: Path) -> List[CocoaRow]:
    with open(path, "r", encoding="utf-8") as f:
        return [CocoaRow.model_validate_json(line) for line in f]
