"""Data models and loaders for MT-GenEval (Currey et al., 2022).

Two subsets:

``sentences`` (counterfactual)
    Parallel masculine/feminine variants of the same source sentence, each with a matching
    reference. The gender cue is inside the sentence to translate.

``context`` (contextual)
    The gender cue appears *only* in the preceding context sentence; the sentence to translate
    is gender-ambiguous in English. Sources are stored as ``context <sep> sentence``. Two
    references are given: ``original`` (agreeing with the context) and ``flipped``.

Because the English context sentence carries the cue explicitly (he/she/his/her), the gold
gender can be recovered from the *source side*. Instructions built from it are therefore
extraction from the input, not leakage from the target reference.
"""
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

DATA_ROOT = Path("datasets/mt_geneval")

LANGS = ["ar", "de", "es", "fr", "hi", "it", "pt", "ru"]

LANG_NAMES = {
    "ar": "Arabic", "de": "German", "es": "Spanish", "fr": "French",
    "hi": "Hindi", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
}

WMT24PP_LOCALE = {
    "ar": "en-ar_EG", "de": "en-de_DE", "es": "en-es_MX", "fr": "en-fr_FR",
    "hi": "en-hi_IN", "it": "en-it_IT", "pt": "en-pt_PT", "ru": "en-ru_RU",
}

_MASC_CUES = re.compile(r"\b(he|him|his|himself)\b", re.I)
_FEM_CUES = re.compile(r"\b(she|her|hers|herself)\b", re.I)


def _read(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


class GenevalSentenceRow(BaseModel):
    """One counterfactual item: the same sentence in masculine and feminine variants."""
    lang: str
    idx: int
    source_masculine: str
    source_feminine: str
    ref_masculine: str
    ref_feminine: str
    pred_masculine: Optional[str] = None
    pred_feminine: Optional[str] = None


class GenevalContextRow(BaseModel):
    """One contextual item: gender is disambiguated only by ``context_en``."""
    lang: str
    idx: int
    context_en: str
    source: str
    ref_original: str   # agrees with the gender expressed in context_en
    ref_flipped: str
    gender: Optional[str] = None          # "masculine" | "feminine", read off context_en
    self_instruction: Optional[str] = None  # X3: model-drafted brief
    pred_text: Optional[str] = None


def infer_gender_from_context(context_en: str) -> Optional[str]:
    """Read the referent's gender off the English context sentence.

    Returns None when the context carries cues for both genders or neither, in which case the
    item cannot support an unambiguous source-side instruction.
    """
    masc = bool(_MASC_CUES.search(context_en))
    fem = bool(_FEM_CUES.search(context_en))
    if masc and not fem:
        return "masculine"
    if fem and not masc:
        return "feminine"
    return None


def load_sentences(lang: str, split: str = "test") -> List[GenevalSentenceRow]:
    base = DATA_ROOT / "sentences" / split
    sm = _read(base / f"geneval-sentences-masculine-{split}.en_{lang}.en")
    sf = _read(base / f"geneval-sentences-feminine-{split}.en_{lang}.en")
    rm = _read(base / f"geneval-sentences-masculine-{split}.en_{lang}.{lang}")
    rf = _read(base / f"geneval-sentences-feminine-{split}.en_{lang}.{lang}")
    if not len(sm) == len(sf) == len(rm) == len(rf):
        raise ValueError(f"Line count mismatch in MT-GenEval sentences for en-{lang}")
    return [
        GenevalSentenceRow(lang=lang, idx=i, source_masculine=a, source_feminine=b,
                           ref_masculine=c, ref_feminine=d)
        for i, (a, b, c, d) in enumerate(zip(sm, sf, rm, rf))
    ]


def load_context(lang: str, split: str = "test") -> List[GenevalContextRow]:
    base = DATA_ROOT / "context"
    src = _read(base / f"geneval-context-wikiprofessions-2to1-{split}.en_{lang}.en")
    orig = _read(base / f"geneval-context-wikiprofessions-original-{split}.en_{lang}.{lang}")
    flip = _read(base / f"geneval-context-wikiprofessions-flipped-{split}.en_{lang}.{lang}")
    n = min(len(src), len(orig), len(flip))
    rows = []
    for i in range(n):
        if "<sep>" not in src[i]:
            continue
        context_en, sentence = (p.strip() for p in src[i].split("<sep>", 1))
        rows.append(GenevalContextRow(
            lang=lang, idx=i, context_en=context_en, source=sentence,
            ref_original=orig[i], ref_flipped=flip[i],
            gender=infer_gender_from_context(context_en),
        ))
    return rows


def get_savepath(model: str, lang: str, subset: str, condition: str) -> Path:
    model_name = model.split("/")[-1]
    return Path(f"translation_results/controlled_mt/geneval_{subset}_{lang}_{model_name}_{condition}.jsonl")


def save(rows: List[BaseModel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def load_sentence_predictions(path: Path) -> List[GenevalSentenceRow]:
    with open(path, "r", encoding="utf-8") as f:
        return [GenevalSentenceRow.model_validate_json(line) for line in f]


def load_context_predictions(path: Path) -> List[GenevalContextRow]:
    with open(path, "r", encoding="utf-8") as f:
        return [GenevalContextRow.model_validate_json(line) for line in f]
