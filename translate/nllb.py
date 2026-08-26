"""One-off: evaluate facebook/nllb-200-3.3B on BOUQuET with ChrF.

Directions are derived from a list of Helsinki opus-mt_tiny model names (used
only to supply language pairs).

Usage:
    PYTHONPATH=. .venv/bin/python eval_nllb_bouquet.py
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from pathlib import Path

import sacrebleu
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common.bouquet_utils import BouquetRow

NLLB_MODEL = "facebook/nllb-200-3.3B"

OPUS_DIRECTIONS = [
    "eng-fra", "eng-ind", "eng-ukr", "eng-khm", "eng-jav",
]

# Map 3-letter opus code -> BOUQuET dataset code.
# NLLB tokenizer language tokens differ for some languages (see NLLB_LANG).
OPUS_TO_BOUQUET = {
    "eng": "eng_Latn",
    "fra": "fra_Latn",
    "ind": "ind_Latn",
    "ukr": "ukr_Cyrl",
    "khm": "khm_Khmr",
    "jav": "jav_Latn",
}

# NLLB-200 tokenizer language token (when different from BOUQuET).
NLLB_LANG = {
    "kor_Kore": "kor_Hang",
    "arz_Arab": "arb_Arab",  # NLLB has no Egyptian Arabic; closest is MSA
    "cmn_Hans": "zho_Hans",
}

SPLIT = "dev"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 512
OUT_PATH_TMPL = "translation_results/nllb_3_3b_bouquet_chrf_{split}.jsonl"
TRANSLATIONS_DIR_TMPL = "translation_results/nllb_3_3b_bouquet_{split}"


def load_lang_texts(flores_lang: str, split: str) -> dict[str, str]:
    """Return {uniq_id: text_in_flores_lang} for BOUQuET split.

    BOUQuET rows have src_text in src_lang and tgt_text in English, so for
    English we load any non-English lang and take tgt_text.
    """
    if flores_lang == "eng_Latn":
        rows = BouquetRow.load("fra_Latn", split)
        return {r.uniq_id: r.tgt_text for r in rows}
    rows = BouquetRow.load(flores_lang, split)
    return {r.uniq_id: r.src_text for r in rows}


def build_pairs(src_flores: str, tgt_flores: str, split: str):
    src = load_lang_texts(src_flores, split)
    tgt = load_lang_texts(tgt_flores, split)
    ids = sorted(set(src) & set(tgt))
    return [(uid, src[uid], tgt[uid]) for uid in ids]


def translate_batch(model, tokenizer, texts, src_flores, tgt_flores, device):
    # The installed tokenizer does not auto-prepend the NLLB src_lang token
    # nor append </s>, so build inputs manually.
    src_id = tokenizer.convert_tokens_to_ids(NLLB_LANG.get(src_flores, src_flores))
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    unk = tokenizer.unk_token_id

    encoded = [tokenizer(t, add_special_tokens=False).input_ids for t in texts]
    encoded = [[i for i in ids if i != unk][:510] for ids in encoded]
    ids = [[src_id] + e + [eos] for e in encoded]
    maxlen = max(len(x) for x in ids)
    input_ids = torch.tensor([x + [pad] * (maxlen - len(x)) for x in ids]).to(device)
    attn = (input_ids != pad).long()

    forced_bos = tokenizer.convert_tokens_to_ids(NLLB_LANG.get(tgt_flores, tgt_flores))
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            forced_bos_token_id=forced_bos,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def eval_direction(model, tokenizer, direction: str, split: str, trans_dir: Path, device) -> dict:
    src_o, tgt_o = direction.split("-")
    src_flores = OPUS_TO_BOUQUET[src_o]
    tgt_flores = OPUS_TO_BOUQUET[tgt_o]

    pairs = build_pairs(src_flores, tgt_flores, split)
    print(f"  {len(pairs)} parallel segments for {src_flores} -> {tgt_flores}")

    trans_path = trans_dir / f"{src_flores}-{tgt_flores}_{split}.jsonl"
    trans_path.parent.mkdir(parents=True, exist_ok=True)

    hyps, refs = [], []
    with trans_path.open("w") as tf:
        for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc=f"{src_flores}->{tgt_flores} ({split})"):
            batch = pairs[i:i + BATCH_SIZE]
            srcs = [p[1] for p in batch]
            rs = [p[2] for p in batch]
            ts = translate_batch(model, tokenizer, srcs, src_flores, tgt_flores, device)
            for (uid, s, r), h in zip(batch, ts):
                tf.write(json.dumps({
                    "uniq_id": uid,
                    "src_lang": src_flores,
                    "tgt_lang": tgt_flores,
                    "src_text": s,
                    "ref_text": r,
                    "hyp_text": h,
                }, ensure_ascii=False) + "\n")
            hyps.extend(ts)
            refs.extend(rs)

    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return {
        "model": NLLB_MODEL,
        "direction": direction,
        "src_lang": src_flores,
        "tgt_lang": tgt_flores,
        "split": split,
        "n": len(pairs),
        "chrf": chrf,
        "translations_path": str(trans_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directions", nargs="*", default=None,
                        help="Subset of directions like fra-eng; default: all")
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()
    directions = args.directions if args.directions else OPUS_DIRECTIONS
    split = args.split
    out_path = Path(OUT_PATH_TMPL.format(split=split))
    trans_dir = Path(TRANSLATIONS_DIR_TMPL.format(split=split))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                r = json.loads(line)
                if "direction" in r and "chrf" in r:
                    done.add(r["direction"])

    print(f"Loading {NLLB_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL, torch_dtype=torch.float16
    ).to(device).eval()

    with out_path.open("a") as f:
        for d in directions:
            if d in done:
                print(f"skip {d} (already done)")
                continue
            print(f"=== {d} ===")
            try:
                result = eval_direction(model, tokenizer, d, split, trans_dir, device)
            except Exception as e:
                print(f"  FAILED: {e}")
                result = {"model": NLLB_MODEL, "direction": d, "error": str(e)}
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(f"  -> chrF = {result.get('chrf')}")


if __name__ == "__main__":
    main()
