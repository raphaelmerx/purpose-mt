"""Translate the controlled-MT benchmarks (CoCoA-MT, MT-GenEval) with vLLM.

Unlike translate/bouquet_vllm.py, this loads the model once and sweeps every
(dataset, language, condition) cell in a single process: the grid is ~87k short generations, so
per-invocation model loading would dominate wall-clock.

Usage:
    python -m controlled.translate --model google/gemma-3-27b-it
    python -m controlled.translate --datasets cocoa --langs en-fr --limit 20
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import re
from typing import List, Optional

from controlled import cocoa_utils as C
from controlled import geneval_utils as G
from controlled.prompts import (
    COCOA_CONDITIONS,
    GENEVAL_CONTEXT_CONDITIONS,
    GENEVAL_SENTENCE_CONDITIONS,
    build_messages,
    cocoa_instruction,
    geneval_brief,
    geneval_context_messages,
    self_instruct_messages,
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_args():
    p = argparse.ArgumentParser(description="Translate controlled-MT benchmarks with vLLM")
    p.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    p.add_argument("--datasets", type=str, default="cocoa,geneval-counterfactual,geneval-contextual",
                   help="comma-separated subset of cocoa,geneval-counterfactual,geneval-contextual")
    p.add_argument("--langs", type=str, default=None,
                   help="comma-separated language filter (CoCoA: en-fr; GenEval: fr)")
    p.add_argument("--conditions", type=str, default=None, help="comma-separated condition filter")
    p.add_argument("--limit", type=int, default=None, help="rows per cell, for smoke tests")
    p.add_argument("--ignore-cache", action="store_true", help="re-translate cells that already exist")
    p.add_argument("--lang-name", type=str, default=None,
                   help="CoCoA only: override the language name in the prompt, e.g. "
                        "'Portuguese (from Portugal)'. Requires a single --langs value.")
    p.add_argument("--fewshot-key", type=str, default=None,
                   help="CoCoA only: override the FEWSHOT_EXAMPLES_BY_LANG key used by C0_3shot, "
                        "e.g. 'pt_PT'. Requires a single --langs value.")
    p.add_argument("--variant", type=str, default="",
                   help="CoCoA only: tag output files so an override run does not clobber the "
                        "original, e.g. 'ptPT'.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    return p.parse_args()


def _clean(text: str) -> str:
    text = _THINK.sub("", text.strip()).strip()
    return text.replace("[START_TRANSLATE]", "").replace("[END_TRANSLATE]", "").strip()


class Generator:
    """Thin wrapper so the model is loaded exactly once."""

    def __init__(self, model: str, max_tokens: int, max_model_len: int):
        from vllm import LLM, SamplingParams

        print(f"Loading {model}...")
        self.llm = LLM(model, max_model_len=max_model_len)
        self.params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        self.chat_kwargs = {}
        if "qwen" in model.lower():
            self.chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    def __call__(self, messages: List[List[dict]]) -> List[str]:
        if not messages:
            return []
        outputs = self.llm.chat(messages, sampling_params=self.params, **self.chat_kwargs)
        return [_clean(o.outputs[0].text) for o in outputs]


def run_cocoa(gen: Generator, args, lang_filter: Optional[set], cond_filter: Optional[set]) -> None:
    for lang_pair in C.LANG_PAIRS:
        if lang_filter and lang_pair not in lang_filter:
            continue
        lang_code = args.fewshot_key or lang_pair.split("-")[1]
        lang_name = args.lang_name or C.LANG_NAMES[lang_pair]
        rows = C.load(lang_pair)
        if args.limit:
            rows = rows[: args.limit]

        for cond in COCOA_CONDITIONS:
            if cond_filter and cond.name not in cond_filter:
                continue
            path = C.get_savepath(args.model, lang_pair, cond.name, args.variant)
            if path.exists() and not args.ignore_cache:
                print(f"skip (cached): {path}")
                continue

            messages = [
                build_messages(r.source, lang_name, lang_code,
                               cocoa_instruction(cond, r.domain), cond.few_shot)
                for r in rows
            ]
            print(f"CoCoA {lang_pair} {cond.name}: {len(messages)} segments")
            preds = gen(messages)
            for r, pred in zip(rows, preds):
                r.pred_text = pred
            C.save(rows, path)
            print(f"  -> {path}")


def run_geneval_counterfactual(gen: Generator, args, lang_filter, cond_filter) -> None:
    for lang in G.LANGS:
        if lang_filter and lang not in lang_filter:
            continue
        lang_name = G.LANG_NAMES[lang]
        rows = G.load_sentences(lang)
        if args.limit:
            rows = rows[: args.limit]

        for cond in GENEVAL_SENTENCE_CONDITIONS:
            if cond_filter and cond not in cond_filter:
                continue
            path = G.get_savepath(args.model, lang, "counterfactual", cond)
            if path.exists() and not args.ignore_cache:
                print(f"skip (cached): {path}")
                continue

            masc_instr = geneval_brief("masculine") if cond == "G1_brief" else None
            fem_instr = geneval_brief("feminine") if cond == "G1_brief" else None
            msgs = (
                [build_messages(r.source_masculine, lang_name, lang, masc_instr) for r in rows]
                + [build_messages(r.source_feminine, lang_name, lang, fem_instr) for r in rows]
            )
            print(f"GenEval/counterfactual {lang} {cond}: {len(msgs)} segments")
            preds = gen(msgs)
            half = len(rows)
            for r, pm, pf in zip(rows, preds[:half], preds[half:]):
                r.pred_masculine, r.pred_feminine = pm, pf
            G.save(rows, path)
            print(f"  -> {path}")


def run_geneval_contextual(gen: Generator, args, lang_filter, cond_filter) -> None:
    for lang in G.LANGS:
        if lang_filter and lang not in lang_filter:
            continue
        lang_name = G.LANG_NAMES[lang]
        # Restrict to items whose gender is unambiguously recoverable from the English context,
        # so every condition is scored on the same subset (~90% of items).
        rows = [r for r in G.load_context(lang) if r.gender]
        if args.limit:
            rows = rows[: args.limit]

        for cond in GENEVAL_CONTEXT_CONDITIONS:
            if cond_filter and cond not in cond_filter:
                continue
            path = G.get_savepath(args.model, lang, "contextual", cond)
            if path.exists() and not args.ignore_cache:
                print(f"skip (cached): {path}")
                continue

            print(f"GenEval/contextual {lang} {cond}: {len(rows)} segments")
            if cond == "X3_self_instruct":
                drafts = gen([self_instruct_messages(r.context_en, r.source) for r in rows])
                for r, d in zip(rows, drafts):
                    r.self_instruction = d
                msgs = [
                    geneval_context_messages(r.context_en, r.source, lang_name, cond, r.self_instruction)
                    for r in rows
                ]
            elif cond == "X2_brief":
                msgs = [
                    geneval_context_messages(r.context_en, r.source, lang_name, cond, geneval_brief(r.gender))
                    for r in rows
                ]
            else:
                msgs = [geneval_context_messages(r.context_en, r.source, lang_name, cond) for r in rows]

            preds = gen(msgs)
            for r, pred in zip(rows, preds):
                r.pred_text = pred
            G.save(rows, path)
            print(f"  -> {path}")


def main() -> None:
    args = parse_args()
    datasets = {d.strip() for d in args.datasets.split(",")}
    lang_filter = {x.strip() for x in args.langs.split(",")} if args.langs else None
    cond_filter = {x.strip() for x in args.conditions.split(",")} if args.conditions else None

    if args.lang_name or args.fewshot_key:
        # A prompt override changes what is being measured, so confine it to one language and
        # force a distinct output tag; otherwise it would silently overwrite the main grid.
        if datasets != {"cocoa"}:
            raise SystemExit("--lang-name/--fewshot-key apply to CoCoA only; pass --datasets cocoa")
        if not lang_filter or len(lang_filter) != 1:
            raise SystemExit("--lang-name/--fewshot-key require exactly one --langs value")
        if not args.variant:
            raise SystemExit("--lang-name/--fewshot-key require --variant to tag the output files")

    gen = Generator(args.model, args.max_tokens, args.max_model_len)

    if "cocoa" in datasets:
        run_cocoa(gen, args, lang_filter, cond_filter)
    if "geneval-counterfactual" in datasets:
        run_geneval_counterfactual(gen, args, lang_filter, cond_filter)
    if "geneval-contextual" in datasets:
        run_geneval_contextual(gen, args, lang_filter, cond_filter)

    print("\nDone. Score with:")
    print(f"  python -m controlled.score_cocoa --model {args.model}")
    print(f"  python -m controlled.score_geneval --model {args.model}")


if __name__ == "__main__":
    main()
