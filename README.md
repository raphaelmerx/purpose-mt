# Beyond "To whom it may concern": Tailoring Machine Translation to Audience and Intent

[![arXiv](https://img.shields.io/badge/arXiv-2606.03259-b31b1b.svg)](https://arxiv.org/abs/2606.03259)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/raphaelmerx/purpose-mt)

Translation quality depends on purpose: the same source text demands different
translations depending on audience, tone and communicative intent. This repository
evaluates whether LLMs can act on an explicit instruction describing that purpose, across
50+ languages, 5 model sizes and 8 text domains.

## Repository layout

Directories follow the paper's order:

```
common/          shared data models and loaders
instructions/    §3.2  generating user instructions from benchmark metadata
translate/       §3.3  translation under baseline / instruction / few-shot / paragraph
judge/           §3.4  LLM-as-judge: translation rating, adaptedness, critical failures
metrics/         §4.5  XCOMET-XL, ChrF++, and judge-metric divergence
self_instruct/   §5    self-generated instructions from document context
controlled/      App K controlled-MT benchmarks (CoCoA-MT, MT-GenEval)
distill/         App I PEFT distillation from a 27B teacher
human_eval/      §3.4  human annotation protocol and human-judge agreement
analysis/        tables and figures
data/            fetches the gated dataset; no data is stored in this repo
```

## Install

```bash
uv sync
cp .env.example .env    # then fill in GEMINI_API_KEY and HF_TOKEN
```

vLLM isn't in `pyproject.toml`, as its build is CUDA- and driver-specific. Install it
to match your GPU stack.

## Data

Data is stored in a separate gated HF dataset, because it derives from BOUQuET and WMT24++, both translation test sets.

```bash
python -m data.download              # everything
python -m data.download --only scores   # just the numbers behind the paper
```

See [`data/README.md`](data/README.md) for access instructions and layout.

## Paper → code

| Paper | Code |
|---|---|
| §3.2 Instruction generation, `fig:pipeline`(a) | `instructions/generate_bouquet.py`, `generate_wmt24pp.py` |
| §3.3 Translation, four conditions | `translate/bouquet_vllm.py`, `wmt24pp_vllm.py` |
| §3.4 LLM judge | `judge/bouquet.py`, `judge/wmt24pp_batch.py` |
| §3.4 Human validation, `tab:human-llm-agreement` | `human_eval/human_llm_correlation.py`, `human_metric_correlation.py` |
| §4.1 Model size, `fig:dumbbell` | `analysis/figures.py` |
| §4.2 Language resource level | `analysis/extract_main_results.py` |
| §4.3 Domain, `fig:domain`, `tab:wmt24pp-domain` | `analysis/figures.py` |
| §4 Win rates, `tab:winrate` | `judge/win_rate.py` |
| §4.4 Few-shot / paragraph, `tab:fewshot-12b` | `translate/bouquet_vllm.py --semantic-fewshot`, `analysis/extract_fewshot.py` |
| §4.5 Traditional metrics, `tab:comet-domain` | `metrics/comet_judge_divergence.py`, `comet.py`, `chrf.py` |
| §5 Self-instruction, `tab:self-instr-summary` | `self_instruct/bouquet.py`, `wmt24pp.py` |
| App A Full per-language results | `analysis/extract_full_tables.py` |
| App B NLLB-3.3B, `tab:nllb-deltas` | `translate/nllb.py`, `analysis/nllb_judge_scores.py`, `nllb_domain.py` |
| App C WMT24++, `fig:wmt24pp-dumbbell` | `judge/wmt24pp_batch.py`, `analysis/extract_wmt24pp_tables.py` |
| App D Qwen 3.5, `tab:qwen-main` | `translate/bouquet_vllm.py --model Qwen/Qwen3.5-27B` |
| App E Prompts | prompt constants in `translate/`, `judge/`, `controlled/prompts.py` |
| App F Paragraph-level, `tab:paragraph-comparison` | `translate/bouquet_vllm.py --paragraph-level` |
| App G Instruction ablation, `tab:ablation-gap` | `instructions/generate_variants.py` |
| App I PEFT with a 27B teacher, `tab:lora-sft` | `distill/*.py` |
| App J Judge-input ablation, `tab:judge-inputs` | `judge/bouquet.py --judge-inputs {source-only,instruction-only}` |
| App K Controlled MT, `tab:cocoa-control` | `controlled/*.py` |
| App L Annotation protocol | `human_eval/setup_project.py`, `human_eval/guidelines/` |

## Pipeline

Every script is run as a module from the repository root.

### 1. Generate instructions (§3.2)

Gemini drafts a language-agnostic instruction from each item's domain, context comment
and linguistic tags. The released instructions were then revised by a human annotator.

```bash
python -m instructions.generate_bouquet --split dev
python -m instructions.generate_wmt24pp
python -m instructions.generate_variants --split dev   # Appendix G: context + purpose
```

### 2. Translate (§3.3)

Four conditions per source sentence: baseline, instruction, semantic few-shot, paragraph.

```bash
python -m translate.bouquet_vllm --tgt-lang fra_Latn --model google/gemma-3-27b-it
python -m translate.bouquet_vllm --tgt-lang fra_Latn --model google/gemma-3-27b-it --use-user-instructions
python -m translate.bouquet_vllm --tgt-lang fra_Latn --model google/gemma-3-27b-it --semantic-fewshot
python -m translate.bouquet_vllm --tgt-lang fra_Latn --model google/gemma-3-27b-it --paragraph-level
```

### 3. Judge (§3.4)

The judge sees source, instruction and translation, and returns an ESA translation
rating, an adaptedness score and a critical-failure flag. It is always shown the
instruction and is blind to whether one was used at translation time.

```bash
python -m judge.bouquet --tgt-lang fra_Latn --print-per-domain
python -m judge.bouquet --tgt-lang fra_Latn --judge-inputs source-only   # Appendix J
python -m judge.wmt24pp_batch --lang en-fr_FR --submit                   # Gemini Batch API
python -m judge.wmt24pp_batch --lang en-fr_FR --parse
```

### 4. Metrics (§4.5)

```bash
python -m metrics.comet --tgt-lang fra_Latn --model google/gemma-3-27b-it
python -m metrics.chrf  --tgt-lang fra_Latn --model google/gemma-3-27b-it
python -m metrics.comet_judge_divergence
```

### 5. Self-instruction (§5)

The model drafts its own instruction from the surrounding paragraph, then translates
conditioned on it.

```bash
python -m self_instruct.bouquet --model google/gemma-3-27b-it --split dev
python -m translate.bouquet_vllm --tgt-lang fra_Latn --model google/gemma-3-27b-it \
    --use-user-instructions --instruction-type self-para-gemma-3-27b-it
```

### 6. Controlled-MT benchmarks (Appendix K)

```bash
python -m controlled.translate --datasets cocoa --model google/gemma-3-27b-it
python -m controlled.score_cocoa
python -m controlled.score_geneval
python -m controlled.report_tables
```

### 7. PEFT distillation (Appendix I)

```bash
python -m distill.load_smol
python -m distill.generate_instructions --lang ukr_Cyrl
python -m distill.translate_teacher --lang ukr_Cyrl --model google/gemma-3-27b-it --n-candidates 1
python -m distill.prepare_sft --lang ukr_Cyrl
python -m distill.train_sft --train-path ... --eval-path ...
```

### 8. Tables and figures

```bash
python -m analysis.aggregate     # item-level results -> data/scores/*.csv
python -m analysis.figures       # -> figures/
```

## Human evaluation (§3.4, Appendix L)

Annotations were collected in Label Studio from one native speaker per target language,
160 items each. `human_eval/guidelines/` holds the instructions shown to annotators.

```bash
python -m human_eval.setup_project --tgt-lang fra_Latn   # needs LABEL_STUDIO_*
python -m human_eval.download_annotations --project-id 9
python -m human_eval.human_llm_correlation             # tab:human-llm-agreement, judge columns
python -m human_eval.human_metric_correlation --matched \
    --human-annotations annotation_exports/human/project_9_annotations.jsonl   # COMET / ChrF++ columns
```

The judge is scored on the text the annotator rated (`human_eval.match_judge`).

## Licence and attribution

Code and our data contributions are released under **CC-BY-4.0**.

This work builds on
[BOUQuET](https://huggingface.co/datasets/facebook/bouquet) (CC-BY-4.0, gated),
[WMT24++](https://huggingface.co/datasets/google/wmt24pp) (Apache-2.0),
[SMOL](https://huggingface.co/datasets/google/smol) (CC-BY-4.0),
CoCoA-MT (CDLA-Sharing-1.0) and MT-GenEval (CC-BY-SA-3.0). Their respective terms
continue to apply to derived files.

## Citation

```bibtex
@inproceedings{merx2026beyond,
  title     = {Beyond ``To whom it may concern'': Tailoring Machine Translation to Audience and Intent},
  author    = {Merx, Raphael and Vylomova, Ekaterina and Cohn, Trevor},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```
