"""Train Gemma-3-4B with SFT + LoRA on teacher translations (knowledge distillation).

Usage:
    python -m distill.train_sft \
        --train-path datasets/smol/ukr_Cyrl_candidates/sft_teacher_train_gemma-3-4b-it.jsonl \
        --eval-path datasets/smol/ukr_Cyrl_candidates/sft_teacher_eval_gemma-3-4b-it.jsonl \
        --output-dir models/gemma-3-4b-it-sft-smol-teacher-ukr_Cyrl
"""
from dotenv import load_dotenv
load_dotenv()

import os
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import SFTConfig, SFTTrainer

os.environ["WANDB_PROJECT"] = "gemma-3-4b-sft"


class TextOnlyTokenTypeIdsCollator:
    """Wrap an inner collator and add token_type_ids=0 for text-only Gemma3 training."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, features):
        batch = self.inner(features)
        if "token_type_ids" not in batch and "input_ids" in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SFT training for Gemma-3-4B using prompt-completion pairs."
    )
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sft_dataset(train_path: Path, eval_path: Path):
    if not train_path.exists():
        raise FileNotFoundError(f"Train JSONL not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval JSONL not found: {eval_path}")
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "eval": str(eval_path)},
    )
    expected = {"messages"}
    for split in ["train", "eval"]:
        cols = set(dataset[split].column_names)
        missing = expected - cols
        if missing:
            raise ValueError(f"Missing required columns in {split}: {sorted(missing)}")
    return dataset


def main() -> None:
    args = parse_args()

    dataset = load_sft_dataset(args.train_path, args.eval_path)
    print(f"Loaded dataset: train={len(dataset['train'])}, eval={len(dataset['eval'])}")
    sample = dataset["train"][0]
    print(f"Messages: {str(sample['messages'])[:300]}...")

    if args.dry_run:
        print("Dry run enabled, skipping model/trainer initialization.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Patch Gemma-3 chat template to mark assistant content for assistant_only_loss.
    # The template has no {% generation %} tags by default; wrap the trim on
    # assistant string content so TRL can mask non-assistant tokens out of the loss.
    if tokenizer.chat_template and "{% generation %}" not in tokenizer.chat_template:
        needle_trim = "{{ message['content'] | trim }}"
        wrapped_trim = (
            "{% if message['role'] == 'assistant' %}{% generation %}"
            "{{ message['content'] | trim }}"
            "{% endgeneration %}{% else %}"
            "{{ message['content'] | trim }}"
            "{% endif %}"
        )
        if needle_trim in tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.replace(
                needle_trim, wrapped_trim, 1
            )
            print("Patched chat template with {% generation %} markers.")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_length=args.max_length,
        assistant_only_loss=True,
        remove_unused_columns=False,
        bf16=True,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_config,
    )
    trainer.data_collator = TextOnlyTokenTypeIdsCollator(trainer.data_collator)

    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"Saved SFT model to {args.output_dir}")


if __name__ == "__main__":
    main()
