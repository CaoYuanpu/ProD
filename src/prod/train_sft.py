"""SFT distillation with LoRA on teacher trajectories (the base student for ProD).

    python -m prod.train_sft --task math --model Qwen/Qwen3-1.7B --output_dir outputs/math/ckpt/sft_Qwen3-1.7B

A checkpoint is saved after every epoch (checkpoint-<step>); the paper uses epoch 3 of a 5-epoch run.
"""

import argparse

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .chat import clean_messages, collator_tokenizer, response_instruction_templates, short_system_prompt, tokenize
from .collators import DataCollatorForCompletionOnlyLMMultiTurn
from .common import data_path, load_json, set_seed


def build_dataset(path, model_id, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    system_prompt = short_system_prompt()
    rows = []
    for messages in load_json(path):
        input_ids, attention_mask = tokenize(clean_messages(messages, system_prompt), tokenizer, max_length)
        rows.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return Dataset.from_list(rows)


def main(args):
    set_seed(args.seed)
    data_file = args.data or data_path(args.output_root, args.task, "sft_gpt-4.1")
    train_dataset = build_dataset(data_file, args.model, args.max_length)
    print(f"[sft] {len(train_dataset)} trajectories from {data_file}")

    response_template, instruction_template = response_instruction_templates(args.model)
    collator = DataCollatorForCompletionOnlyLMMultiTurn(
        response_template, instruction_template=instruction_template, tokenizer=collator_tokenizer(args.model)
    )
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, target_modules="all-linear",
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    train_args = SFTConfig(
        output_dir=args.output_dir,
        model_init_kwargs={"torch_dtype": "auto", "attn_implementation": args.attn_implementation},
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="no",
        label_names=["labels"],
        dataset_kwargs={"skip_prepare_dataset": True},  # already tokenized by build_dataset
        report_to="wandb" if args.wandb else "none",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        args.model, args=train_args, train_dataset=train_dataset, data_collator=collator, peft_config=peft_config
    )
    trainer.train()
    trainer.save_model(args.output_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["math", "hotpotqa"])
    p.add_argument("--model", required=True, help="base student, e.g. Qwen/Qwen3-1.7B")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data", default=None, help="default: {output_root}/{task}/data/sft_gpt-4.1.json")
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--max_length", type=int, default=32768)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--max_steps", type=int, default=-1, help="stop early (for quick checks)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
