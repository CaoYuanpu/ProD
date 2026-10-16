"""ProD: continue training an SFT-distilled LoRA student on (teacher, student) trajectory pairs.

    python -m prod.train_prod --task math --model Qwen/Qwen3-1.7B \\
        --init_lora outputs/math/ckpt/sft_Qwen3-1.7B/checkpoint-261 \\
        --data outputs/math/data/prod_sft_Qwen3-1.7B.json \\
        --output_dir outputs/math/ckpt/prod_Qwen3-1.7B --lr 5e-6 --c0 0.5

For iterative ProD (Sec. 4.4), roll out the resulting student on the training set, rebuild the pair
data and run this script again with --init_lora pointing at the new checkpoint.
"""

import argparse

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from .chat import clean_messages, collator_tokenizer, response_instruction_templates, short_system_prompt, tokenize
from .collators import ProDPairCollator
from .common import load_json, set_seed
from .trainer import ProDTrainer


def build_dataset(path, model_id, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    system_prompt = short_system_prompt()
    rows = []
    for pair in load_json(path):
        t_ids, t_mask = tokenize(clean_messages(pair["teacher"], system_prompt), tokenizer, max_length)
        s_ids, s_mask = tokenize(clean_messages(pair["student"], system_prompt), tokenizer, max_length)
        rows.append({
            "teacher_input_ids": t_ids,
            "teacher_attention_mask": t_mask,
            "student_input_ids": s_ids,
            "student_attention_mask": s_mask,
            "stagnation_gap": max(0.0, pair["student_stagnation"] - pair["teacher_stagnation"]),
            "wrong_answer": 1.0 - pair["student_correct"],
        })
    return Dataset.from_list(rows)


def main(args):
    set_seed(args.seed)
    train_dataset = build_dataset(args.data, args.model, args.max_length)
    print(f"[prod] {len(train_dataset)} pairs from {args.data}; "
          f"{int(sum(train_dataset['wrong_answer']))} with a wrong student answer")

    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", attn_implementation=args.attn_implementation
    )
    model = PeftModel.from_pretrained(base, args.init_lora, is_trainable=True)
    if not args.no_gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    response_template, instruction_template = response_instruction_templates(args.model)
    collator = ProDPairCollator(
        response_template, instruction_template=instruction_template, tokenizer=collator_tokenizer(args.model)
    )
    train_args = SFTConfig(
        output_dir=args.output_dir,
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
    trainer = ProDTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=collator,
        beta=args.beta,
        c0=args.c0,
        alpha=args.alpha,
        lambda_sft=args.lambda_sft,
        fixed_margin=args.fixed_margin,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["math", "hotpotqa"])
    p.add_argument("--model", required=True, help="base model of the student, e.g. Qwen/Qwen3-1.7B")
    p.add_argument("--init_lora", required=True, help="LoRA of the SFT (or previous-iteration ProD) student")
    p.add_argument("--data", required=True, help="pair data from prod.build_prod_data")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=2.5, help="log-likelihood scale (SimPO beta)")
    p.add_argument("--c0", type=float, default=0.5, help="base margin constant")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="weight of the stagnation gap vs. answer correctness in the margin (paper Eq. 3)")
    p.add_argument("--lambda_sft", type=float, default=1.0, help="weight of the SFT loss on teacher trajectories")
    p.add_argument("--fixed_margin", action="store_true", help="ablation: constant margin c0 instead of Eq. 3")
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1, help="pairs per device")
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--max_length", type=int, default=10240)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--max_steps", type=int, default=-1, help="stop early (for quick checks)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
