"""Run a CodeAct agent (teacher, base model or distilled student) and save its trajectories.

Examples:
    # teacher (OpenAI API, needs OPENAI_API_KEY)
    python -m prod.rollout --task math --split train --backend litellm --model gpt-4.1-2025-04-14 --tag gpt-4.1
    # distilled student (vLLM + LoRA)
    python -m prod.rollout --task math --split val --model Qwen/Qwen3-1.7B --lora outputs/math/ckpt/sft_Qwen3-1.7B/checkpoint-261 --tag sft_Qwen3-1.7B
"""

import argparse
import os
import time

from tqdm import tqdm

from .common import (
    dump_json,
    get_total_steps,
    read_examples,
    serialize_generated_answer,
    serialize_messages,
    set_seed,
    traj_path,
)
from .tasks import MAX_STEPS, MAX_TOKENS, get_task, make_tools, train_dataset_name


def build_agent(args, task_cfg):
    from smolagents import CodeAgent, LiteLLMModel, VLLMModel

    if args.backend == "litellm":
        model = LiteLLMModel(model_id=args.model, api_key=os.environ["OPENAI_API_KEY"], max_tokens=MAX_TOKENS)
    else:
        model_kwargs = {}
        if args.gpu_memory_utilization is not None:
            model_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
        model = VLLMModel(model_id=args.model, lora_path=args.lora, model_kwargs=model_kwargs)

    # Distilled students are trained (and evaluated) with the compact system prompt.
    short_prompt = args.prompt == "short" or (args.prompt == "auto" and args.lora is not None)
    return CodeAgent(
        tools=make_tools(args.task, args.retriever_port),
        model=model,
        max_steps=args.max_steps,
        additional_authorized_imports=task_cfg["authorized_imports"],
        instruction=task_cfg["instruction"],
        return_full_result=True,
        short_system_prompt=short_prompt,
    )


def main(args):
    set_seed(args.seed)
    task_cfg = get_task(args.task)
    dataset = args.dataset or train_dataset_name(args.task)
    examples = read_examples(args.task, args.split, dataset)
    end = len(examples) - 1 if args.end is None else args.end
    examples = examples[args.start : end + 1]

    agent = build_agent(args, task_cfg)
    print(f"[rollout] task={args.task} split={args.split} dataset={dataset} model={args.model} "
          f"lora={args.lora} tag={args.tag} n={len(examples)}")

    start_time, n_done = time.time(), 0
    for exp in tqdm(examples):
        id_, question = exp["id"], exp["question"]
        out_file = traj_path(args.output_root, args.task, args.tag, args.split, dataset, id_)
        if os.path.exists(out_file):
            continue
        try:
            result = agent.run(question + task_cfg["instruction"])
            messages = serialize_messages(agent.write_memory_to_messages())
            traj = {
                "id": id_,
                "question": question,
                "messages": messages,
                "total_steps": get_total_steps(messages),
                "true_answer": exp["answer"],
                "generated_answer": serialize_generated_answer(result.output),
                "token_usage": {
                    "input_tokens": result.token_usage.input_tokens,
                    "output_tokens": result.token_usage.output_tokens,
                    "total_tokens": result.token_usage.total_tokens,
                },
                "duration": result.timing.duration,
                "state": result.state,
            }
            dump_json(traj, out_file)
            n_done += 1
        except Exception as e:  # keep going; a missing trajectory is simply skipped downstream
            print(f"[rollout] id={id_} failed: {e}")

    elapsed = time.time() - start_time
    print(f"[rollout] wrote {n_done}/{len(examples)} trajectories in {elapsed:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["math", "hotpotqa"])
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--dataset", default=None, help="dataset within the split (default: the task's main dataset)")
    p.add_argument("--backend", default="vllm", choices=["vllm", "litellm"])
    p.add_argument("--model", required=True, help="HF model id / local path (vllm) or API model id (litellm)")
    p.add_argument("--lora", default=None, help="LoRA adapter directory (vllm only)")
    p.add_argument("--tag", required=True, help="name under which trajectories are stored")
    p.add_argument("--prompt", default="auto", choices=["auto", "full", "short"],
                   help="system prompt; auto = short for LoRA students, full otherwise")
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="inclusive")
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    p.add_argument("--gpu_memory_utilization", type=float, default=None)
    p.add_argument("--retriever_port", default="8005")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
