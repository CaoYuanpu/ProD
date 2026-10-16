"""Build the SFT distillation set: the teacher's successful training trajectories (paper Sec. 2.1).

    python -m prod.build_sft_data --task math --teacher_tag gpt-4.1

Output: {output_root}/{task}/data/sft_{teacher_tag}.json, a shuffled list of message lists.
"""

import argparse
import os
import random

from .common import data_path, dump_json, eval_path, load_json, read_examples, traj_path
from .tasks import train_dataset_name


def main(args):
    dataset = train_dataset_name(args.task)
    examples = read_examples(args.task, "train", dataset)
    kept = []
    for exp in examples:
        ev = eval_path(args.output_root, args.task, args.teacher_tag, "train", dataset, exp["id"])
        if os.path.exists(ev) and load_json(ev)["score"] == 1:
            kept.append(load_json(traj_path(args.output_root, args.task, args.teacher_tag, "train", dataset, exp["id"]))["messages"])
    print(f"[sft-data] teacher accuracy on train: {len(kept)}/{len(examples)} = {len(kept) / len(examples):.4f}")

    random.seed(args.seed)
    random.shuffle(kept)
    out = args.output or data_path(args.output_root, args.task, f"sft_{args.teacher_tag}")
    dump_json(kept, out)
    print(f"[sft-data] {len(kept)} trajectories -> {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["math", "hotpotqa"])
    p.add_argument("--teacher_tag", default="gpt-4.1")
    p.add_argument("--output", default=None)
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
