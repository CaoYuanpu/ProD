"""Report accuracy and efficiency (avg turns / tokens) of a judged run on the paper's evaluation sets.

    python -m prod.evaluate --task math --tag prod_Qwen3-1.7B

Accuracy is the number of correct answers over the full set size (missing trajectories count as wrong).
For MATH the in-domain set pools the val and test splits (770 problems).
"""

import argparse
import os

from .common import eval_path, load_json, read_examples, traj_path
from .tasks import get_task


def summarize(output_root, task, tag, parts):
    n, correct, steps, tokens = 0, 0, [], []
    for split, dataset in parts:
        for exp in read_examples(task, split, dataset):
            n += 1
            ev = eval_path(output_root, task, tag, split, dataset, exp["id"])
            if os.path.exists(ev):
                result = load_json(ev)
                correct += int(result["score"] == 1)
                steps.append(result["total_steps"])
            tr = traj_path(output_root, task, tag, split, dataset, exp["id"])
            if os.path.exists(tr):
                tokens.append(load_json(tr)["token_usage"]["total_tokens"])
    return {
        "n": n,
        "judged": len(steps),
        "accuracy": 100.0 * correct / n,
        "avg_turns": sum(steps) / len(steps) if steps else float("nan"),
        "avg_tokens": sum(tokens) / len(tokens) if tokens else float("nan"),
    }


def main(args):
    eval_sets = get_task(args.task)["eval_sets"]
    names = args.sets.split(",") if args.sets else list(eval_sets)
    print(f"{'set':<20}{'acc (%)':>9}{'judged/n':>12}{'turns':>8}{'tokens':>10}")
    for name in names:
        r = summarize(args.output_root, args.task, args.tag, eval_sets[name])
        print(f"{name:<20}{r['accuracy']:>9.1f}{r['judged']:>7}/{r['n']:<4}{r['avg_turns']:>8.3f}{r['avg_tokens']:>10.0f}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["math", "hotpotqa"])
    p.add_argument("--tag", required=True)
    p.add_argument("--sets", default=None, help="comma-separated subset of the task's eval sets")
    p.add_argument("--output_root", default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
