"""Pair student rollouts with teacher trajectories for ProD (paper Sec. 3, Algorithm 1 lines 3-5).

For every training question the teacher solved, the student's own trajectory is paired with the
teacher's, together with both trajectories' average stagnation (Eq. 2) and the student's correctness.
The adaptive margin (Eq. 3) is computed from these fields at training time.

Requires, for the student tag, train-split trajectories (prod.rollout), judgments (prod.judge) and
stagnation scores (prod.stagnation), and the same for the teacher tag.

    python -m prod.build_prod_data --task math --student_tag sft_Qwen3-1.7B

Output: {output_root}/{task}/data/prod_{student_tag}.json, a shuffled list of
    {"id", "teacher": messages, "student": messages, "student_correct": 0/1,
     "teacher_stagnation": float, "student_stagnation": float}
"""

import argparse
import os
import random

from .common import data_path, dump_json, eval_path, load_json, read_examples, stagnation_path, traj_path
from .stagnation import trajectory_stagnation
from .tasks import train_dataset_name


def main(args):
    dataset = train_dataset_name(args.task)
    root, task = args.output_root, args.task
    teacher_stag = load_json(stagnation_path(root, task, args.teacher_tag, "train", dataset, args.metric))
    student_stag = load_json(stagnation_path(root, task, args.student_tag, "train", dataset, args.metric))

    pairs = []
    for exp in read_examples(task, "train", dataset):
        id_ = exp["id"]
        s_traj = traj_path(root, task, args.student_tag, "train", dataset, id_)
        s_eval = eval_path(root, task, args.student_tag, "train", dataset, id_)
        if not (os.path.exists(s_traj) and os.path.exists(s_eval)):
            continue
        t_eval = eval_path(root, task, args.teacher_tag, "train", dataset, id_)
        if load_json(t_eval)["score"] != 1:  # only questions the teacher solved
            continue
        teacher = load_json(traj_path(root, task, args.teacher_tag, "train", dataset, id_))["messages"]
        student = load_json(s_traj)["messages"]
        if teacher[1]["content"][0]["text"] != student[1]["content"][0]["text"]:
            raise ValueError(f"Question mismatch between teacher and student for id {id_}")
        pairs.append({
            "id": id_,
            "teacher": teacher,
            "student": student,
            "student_correct": load_json(s_eval)["score"],
            "teacher_stagnation": trajectory_stagnation(teacher_stag[str(id_)]),
            "student_stagnation": trajectory_stagnation(student_stag[str(id_)]),
        })

    random.seed(args.seed)
    random.shuffle(pairs)
    gaps = [p["student_stagnation"] - p["teacher_stagnation"] for p in pairs]
    print(f"[prod-data] {len(pairs)} pairs | student correct {sum(p['student_correct'] for p in pairs)} | "
          f"stagnation gap mean {sum(gaps) / len(gaps):.4f}, min {min(gaps):.4f}, max {max(gaps):.4f}")
    out = args.output or data_path(root, task, f"prod_{args.student_tag}")
    dump_json(pairs, out)
    print(f"[prod-data] -> {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["math", "hotpotqa"])
    p.add_argument("--student_tag", required=True)
    p.add_argument("--teacher_tag", default="gpt-4.1")
    p.add_argument("--metric", default="lexical", choices=["lexical", "semantic"])
    p.add_argument("--output", default=None)
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
