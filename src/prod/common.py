"""Shared helpers: seeding, dataset I/O, output layout and trajectory serialization."""

import json
import os
import random

import numpy as np

from .tasks import DATA_DIR, get_task


def set_seed(seed=42):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_examples(task, split, dataset):
    try:
        rel_path = get_task(task)["datasets"][split][dataset]
    except KeyError:
        raise ValueError(f"Dataset not found: {task} / {split} / {dataset}")
    with open(DATA_DIR / rel_path) as f:
        return json.load(f)["examples"]


# ---------------------------------------------------------------------------
# Output layout: {output_root}/{task}/{kind}/{tag}/{split}/{dataset}/...
#   kind = trajectories | evaluations | stagnation
# ---------------------------------------------------------------------------


def run_dir(output_root, task, kind, tag, split, dataset):
    return os.path.join(output_root, task, kind, tag, split, dataset)


def traj_path(output_root, task, tag, split, dataset, id_):
    return os.path.join(run_dir(output_root, task, "trajectories", tag, split, dataset), f"traj_{id_}.json")


def eval_path(output_root, task, tag, split, dataset, id_):
    return os.path.join(run_dir(output_root, task, "evaluations", tag, split, dataset), f"traj_{id_}.json")


def stagnation_path(output_root, task, tag, split, dataset, metric="lexical"):
    return os.path.join(run_dir(output_root, task, "stagnation", tag, split, dataset), f"{metric}.json")


def data_path(output_root, task, name):
    return os.path.join(output_root, task, "data", f"{name}.json")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Trajectory serialization (smolagents memory -> JSON)
# ---------------------------------------------------------------------------


def serialize_messages(messages):
    serialized = []
    for msg in messages:
        msg = msg.dict()
        if "role" in msg:
            msg["role"] = msg["role"].value
        serialized.append(msg)
    return serialized


def serialize_generated_answer(answer):
    import sympy

    if isinstance(answer, str):
        return answer
    if isinstance(answer, (list, tuple)):
        return [serialize_generated_answer(a) for a in answer]
    if isinstance(answer, np.ndarray):
        return answer.tolist()
    if isinstance(answer, (np.integer, np.floating)):
        return answer.item()
    if isinstance(answer, sympy.Basic):
        return str(answer)
    return str(answer)


def get_total_steps(messages):
    return sum(1 for msg in messages if msg["role"] == "assistant")


def assistant_turns(messages):
    """Text of every assistant turn (thought + code action) in a serialized trajectory."""
    return [msg["content"][0]["text"] for msg in messages if msg["role"] == "assistant"]
