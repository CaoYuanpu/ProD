"""Turn-level stagnation scores (paper Sec. 2.2, Table 1).

For the thought+action h_i (+) a_i of turn i, the score is its maximum similarity to any earlier turn:
    lexical : max_{j<i} ROUGE-L F1(h_i+a_i, h_j+a_j)
    semantic: max_{j<i} cos(E(h_i+a_i), E(h_j+a_j)),  E = all-mpnet-base-v2
The first turn has no history and scores 0. ProD's adaptive margin uses the lexical score.

    python -m prod.stagnation --task math --tag sft_Qwen3-1.7B --split train

Writes {id: [score_turn1, score_turn2, ...]} and prints the run's average stagnation.
"""

import argparse
import os

from tqdm import tqdm

from .common import assistant_turns, dump_json, load_json, read_examples, stagnation_path, traj_path
from .tasks import train_dataset_name


def lexical_scores(turns):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [0.0]
    for i in range(1, len(turns)):
        scores.append(max(scorer.score(turns[i], turns[j])["rougeL"].fmeasure for j in range(i)))
    return scores


class SemanticScorer:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def __call__(self, turns):
        import numpy as np

        if len(turns) == 1:
            return [0.0]
        emb = self.model.encode(turns, convert_to_numpy=True, show_progress_bar=False)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        return [0.0] + [float((emb[:i] @ emb[i]).max()) for i in range(1, len(turns))]


def trajectory_stagnation(scores):
    """Average turn-wise stagnation of one trajectory (paper Eq. 2, first turn included as 0)."""
    return sum(scores) / len(scores)


def main(args):
    dataset = args.dataset or train_dataset_name(args.task)
    score_fn = lexical_scores if args.metric == "lexical" else SemanticScorer()

    results = {}
    for exp in tqdm(read_examples(args.task, args.split, dataset)):
        tf = traj_path(args.output_root, args.task, args.tag, args.split, dataset, exp["id"])
        if not os.path.exists(tf):
            continue
        turns = assistant_turns(load_json(tf)["messages"])
        if not turns:
            raise ValueError(f"No assistant turns in {tf}")
        results[str(exp["id"])] = score_fn(turns)

    out = stagnation_path(args.output_root, args.task, args.tag, args.split, dataset, args.metric)
    dump_json(results, out)
    avg = sum(trajectory_stagnation(s) for s in results.values()) / max(len(results), 1)
    print(f"[stagnation] {args.metric}: {len(results)} trajectories, mean trajectory stagnation {avg:.4f} -> {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["math", "hotpotqa"])
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--metric", default="lexical", choices=["lexical", "semantic"])
    p.add_argument("--output_root", default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
