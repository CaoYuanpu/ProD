"""Score saved trajectories with an LLM judge (gpt-4.1-mini), as in the paper (Appendix B.2).

    python -m prod.judge --task math --tag sft_Qwen3-1.7B --split eval
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel
from tqdm import tqdm

from .common import dump_json, eval_path, load_json, read_examples, traj_path
from .tasks import JUDGE_MODEL, get_task, train_dataset_name


FACTUAL_PROMPT = """Compare the following predicted answer with the gold (correct) answer and determine if they match in meaning given the question.
Be reasonable - phrasing differences are okay if the core meaning is the same. You should accept the aliases and the answer conveying the same conclusion.
If the predicted answer is overly verbose and fails to capture the key information found in the gold answer (i.e., low recall), consider it a false answer.

Question: {question}
Predicted answer: {predicted}
Gold answer: {gold}

Output your evaluation as a JSON object with two fields:
- explanation: Brief explanation of your scoring decision
- score: 0 or 1 indicating if answers match
"""

MATH_PROMPT = """Compare the following predicted answer with the correct answer and determine if they are mathematically equivalent.

Question: {question}
Predicted answer: {predicted}
Correct answer: {gold}

Output your evaluation as a JSON object with two fields:
- explanation: Brief explanation of your scoring decision
- score: 0 or 1 indicating if answers match
"""


class Verdict(BaseModel):
    explanation: str
    score: int


def judge_answer(client, kind, question, predicted, gold):
    prompt = (MATH_PROMPT if kind == "math" else FACTUAL_PROMPT).format(
        question=question, predicted=predicted, gold=gold
    )
    response = client.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=Verdict,
    )
    verdict = response.choices[0].message.parsed
    return {
        "score": verdict.score,
        "explanation": verdict.explanation,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }


def score_one(client, kind, traj_file, out_file):
    traj = load_json(traj_file)
    result = {k: traj[k] for k in ["id", "question", "total_steps", "true_answer", "generated_answer", "state"]}
    if traj["state"] != "success":  # ran out of steps / crashed without a final answer
        result.update(score=0, explanation="State is not success.")
    else:
        result.update(judge_answer(client, kind, traj["question"], traj["generated_answer"], traj["true_answer"]))
    dump_json(result, out_file)
    return result["score"]


def main(args):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    kind = get_task(args.task)["judge"]
    dataset = args.dataset or train_dataset_name(args.task)

    jobs = []
    for exp in read_examples(args.task, args.split, dataset):
        traj_file = traj_path(args.output_root, args.task, args.tag, args.split, dataset, exp["id"])
        out_file = eval_path(args.output_root, args.task, args.tag, args.split, dataset, exp["id"])
        if os.path.exists(traj_file) and not os.path.exists(out_file):
            jobs.append((traj_file, out_file))

    with ThreadPoolExecutor(args.num_workers) as pool:
        scores = list(tqdm(pool.map(lambda j: score_one(client, kind, *j), jobs), total=len(jobs)))
    print(f"[judge] scored {len(scores)} new trajectories ({sum(scores)} correct)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["math", "hotpotqa"])
    p.add_argument("--split", required=True, choices=["train", "eval"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
