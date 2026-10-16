"""Task definitions: datasets, agent instruction, tools and evaluation sets for MATH and HotpotQA."""

from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data"

MAX_STEPS = 10  # interaction horizon (turns)
MAX_TOKENS = 2048  # max new tokens per turn

TEACHER_MODEL = "gpt-4.1-2025-04-14"
JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"

_FORMAT_INST = (
    "\n\nIMPORTANT: Always provide a 'Thought:' sequence, and a '<code>' sequence ending with '</code>', "
    "else you will fail."
)

TASKS = {
    "math": {
        # appended to every question
        "instruction": _FORMAT_INST
        + " For math problems that are not multiple-choice, always output the final answer using LaTeX \\boxed{} "
        "format. Provide the exact value (e.g., \\boxed{\\frac{9}{14}}), not a decimal approximation "
        "(e.g., \\boxed{0.642857}).",
        "authorized_imports": ["numpy", "sympy", "numpy.linalg", "fractions"],
        "use_retriever": False,
        "judge": "math",
        "datasets": {
            "train": {"math": "math/train/math.json"},
            "val": {"math": "math/val/math.json"},
            "test": {
                "math": "math/test/math.json",
                "omnimath_easy": "math/test/omnimath_easy.json",  # Omni-MATH levels 1-3
                "omnimath_medium": "math/test/omnimath_medium.json",  # Omni-MATH levels 4-5
            },
        },
        # name -> list of (split, dataset); accuracy is pooled over all listed sets
        "eval_sets": {
            "MATH": [("val", "math"), ("test", "math")],
            "Omni-MATH (L1-3)": [("test", "omnimath_easy")],
            "Omni-MATH (L4-5)": [("test", "omnimath_medium")],
        },
    },
    "hotpotqa": {
        "instruction": _FORMAT_INST
        + " You must base your final answer only on the information retrieved from tools. Do not use any prior "
        "knowledge.",
        "authorized_imports": ["numpy"],
        "use_retriever": True,
        "judge": "factual",
        "datasets": {
            "train": {"hotpotqa": "hotpotqa/train/hotpotqa.json"},
            "val": {"hotpotqa": "hotpotqa/val/hotpotqa.json"},
            "test": {"2wikimultihopqa": "hotpotqa/test/2wikimultihopqa.json"},
        },
        "eval_sets": {
            "HotpotQA": [("val", "hotpotqa")],
            "2WikiMultiHopQA": [("test", "2wikimultihopqa")],
        },
    },
}


def get_task(name):
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}. Choose from {list(TASKS)}")
    return TASKS[name]


def train_dataset_name(task):
    return next(iter(get_task(task)["datasets"]["train"]))


def make_tools(task, retriever_port="8005"):
    if not get_task(task)["use_retriever"]:
        return []
    from smolagents import WikipediaRetrieverTool

    return [WikipediaRetrieverTool(port=retriever_port)]
