"""Turning serialized trajectories into token ids for training.

The pipeline mirrors what trl==0.18.1's SFTTrainer does for conversational data
(apply chat template -> tokenize without special tokens -> truncate to max_length),
plus one ProD-specific fix for Qwen3: an empty ``<think>\\n\\n</think>\\n\\n`` block is
inserted in *every* assistant turn, matching the prompt the agent sees at inference
time (``enable_thinking=False``). The datasets are tokenized here and handed to the
trainers already processed, so no trl internals need to be patched.
"""

import importlib.resources

import yaml
from transformers import AutoTokenizer

from smolagents import FinalAnswerTool
from smolagents.agents import populate_template
from smolagents.models import get_clean_message_list


CODE_BLOCK_TAGS = ("<code>", "</code>")

QWEN3_EMPTY_THINK = "<think>\n\n</think>\n\n"
QWEN3_ASSISTANT = "<|im_start|>assistant\n"


def short_system_prompt():
    """The compact system prompt used by distilled students (smolagents ``system_prompt_short``).

    Note: as in the paper's experiments, it is rendered with only the ``final_answer`` tool listed,
    for every task, and replaces the system prompt of all training trajectories.
    """
    templates = yaml.safe_load(
        importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
    )
    return populate_template(
        templates["system_prompt_short"],
        variables={
            "tools": {"final_answer": FinalAnswerTool()},
            "code_block_opening_tag": CODE_BLOCK_TAGS[0],
            "code_block_closing_tag": CODE_BLOCK_TAGS[1],
        },
    )


def clean_messages(messages, system_prompt):
    """Serialized smolagents memory -> plain chat messages (tool responses become user turns)."""
    messages = [m for m in messages if m["role"] != "tool-call"]
    messages = get_clean_message_list(
        messages,
        role_conversions={"tool-response": "user"},
        flatten_messages_as_text=True,
    )
    if messages[-1]["role"] == "user":
        messages = messages[:-1]
    messages[0]["content"] = system_prompt
    return messages


def model_family(model_id):
    if "gemma" in model_id:
        return "gemma"
    if "Llama" in model_id:
        return "llama"
    return "qwen"


def response_instruction_templates(model_id):
    """Markers of assistant / user turns, used to restrict the loss to assistant tokens."""
    return {
        "gemma": ("<start_of_turn>model\n", "<start_of_turn>user\n"),
        "llama": ("<|start_header_id|>assistant<|end_header_id|>\n\n", "<|start_header_id|>user<|end_header_id|>\n\n"),
        "qwen": (QWEN3_ASSISTANT + QWEN3_EMPTY_THINK, "<|im_start|>user\n"),
    }[model_family(model_id)]


def collator_tokenizer(model_id):
    """Tokenizer used by the data collators for padding (left padding, family-specific pad token)."""
    family = model_family(model_id)
    kwargs = {"padding_side": "left", "add_eos_token": True}
    if family == "llama":
        kwargs["pad_token"] = "<|eot_id|>"
    elif family == "qwen":
        kwargs["pad_token"] = "<|endoftext|>"
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def render(messages, tokenizer):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if "<think>\n\n</think>" in text:
        parts = text.split(QWEN3_ASSISTANT)
        for i in range(1, len(parts)):
            if not parts[i].startswith(QWEN3_EMPTY_THINK):
                parts[i] = QWEN3_EMPTY_THINK + parts[i]
        text = QWEN3_ASSISTANT.join(parts)
    return text


def tokenize(messages, tokenizer, max_length=None):
    """Chat template -> ids, without adding special tokens (the template already has them), then truncate."""
    enc = tokenizer(text=render(messages, tokenizer), add_special_tokens=False)
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    if max_length is not None:
        input_ids, attention_mask = input_ids[:max_length], attention_mask[:max_length]
    return input_ids, attention_mask
