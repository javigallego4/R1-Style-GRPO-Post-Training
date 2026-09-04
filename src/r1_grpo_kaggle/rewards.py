from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
REASONING_BLOCK_RE = re.compile(
    r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL
)
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")
GSM8K_FINAL_RE = re.compile(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)")


def normalize_number(value: str | int | float | Decimal | None) -> Decimal | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def extract_gsm8k_answer(answer: str) -> str | None:
    match = GSM8K_FINAL_RE.search(answer)
    if match:
        return match.group(1).replace(",", "")
    numbers = NUMBER_RE.findall(answer)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion or "")


def extract_answer_block(completion: Any) -> str | None:
    text = completion_to_text(completion)
    match = ANSWER_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def has_strict_format(completion: Any) -> bool:
    text = completion_to_text(completion).strip()
    pattern = re.compile(
        r"^<reasoning>\s*.+?\s*</reasoning>\s*<answer>\s*.+?\s*</answer>$",
        re.IGNORECASE | re.DOTALL,
    )
    return bool(pattern.match(text))


def has_soft_format(completion: Any) -> bool:
    text = completion_to_text(completion)
    return bool(REASONING_BLOCK_RE.search(text) and ANSWER_BLOCK_RE.search(text))


def correctness_reward_func(
    prompts: list[str] | None = None,
    completions: list[str] | None = None,
    answer: list[str] | None = None,
    **_: object,
) -> list[float]:
    rewards: list[float] = []
    for completion, expected in zip(completions or [], answer or []):
        predicted_number = normalize_number(extract_answer_block(completion))
        expected_number = normalize_number(expected)
        rewards.append(1.0 if predicted_number is not None and predicted_number == expected_number else 0.0)
    return rewards


def integer_answer_reward_func(
    completions: list[str] | None = None,
    **_: object,
) -> list[float]:
    rewards: list[float] = []
    for completion in completions or []:
        parsed = normalize_number(extract_answer_block(completion))
        rewards.append(1.0 if parsed is not None and parsed == parsed.to_integral_value() else 0.0)
    return rewards


def strict_format_reward_func(
    completions: list[str] | None = None,
    **_: object,
) -> list[float]:
    return [1.0 if has_strict_format(completion) else 0.0 for completion in completions or []]


def soft_format_reward_func(
    completions: list[str] | None = None,
    **_: object,
) -> list[float]:
    return [1.0 if has_soft_format(completion) else 0.0 for completion in completions or []]


def reasoning_tags_reward_func(
    completions: list[str] | None = None,
    **_: object,
) -> list[float]:
    rewards: list[float] = []
    for completion in completions or []:
        score = 0.0
        text = completion_to_text(completion)
        if "<reasoning>" in text:
            score += 0.25
        if "</reasoning>" in text:
            score += 0.25
        if "<answer>" in text:
            score += 0.25
        if "</answer>" in text:
            score += 0.25
        rewards.append(score)
    return rewards


def reward_functions() -> list:
    return [
        correctness_reward_func,
        integer_answer_reward_func,
        strict_format_reward_func,
        soft_format_reward_func,
        reasoning_tags_reward_func,
    ]


def summarize_rewards(completions: Iterable[str], answers: Iterable[str]) -> list[dict[str, float]]:
    completions_list = list(completions)
    answers_list = list(answers)
    components = {
        "correctness": correctness_reward_func(completions=completions_list, answer=answers_list),
        "integer_answer": integer_answer_reward_func(completions=completions_list),
        "strict_format": strict_format_reward_func(completions=completions_list),
        "soft_format": soft_format_reward_func(completions=completions_list),
        "reasoning_tags": reasoning_tags_reward_func(completions=completions_list),
    }
    rows: list[dict[str, float]] = []
    for idx in range(len(completions_list)):
        row = {name: values[idx] for name, values in components.items()}
        row["total"] = sum(row.values())
        rows.append(row)
    return rows
