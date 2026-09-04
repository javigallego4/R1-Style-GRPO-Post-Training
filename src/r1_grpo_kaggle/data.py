from __future__ import annotations

from typing import Any

from .rewards import extract_gsm8k_answer


def build_prompt(question: str, config: dict[str, Any]) -> str:
    system = config["prompt"].get("system", "").strip()
    template = config["prompt"]["template"]
    user_prompt = template.format(question=question).strip()
    if not system:
        return user_prompt
    return f"{system}\n\n{user_prompt}"


def load_gsm8k_dataset(config: dict[str, Any]):
    from datasets import load_dataset

    dataset_cfg = config["dataset"]
    return load_dataset(dataset_cfg["name"], dataset_cfg.get("subset", "main"))


def prepare_split(config: dict[str, Any], split_name: str, sample_size: int | None = None):
    dataset = load_gsm8k_dataset(config)[split_name]
    seed = int(config.get("project", {}).get("seed", 42))
    if sample_size is not None:
        sample_size = min(sample_size, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    def convert(example: dict[str, str]) -> dict[str, str]:
        final_answer = extract_gsm8k_answer(example["answer"])
        if final_answer is None:
            final_answer = ""
        return {
            "prompt": build_prompt(example["question"], config),
            "question": example["question"],
            "reference_solution": example["answer"],
            "answer": final_answer,
        }

    return dataset.map(convert, remove_columns=dataset.column_names)


def prepare_train_dataset(config: dict[str, Any]):
    dataset_cfg = config["dataset"]
    return prepare_split(
        config,
        split_name=dataset_cfg["train_split"],
        sample_size=dataset_cfg.get("train_size"),
    )


def prepare_eval_dataset(config: dict[str, Any], final: bool = False):
    dataset_cfg = config["dataset"]
    size_key = "final_eval_size" if final else "quick_eval_size"
    return prepare_split(
        config,
        split_name=dataset_cfg["eval_split"],
        sample_size=dataset_cfg.get(size_key),
    )
