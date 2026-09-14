from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .data import prepare_eval_dataset
from .rewards import extract_answer_block, normalize_number, summarize_rewards


def load_model_and_tokenizer(config: dict[str, Any], adapter_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def generate_completion(model, tokenizer, prompt: str, config: dict[str, Any]) -> str:
    import torch

    eval_cfg = config["evaluation"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=eval_cfg["max_new_tokens"],
            do_sample=eval_cfg["temperature"] > 0,
            temperature=max(eval_cfg["temperature"], 1e-6),
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_evaluation_row(sample: dict[str, Any], completion: str) -> dict[str, Any]:
    predicted = normalize_number(extract_answer_block(completion))
    expected = normalize_number(sample["answer"])
    correct = predicted is not None and expected is not None and predicted == expected
    reward_summary = summarize_rewards([completion], [sample["answer"]])[0]
    return {
        "question": sample["question"],
        "completion": completion,
        "predicted_answer": str(predicted) if predicted is not None else None,
        "expected_answer": str(expected) if expected is not None else None,
        "correct": correct,
        "parse_failed": predicted is None,
        "rewards": reward_summary,
    }


def summarize_evaluation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(rows)
    reward_names = sorted(
        {
            reward_name
            for row in rows
            for reward_name in row.get("rewards", {}).keys()
        }
    )
    reward_means = {
        reward_name: mean(
            [
                float(row.get("rewards", {}).get(reward_name, 0.0))
                for row in rows
            ]
        )
        for reward_name in reward_names
    }
    return {
        "sample_count": sample_count,
        "accuracy": mean([1.0 if row.get("correct") else 0.0 for row in rows]),
        "parse_failure_rate": mean(
            [1.0 if row.get("parse_failed") else 0.0 for row in rows]
        ),
        "strict_format_rate": reward_means.get("strict_format", 0.0),
        "soft_format_rate": reward_means.get("soft_format", 0.0),
        "reward_means": reward_means,
    }


def evaluate(config_path: str, adapter_path: str | None = None, final: bool = False) -> Path:
    config = load_config(config_path)
    dataset = prepare_eval_dataset(config, final=final)
    model, tokenizer = load_model_and_tokenizer(config, adapter_path)
    rows: list[dict[str, Any]] = []

    for sample in dataset:
        completion = generate_completion(model, tokenizer, sample["prompt"], config)
        rows.append(build_evaluation_row(sample, completion))

    result = {
        "adapter_path": adapter_path,
        "metrics": summarize_evaluation_rows(rows),
        "examples": rows,
    }
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "final_eval.json" if final else "quick_eval.json"
    output_path = output_dir / name
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    output_path = evaluate(args.config, adapter_path=args.adapter_path, final=args.final)
    print(f"Wrote evaluation results to {output_path}")


if __name__ == "__main__":
    main()
