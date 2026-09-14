from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .data import prepare_eval_dataset
from .rewards import extract_answer_block, normalize_number, summarize_rewards
from .tracking import initialize_wandb, is_wandb_enabled


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


def evaluate_model(
    config: dict[str, Any],
    dataset,
    adapter_path: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    label = label or ("adapter" if adapter_path else "base")
    model, tokenizer = load_model_and_tokenizer(config, adapter_path)
    rows: list[dict[str, Any]] = []

    for sample in dataset:
        completion = generate_completion(model, tokenizer, sample["prompt"], config)
        rows.append(build_evaluation_row(sample, completion))

    result = {
        "label": label,
        "adapter_path": adapter_path,
        "metrics": summarize_evaluation_rows(rows),
        "examples": rows,
    }
    return result


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_examples_csv(path: Path, result: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "predicted_answer",
        "expected_answer",
        "correct",
        "parse_failed",
        "completion",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["examples"]:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return path


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "reward_means":
            for reward_name, reward_value in value.items():
                flattened[f"{prefix}/reward_mean/{reward_name}"] = float(reward_value)
        elif isinstance(value, int | float):
            flattened[f"{prefix}/{key}"] = float(value)
    return flattened


def compare_evaluation_results(
    base_result: dict[str, Any],
    adapter_result: dict[str, Any],
) -> dict[str, Any]:
    base_metrics = base_result["metrics"]
    adapter_metrics = adapter_result["metrics"]
    metric_keys = sorted(
        {
            key
            for metrics in (base_metrics, adapter_metrics)
            for key, value in metrics.items()
            if isinstance(value, int | float)
        }
    )
    deltas = {
        key: float(adapter_metrics.get(key, 0.0)) - float(base_metrics.get(key, 0.0))
        for key in metric_keys
    }
    reward_names = sorted(
        set(base_metrics.get("reward_means", {}))
        | set(adapter_metrics.get("reward_means", {}))
    )
    reward_deltas = {
        reward_name: float(adapter_metrics.get("reward_means", {}).get(reward_name, 0.0))
        - float(base_metrics.get("reward_means", {}).get(reward_name, 0.0))
        for reward_name in reward_names
    }
    return {
        "base": {
            "label": base_result["label"],
            "adapter_path": base_result["adapter_path"],
            "metrics": base_metrics,
        },
        "adapter": {
            "label": adapter_result["label"],
            "adapter_path": adapter_result["adapter_path"],
            "metrics": adapter_metrics,
        },
        "deltas": deltas,
        "reward_deltas": reward_deltas,
    }


def write_comparison_metrics_csv(path: Path, comparison: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric_name, base_value in comparison["base"]["metrics"].items():
        if not isinstance(base_value, int | float):
            continue
        adapter_value = comparison["adapter"]["metrics"].get(metric_name, 0.0)
        rows.append(
            {
                "metric": metric_name,
                "base": base_value,
                "adapter": adapter_value,
                "delta": comparison["deltas"].get(metric_name, 0.0),
            }
        )
    for reward_name, base_value in comparison["base"]["metrics"].get("reward_means", {}).items():
        adapter_value = comparison["adapter"]["metrics"].get("reward_means", {}).get(
            reward_name,
            0.0,
        )
        rows.append(
            {
                "metric": f"reward_mean/{reward_name}",
                "base": base_value,
                "adapter": adapter_value,
                "delta": comparison["reward_deltas"].get(reward_name, 0.0),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "base", "adapter", "delta"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_comparison_markdown(path: Path, comparison: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Comparison",
        "",
        f"- Base label: `{comparison['base']['label']}`",
        f"- Adapter label: `{comparison['adapter']['label']}`",
        f"- Adapter path: `{comparison['adapter']['adapter_path']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Base | Adapter | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric_name, delta in comparison["deltas"].items():
        base_value = comparison["base"]["metrics"].get(metric_name, 0.0)
        adapter_value = comparison["adapter"]["metrics"].get(metric_name, 0.0)
        lines.append(f"| {metric_name} | {base_value:.4f} | {adapter_value:.4f} | {delta:+.4f} |")
    if comparison["reward_deltas"]:
        lines.extend(["", "## Reward Component Means", "", "| Reward | Base | Adapter | Delta |", "|---|---:|---:|---:|"])
        for reward_name, delta in comparison["reward_deltas"].items():
            base_value = comparison["base"]["metrics"].get("reward_means", {}).get(reward_name, 0.0)
            adapter_value = comparison["adapter"]["metrics"].get("reward_means", {}).get(reward_name, 0.0)
            lines.append(f"| {reward_name} | {base_value:.4f} | {adapter_value:.4f} | {delta:+.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def log_evaluation_to_wandb(
    config: dict[str, Any],
    comparison: dict[str, Any],
    base_result: dict[str, Any],
    adapter_result: dict[str, Any],
    max_examples: int = 20,
) -> None:
    if not is_wandb_enabled(config):
        return

    initialize_wandb(config)
    import wandb

    payload = {}
    payload.update(flatten_metrics("eval/base", comparison["base"]["metrics"]))
    payload.update(flatten_metrics("eval/adapter", comparison["adapter"]["metrics"]))
    for metric_name, delta in comparison["deltas"].items():
        payload[f"eval/delta/{metric_name}"] = float(delta)
    for reward_name, delta in comparison["reward_deltas"].items():
        payload[f"eval/delta/reward_mean/{reward_name}"] = float(delta)
    wandb.log(payload)

    rows = []
    for base_row, adapter_row in zip(
        base_result["examples"][:max_examples],
        adapter_result["examples"][:max_examples],
    ):
        rows.append(
            [
                base_row["question"],
                base_row["expected_answer"],
                base_row["predicted_answer"],
                bool(base_row["correct"]),
                adapter_row["predicted_answer"],
                bool(adapter_row["correct"]),
                base_row["completion"],
                adapter_row["completion"],
            ]
        )
    if rows:
        table = wandb.Table(
            columns=[
                "question",
                "expected_answer",
                "base_predicted_answer",
                "base_correct",
                "adapter_predicted_answer",
                "adapter_correct",
                "base_completion",
                "adapter_completion",
            ],
            data=rows,
        )
        wandb.log({"eval/examples": table})


def evaluation_prefix(final: bool) -> str:
    return "final" if final else "quick"


def write_evaluation_result(
    result: dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> dict[str, str]:
    label = result["label"]
    json_path = write_json(output_dir / f"{label}_{prefix}_eval.json", result)
    csv_path = write_examples_csv(output_dir / f"{label}_{prefix}_examples.csv", result)
    return {
        "json": str(json_path),
        "examples_csv": str(csv_path),
    }


def evaluate(config_path: str, adapter_path: str | None = None, final: bool = False) -> Path:
    config = load_config(config_path)
    dataset = prepare_eval_dataset(config, final=final)
    result = evaluate_model(
        config,
        dataset,
        adapter_path=adapter_path,
        label="adapter" if adapter_path else "base",
    )
    output_dir = Path(config["evaluation"]["output_dir"])
    prefix = evaluation_prefix(final)
    output_path = Path(write_evaluation_result(result, output_dir, prefix)["json"])
    return output_path


def evaluate_comparison(
    config_path: str,
    adapter_path: str,
    final: bool = False,
    log_to_wandb: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    dataset = prepare_eval_dataset(config, final=final)
    output_dir = Path(config["evaluation"]["output_dir"])
    prefix = evaluation_prefix(final)

    base_result = evaluate_model(config, dataset, adapter_path=None, label="base")
    adapter_result = evaluate_model(
        config,
        dataset,
        adapter_path=adapter_path,
        label="adapter",
    )
    comparison = compare_evaluation_results(base_result, adapter_result)

    paths = {
        "base": write_evaluation_result(base_result, output_dir, prefix),
        "adapter": write_evaluation_result(adapter_result, output_dir, prefix),
        "comparison_json": str(write_json(output_dir / f"{prefix}_comparison.json", comparison)),
        "comparison_csv": str(
            write_comparison_metrics_csv(output_dir / f"{prefix}_comparison_metrics.csv", comparison)
        ),
        "comparison_md": str(
            write_comparison_markdown(output_dir / f"{prefix}_comparison.md", comparison)
        ),
    }
    if log_to_wandb:
        log_evaluation_to_wandb(config, comparison, base_result, adapter_result)

    return {
        "base": base_result,
        "adapter": adapter_result,
        "comparison": comparison,
        "paths": paths,
    }


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
