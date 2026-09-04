from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = {
    "project",
    "model",
    "dataset",
    "prompt",
    "training",
    "lora",
    "rewards",
    "tracking",
    "export",
    "evaluation",
}

WANDB_MODES = {"online", "offline", "disabled"}


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    validate_config(config)
    return config


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def validate_config(config: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    if config["export"].get("publish_adapter", False):
        raise ValueError("Adapter publication must stay disabled by default.")

    if config["export"].get("save_adapter", True) and not config["export"].get("adapter_dir"):
        raise ValueError("export.adapter_dir is required when export.save_adapter is enabled.")

    tracking = config["tracking"]
    if not isinstance(tracking.get("enabled", False), bool):
        raise ValueError("tracking.enabled must be a boolean.")
    if not isinstance(tracking.get("project_name", ""), str):
        raise ValueError("tracking.project_name must be a string.")
    if tracking.get("mode") is not None and tracking["mode"] not in WANDB_MODES:
        raise ValueError("tracking.mode must be one of: disabled, offline, online.")
    if tracking.get("secret_names") is not None:
        secret_names = tracking["secret_names"]
        if not isinstance(secret_names, list) or not all(isinstance(name, str) for name in secret_names):
            raise ValueError("tracking.secret_names must be a list of strings.")
    for key in ("log_samples", "log_code", "log_adapter_artifacts", "log_model_artifacts"):
        if not isinstance(tracking.get(key, False), bool):
            raise ValueError(f"tracking.{key} must be a boolean.")

    if config["training"]["num_generations"] < 2:
        raise ValueError("GRPO requires at least two generations per prompt.")

    generation_batch_size = config["training"].get(
        "generation_batch_size",
        config["training"]["per_device_train_batch_size"],
    )
    if generation_batch_size % config["training"]["num_generations"] != 0:
        raise ValueError("training.generation_batch_size must be divisible by num_generations.")

    if config["dataset"]["train_size"] <= 0:
        raise ValueError("dataset.train_size must be positive.")

    if config["dataset"]["quick_eval_size"] <= 0:
        raise ValueError("dataset.quick_eval_size must be positive.")
