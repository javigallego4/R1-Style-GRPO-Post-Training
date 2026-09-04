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

    if config["training"]["num_generations"] < 2:
        raise ValueError("GRPO requires at least two generations per prompt.")

    if config["dataset"]["train_size"] <= 0:
        raise ValueError("dataset.train_size must be positive.")

    if config["dataset"]["quick_eval_size"] <= 0:
        raise ValueError("dataset.quick_eval_size must be positive.")
