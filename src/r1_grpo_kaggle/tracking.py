from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def default_run_name(config: dict[str, Any]) -> str:
    model_name = config["model"]["name"].split("/")[-1].replace("_", "-")
    dataset_name = config["dataset"]["name"].split("/")[-1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{model_name}-{dataset_name}-grpo-{stamp}"


def wandb_run_name(config: dict[str, Any]) -> str:
    configured = config.get("tracking", {}).get("run_name")
    return configured or default_run_name(config)


def wandb_project_name(config: dict[str, Any]) -> str:
    return config.get("tracking", {}).get("project_name") or config["project"]["name"]


def is_wandb_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("tracking", {}).get("enabled", False))

