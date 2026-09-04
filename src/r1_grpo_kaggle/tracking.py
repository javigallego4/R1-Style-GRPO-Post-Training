from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


WANDB_SECRET_NAMES = ("WANDB_API_KEY", "wandb_api_key", "wandb_api")


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


def sanitized_tracking_config(config: dict[str, Any]) -> dict[str, Any]:
    safe_config = deepcopy(config)
    for key in ("api_key", "token", "password", "secret"):
        safe_config.get("tracking", {}).pop(key, None)
    return safe_config


def kaggle_secret_value(secret_names: tuple[str, ...] = WANDB_SECRET_NAMES) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None

    user_secrets = UserSecretsClient()
    for secret_name in secret_names:
        try:
            value = user_secrets.get_secret(secret_name)
        except Exception:
            continue
        if value:
            return value
    return None


def ensure_wandb_api_key() -> bool:
    for secret_name in WANDB_SECRET_NAMES:
        value = os.environ.get(secret_name)
        if value:
            os.environ.setdefault("WANDB_API_KEY", value)
            return True

    value = kaggle_secret_value()
    if value:
        os.environ.setdefault("WANDB_API_KEY", value)
        return True
    return False


def configure_wandb(config: dict[str, Any]) -> None:
    if not is_wandb_enabled(config):
        return

    run_name = wandb_run_name(config)
    config.setdefault("tracking", {})["run_name"] = run_name
    os.environ.setdefault("WANDB_PROJECT", wandb_project_name(config))
    os.environ.setdefault("WANDB_RUN_NAME", run_name)
    ensure_wandb_api_key()


def initialize_wandb(config: dict[str, Any]) -> None:
    if not is_wandb_enabled(config):
        return

    configure_wandb(config)
    import wandb

    wandb.init(
        project=wandb_project_name(config),
        name=wandb_run_name(config),
        config=sanitized_tracking_config(config),
        reinit=False,
    )
