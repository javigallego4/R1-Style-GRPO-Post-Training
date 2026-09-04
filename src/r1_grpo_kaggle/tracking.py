from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


WANDB_SECRET_NAMES = ("WANDB_API_KEY", "wandb_api_key", "wandb_api")


def tracking_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.setdefault("tracking", {})


def default_run_name(config: dict[str, Any]) -> str:
    model_name = config["model"]["name"].split("/")[-1].replace("_", "-")
    dataset_name = config["dataset"]["name"].split("/")[-1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{model_name}-{dataset_name}-grpo-{stamp}"


def wandb_run_name(config: dict[str, Any]) -> str:
    configured = tracking_config(config).get("run_name")
    return configured or default_run_name(config)


def wandb_project_name(config: dict[str, Any]) -> str:
    return tracking_config(config).get("project_name") or config["project"]["name"]


def is_wandb_enabled(config: dict[str, Any]) -> bool:
    return bool(tracking_config(config).get("enabled", False))


def configured_secret_names(config: dict[str, Any]) -> tuple[str, ...]:
    secret_names = tracking_config(config).get("secret_names")
    if not secret_names:
        return WANDB_SECRET_NAMES
    return tuple(str(secret_name) for secret_name in secret_names)


def sanitized_tracking_config(config: dict[str, Any]) -> dict[str, Any]:
    safe_config = deepcopy(config)
    for key in ("api_key", "token", "password", "secret"):
        safe_config.get("tracking", {}).pop(key, None)
    safe_config.get("tracking", {}).pop("secret_names", None)
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


def ensure_wandb_api_key(secret_names: tuple[str, ...] = WANDB_SECRET_NAMES) -> bool:
    for secret_name in secret_names:
        value = os.environ.get(secret_name)
        if value:
            os.environ.setdefault("WANDB_API_KEY", value)
            return True

    value = kaggle_secret_value(secret_names)
    if value:
        os.environ.setdefault("WANDB_API_KEY", value)
        return True
    return False


def configure_wandb(config: dict[str, Any]) -> None:
    if not is_wandb_enabled(config):
        return

    tracking = tracking_config(config)
    run_name = wandb_run_name(config)
    tracking["run_name"] = run_name
    os.environ.setdefault("WANDB_PROJECT", wandb_project_name(config))
    os.environ.setdefault("WANDB_RUN_NAME", run_name)
    if tracking.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", str(tracking["entity"]))
    if tracking.get("mode"):
        os.environ.setdefault("WANDB_MODE", str(tracking["mode"]))
    ensure_wandb_api_key(configured_secret_names(config))


def wandb_init_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    tracking = tracking_config(config)
    kwargs: dict[str, Any] = {
        "project": wandb_project_name(config),
        "name": wandb_run_name(config),
        "config": sanitized_tracking_config(config),
        "reinit": False,
    }
    for config_key, wandb_key in (
        ("entity", "entity"),
        ("group", "group"),
        ("job_type", "job_type"),
        ("tags", "tags"),
        ("notes", "notes"),
        ("mode", "mode"),
    ):
        value = tracking.get(config_key)
        if value:
            kwargs[wandb_key] = value
    return kwargs


def initialize_wandb(config: dict[str, Any]) -> None:
    if not is_wandb_enabled(config):
        return

    configure_wandb(config)
    import wandb

    run = wandb.init(**wandb_init_kwargs(config))
    if tracking_config(config).get("log_code") and run is not None:
        run.log_code(".")
