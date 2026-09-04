from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def entrypoint_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


ROOT = entrypoint_root()
REPO_URL = "https://github.com/javigallego4/R1-Style-GRPO-Post-Training.git"
KAGGLE_REPO_DIR = Path("/kaggle/working/r1-grpo-kaggle")


def configure_runtime_environment() -> None:
    os.environ.setdefault("UNSLOTH_VLLM_NO_FLASHINFER", "1")
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")


def bootstrap_wandb_from_kaggle_secret() -> None:
    if os.environ.get("WANDB_API_KEY"):
        print("Direct Kaggle secret bootstrap: WANDB_API_KEY already available from environment.")
        return

    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        print("Direct Kaggle secret bootstrap: kaggle_secrets is not available.")
        return

    user_secrets = UserSecretsClient()
    try:
        secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")
    except Exception as exc:
        print(f"Direct Kaggle secret bootstrap: WANDB_API_KEY unavailable ({exc.__class__.__name__}).")
        return

    if secret_value_0:
        os.environ["WANDB_API_KEY"] = secret_value_0
        print("Direct Kaggle secret bootstrap: WANDB_API_KEY loaded.")
    else:
        print("Direct Kaggle secret bootstrap: WANDB_API_KEY returned an empty value.")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.check_call(command, cwd=cwd)


def resolve_project_dir() -> Path:
    if (ROOT / "requirements.txt").exists() and (ROOT / "src").exists():
        return ROOT

    if not KAGGLE_REPO_DIR.exists():
        repo_ref = os.environ.get("KAGGLE_REPO_REF", "main")
        run(["git", "clone", "--depth", "1", "--branch", repo_ref, REPO_URL, str(KAGGLE_REPO_DIR)])
    return KAGGLE_REPO_DIR


def install_dependencies(project_dir: Path) -> None:
    if os.environ.get("INSTALL_DEPS", "1") != "1":
        print("Skipping dependency installation because INSTALL_DEPS is not 1.")
        return

    if os.environ.get("KAGGLE_STRICT_UNSLOTH_SETUP", "1") == "1":
        run([sys.executable, "-m", "pip", "install", "-q", "pip3-autoremove"])
        subprocess.run(
            ["pip-autoremove", "torch", "torchvision", "torchaudio", "vllm", "-y"],
            check=False,
        )
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torch",
                "torchvision",
                "torchaudio",
                "xformers",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
        )
        run([sys.executable, "-m", "pip", "install", "-q", "unsloth", "vllm", "pynvml==12.0.0", "ninja"])
        run([sys.executable, "-m", "pip", "install", "-q", "datasets", "pyyaml", "wandb"])
        return

    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(project_dir / "requirements.txt")])


def main() -> None:
    configure_runtime_environment()
    bootstrap_wandb_from_kaggle_secret()

    project_dir = resolve_project_dir()
    os.chdir(project_dir)
    install_dependencies(project_dir)
    sys.path.insert(0, str(project_dir / "src"))

    from r1_grpo_kaggle.config import load_config
    from r1_grpo_kaggle.tracking import (
        configure_wandb,
        configured_secret_names,
        kaggle_secret_diagnostics,
        run_wandb_probe,
        wandb_status,
    )
    from r1_grpo_kaggle.train_grpo import train

    config_path = os.environ.get("CONFIG_PATH", "configs/kaggle_smoke.yaml")
    print(f"Project directory: {project_dir}")
    print(f"Config path: {config_path}")
    config = load_config(config_path)
    configure_wandb(config)
    print(f"Kaggle secret preflight: {kaggle_secret_diagnostics(configured_secret_names(config))}")
    print(f"W&B preflight: {wandb_status(config)}")
    if os.environ.get("WANDB_PROBE_ONLY", "0") == "1":
        print("Running W&B probe only because WANDB_PROBE_ONLY=1.")
        print(f"W&B probe result: {run_wandb_probe(config)}")
        return
    train(config_path)


if __name__ == "__main__":
    main()
