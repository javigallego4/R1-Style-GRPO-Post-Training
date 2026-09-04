from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/javigallego4/R1-Style-GRPO-Post-Training.git"
KAGGLE_REPO_DIR = Path("/kaggle/working/r1-grpo-kaggle")


def configure_runtime_environment() -> None:
    os.environ.setdefault("UNSLOTH_VLLM_NO_FLASHINFER", "1")
    os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.check_call(command, cwd=cwd)


def resolve_project_dir() -> Path:
    if (ROOT / "requirements.txt").exists() and (ROOT / "src").exists():
        return ROOT

    if not KAGGLE_REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", REPO_URL, str(KAGGLE_REPO_DIR)])
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

    project_dir = resolve_project_dir()
    os.chdir(project_dir)
    install_dependencies(project_dir)
    sys.path.insert(0, str(project_dir / "src"))

    from r1_grpo_kaggle.train_grpo import train

    config_path = os.environ.get("CONFIG_PATH", "configs/smoke.yaml")
    train(config_path)


if __name__ == "__main__":
    main()
