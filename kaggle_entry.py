from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def install_dependencies() -> None:
    if os.environ.get("INSTALL_DEPS", "1") != "1":
        print("Skipping dependency installation because INSTALL_DEPS is not 1.")
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")]
    )


def main() -> None:
    install_dependencies()
    sys.path.insert(0, str(ROOT / "src"))

    from r1_grpo_kaggle.train_grpo import train

    config_path = os.environ.get("CONFIG_PATH", "configs/smoke.yaml")
    train(config_path)


if __name__ == "__main__":
    main()
