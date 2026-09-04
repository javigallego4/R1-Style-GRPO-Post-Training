from __future__ import annotations

import importlib.util
import os
import platform
import sys


PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "wandb",
    "unsloth",
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    for package in PACKAGES:
        status = "OK" if importlib.util.find_spec(package) else "missing"
        print(f"{package}: {status}")
    print(f"WANDB_API_KEY available: {bool(os.environ.get('WANDB_API_KEY'))}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(idx)
            supported = major >= 7
            print(f"GPU {idx}: {torch.cuda.get_device_name(idx)} sm_{major}{minor} supported: {supported}")
    except Exception as exc:
        print(f"torch/CUDA check failed: {exc}")


if __name__ == "__main__":
    main()
