from __future__ import annotations

import argparse
import json

from r1_grpo_kaggle.config import load_config
from r1_grpo_kaggle.tracking import run_wandb_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal W&B connectivity probe.")
    parser.add_argument("--config", default="configs/kaggle_smoke.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_wandb_probe(config)
    print("W&B probe result:")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
