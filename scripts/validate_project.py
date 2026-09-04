import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r1_grpo_kaggle.config import load_config
from r1_grpo_kaggle.data import build_prompt
from r1_grpo_kaggle.rewards import summarize_rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    prompt = build_prompt("Janet has 3 apples and buys 4 more. How many apples does she have?", config)
    completion = "<reasoning>\n3 + 4 = 7\n</reasoning>\n<answer>\n7\n</answer>"
    reward_rows = summarize_rewards([completion], ["7"])

    print("Config OK")
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"W&B project: {config['tracking']['project_name']}")
    print(f"Adapter publication enabled: {config['export']['publish_adapter']}")
    print(f"Prompt preview length: {len(prompt)}")
    print(f"Reward smoke test: {reward_rows[0]}")


if __name__ == "__main__":
    main()
