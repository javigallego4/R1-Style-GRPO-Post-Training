import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r1_grpo_kaggle.config import load_config
from r1_grpo_kaggle.data import prepare_eval_dataset, prepare_train_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_dataset = prepare_train_dataset(config)
    eval_dataset = prepare_eval_dataset(config)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print("First training sample:")
    sample = train_dataset[0]
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Prompt length: {len(sample['prompt'])}")


if __name__ == "__main__":
    main()

