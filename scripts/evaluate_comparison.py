import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r1_grpo_kaggle.evaluate import evaluate_comparison


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    result = evaluate_comparison(
        args.config,
        adapter_path=args.adapter_path,
        final=args.final,
        log_to_wandb=not args.no_wandb,
    )
    print("Wrote evaluation comparison outputs:")
    for key, value in result["paths"].items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
