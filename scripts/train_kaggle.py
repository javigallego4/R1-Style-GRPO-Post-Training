import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r1_grpo_kaggle.train_grpo import main


if __name__ == "__main__":
    main()
