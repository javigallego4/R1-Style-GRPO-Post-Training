from __future__ import annotations

import os

from kaggle_entry import main


if __name__ == "__main__":
    os.environ.setdefault("CONFIG_PATH", "configs/kaggle_pilot.yaml")
    main()
