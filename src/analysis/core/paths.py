import os
from pathlib import Path


def root_dir() -> Path:
    return Path(os.environ.get("PYTHONPATH").split(";")[0])


DATA_DIR: Path = root_dir() / "data" / "currencies"
