from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


FILE_ID = "128_J3kojPXMWCEN54gbf86H0aSVxO_cI"
ARCHIVE_NAME = "hierarchical-loss-pinn-data.zip"
REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = REPO_ROOT / ARCHIVE_NAME


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def ensure_unzip() -> None:
    if shutil.which("unzip") is None:
        raise RuntimeError("The 'unzip' command is required but was not found.")


def main() -> None:
    ensure_unzip()
    run(["python3", "-m", "pip", "install", "gdown"])
    run(
        [
            "python3",
            "-m",
            "gdown",
            f"https://drive.google.com/uc?id={FILE_ID}",
            "-O",
            ARCHIVE_NAME,
        ]
    )
    run(["unzip", "-o", ARCHIVE_NAME])
    ARCHIVE_PATH.unlink(missing_ok=True)
    print(f"Dataset archive downloaded and unpacked into {REPO_ROOT}")


if __name__ == "__main__":
    main()
