import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from benchmarking.ablation_report_tools import run_table_cli


if __name__ == "__main__":
    run_table_cli(Path(__file__).resolve().parent)
