#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cpu}"

EPOCHS_TEST1="${EPOCHS_TEST1:-20000}"
EPOCHS_TEST2="${EPOCHS_TEST2:-20000}"
EPOCHS_TEST3="${EPOCHS_TEST3:-20000}"
EPOCHS_TEST4="${EPOCHS_TEST4:-20000}"

PLOT_EVERY_2_4="${PLOT_EVERY_2_4:-2000}"

run_test1() {
  echo "=== test1: run_ablation ==="
  "$PYTHON_BIN" test1-therm-conduct/run_ablation.py \
    --device "$DEVICE" \
    --epochs "$EPOCHS_TEST1"

  echo "=== test1: plot_ablation_results ==="
  "$PYTHON_BIN" test1-therm-conduct/plot_ablation_results.py

  echo "=== test1: build_ablation_table ==="
  "$PYTHON_BIN" test1-therm-conduct/build_ablation_table.py
}

run_test2() {
  echo "=== test2: run_ablation ==="
  "$PYTHON_BIN" test2-2phase-disp-simple/run_ablation.py \
    --device "$DEVICE" \
    --epochs "$EPOCHS_TEST2" \
    --plot-every "$PLOT_EVERY_2_4" \
    --prepare-validation-gt

  echo "=== test2: plot_ablation_results ==="
  "$PYTHON_BIN" test2-2phase-disp-simple/plot_ablation_results.py

  echo "=== test2: build_ablation_table ==="
  "$PYTHON_BIN" test2-2phase-disp-simple/build_ablation_table.py
}

run_test3() {
  echo "=== test3: run_ablation ==="
  "$PYTHON_BIN" test3-2phase-disp-perm-nonlinear/run_ablation.py \
    --device "$DEVICE" \
    --epochs "$EPOCHS_TEST3" \
    --plot-every "$PLOT_EVERY_2_4" \
    --prepare-validation-gt

  echo "=== test3: plot_ablation_results ==="
  "$PYTHON_BIN" test3-2phase-disp-perm-nonlinear/plot_ablation_results.py

  echo "=== test3: build_ablation_table ==="
  "$PYTHON_BIN" test3-2phase-disp-perm-nonlinear/build_ablation_table.py
}

run_test4() {
  echo "=== test4: run_ablation ==="
  "$PYTHON_BIN" test4-2phase-disp-7d/run_ablation.py \
    --device "$DEVICE" \
    --epochs "$EPOCHS_TEST4" \
    --plot-every "$PLOT_EVERY_2_4" \
    --prepare-validation-gt

  echo "=== test4: plot_ablation_results ==="
  "$PYTHON_BIN" test4-2phase-disp-7d/plot_ablation_results.py

  echo "=== test4: build_ablation_table ==="
  "$PYTHON_BIN" test4-2phase-disp-7d/build_ablation_table.py
}

run_test1
run_test2
run_test3
run_test4
