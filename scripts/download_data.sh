#!/usr/bin/env bash
set -euo pipefail

FILE_ID="128_J3kojPXMWCEN54gbf86H0aSVxO_cI"
ARCHIVE_NAME="hierarchical-loss-pinn-data.zip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found."
  exit 1
fi

python3 -m pip install gdown
python3 -m gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${ARCHIVE_NAME}"
unzip -o "${ARCHIVE_NAME}"
rm -f "${ARCHIVE_NAME}"

echo "Dataset archive downloaded and unpacked into ${REPO_ROOT}"
