#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m src.train_mcq \
  --config configs/experiment.yaml \
  --data-dir data/raw \
  --output-dir outputs/models/run_mcq \
  --report-dir outputs/reports

python -m src.evaluate_model \
  --model-dir outputs/models/run_mcq \
  --data-dir data/raw \
  --split validation \
  --pred-dir outputs/predictions \
  --report-dir outputs/reports
