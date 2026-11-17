#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
usage() {
  cat <<'HELP'
Usage:
  scripts/pipeline.sh plan DATASET_ID [2d|3d_fullres]
  scripts/pipeline.sh train DATASET_ID CONFIG FOLD [extra nnU-Net arguments]
  scripts/pipeline.sh predict DATASET_ID CONFIG FOLD INPUT_DIR OUTPUT_DIR [extra arguments]

Example: scripts/pipeline.sh train 501 2d 0 --npz
Prediction uses exactly the specified fold, not an untrained five-fold ensemble.
Activate your Python environment (or use Makefile) and prepare a real dataset first. No pretrained weights are bundled.
HELP
}
if [[ $# -eq 0 || "$1" == "--help" ]]; then usage; exit 0; fi
stage="$1"; shift
case "$stage" in
  plan)
    [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
    exec nnUNetv2_plan_and_preprocess -d "$1" -c "${2:-2d}" --verify_dataset_integrity
    ;;
  train)
    [[ $# -ge 3 ]] || { usage; exit 2; }
    dataset="$1"; configuration="$2"; fold="$3"; shift 3
    exec nnUNetv2_train "$dataset" "$configuration" "$fold" "$@"
    ;;
  predict)
    [[ $# -ge 5 ]] || { usage; exit 2; }
    dataset="$1"; configuration="$2"; fold="$3"; input="$4"; output="$5"; shift 5
    exec nnUNetv2_predict -d "$dataset" -c "$configuration" -f "$fold" -i "$input" -o "$output" "$@"
    ;;
  *) usage; exit 2 ;;
esac
