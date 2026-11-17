# Source this file from any directory: source /absolute/path/scripts/env.sh
BRAIN_MRI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export nnUNet_raw="${nnUNet_raw:-$BRAIN_MRI_ROOT/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$BRAIN_MRI_ROOT/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$BRAIN_MRI_ROOT/artifacts/nnUNet_results}"
mkdir -p -- "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
