#!/bin/bash
#SBATCH --job-name=pore_locked_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/pore_locked_eval_%A_%a.out
#SBATCH --error=logs/pore_locked_eval_%A_%a.err

# This is the only supported AIRE entry point for held-out neural evaluation.
# Submit the primary and plain-U-Net roles as separate arrays.  Each array is
# serial (%1), and each task deterministically selects one of the three frozen
# selected-retraining seeds through a content-addressed neural-freeze manifest.

set -eo pipefail

fail_with_message() {
  echo "$1" >&2
  exit 2
}

variable_is_set() {
  declare -p "$1" >/dev/null 2>&1
}

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  fail_with_message "locked evaluation requires an active Slurm array allocation"
fi

pore_task_index="$SLURM_ARRAY_TASK_ID"
case "$pore_task_index" in
  0|1|2) ;;
  *) fail_with_message "locked evaluation SLURM_ARRAY_TASK_ID must be 0, 1, or 2" ;;
esac

pore_neural_freeze_id="${PORE_NEURAL_FREEZE_ID:-}"
if [[ ! "$pore_neural_freeze_id" =~ ^neural-freeze-[0-9a-f]{16}$ ]]; then
  fail_with_message "PORE_NEURAL_FREEZE_ID must be a canonical neural-freeze identifier"
fi

pore_architecture_role="${PORE_SELECTED_ARCHITECTURE_ROLE:-}"
case "$pore_architecture_role" in
  primary_multiscale|plain_unet_comparator) ;;
  *) fail_with_message "PORE_SELECTED_ARCHITECTURE_ROLE must be primary_multiscale or plain_unet_comparator" ;;
esac

# No user-provided data, checkpoint, model, device, or output substitutions are
# accepted on this held-out path. The content-addressed neural freeze is the
# sole scientific artifact identifier supplied at submission.
for forbidden in \
  PORE_SELECTED_RETRAIN_CAMPAIGN_ID PORE_SELECTED_METHOD_LOCK \
  PORE_PROJECT_DIR PORE_CHECKPOINT PORE_EVALUATION_CHECKPOINT \
  PORE_OUTPUT_DIR PORE_EVALUATION_OUTPUT_DIR PORE_ANNOTATIONS \
  PORE_IMAGE_DIR PORE_MASK_DIR PORE_SPLIT_MANIFEST PORE_MODEL_TYPE \
  PORE_DEVICE PORE_EXPERIMENT_PREFIX QUICK_TEST SINGLE_BATCH \
  DISABLE_AMP; do
  if variable_is_set "$forbidden"; then
    fail_with_message "locked evaluation forbids $forbidden"
  fi
done

pore_project_dir="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"
cd "$pore_project_dir"

module purge
module load cuda/12.6.2 miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pore-seg
set -u

pore_gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
if [[ "$pore_gpu_name" != *L40S* ]]; then
  fail_with_message "locked evaluation requires an allocated NVIDIA L40S GPU"
fi

# Resolve and authenticate the exact freeze/role/task checkpoint before the
# evaluator is allowed to reserve or read any held-out data.
python3 - "$pore_neural_freeze_id" "$pore_architecture_role" "$pore_task_index" <<'PY'
import json
import sys
from pathlib import Path

from scripts.evaluate_confirmatory_checkpoint import (
    validate_neural_freeze_locked_evaluation_identity,
)
from src.training.neural_freeze import load_verified_neural_freeze_manifest


freeze_id, architecture_role, cell_index = sys.argv[1:]
verified = load_verified_neural_freeze_manifest(freeze_id, Path.cwd())
checkpoint, _lock, output, identity = validate_neural_freeze_locked_evaluation_identity(
    verified,
    architecture_role=architecture_role,
    cell_index=int(cell_index),
)
print(json.dumps({
    "checkpoint": checkpoint.as_posix(),
    "output": output.as_posix(),
    "reservation_key": identity["reservation_key"],
}, sort_keys=True))
PY

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

pore_evaluation_command=(
  python3 scripts/evaluate_confirmatory_checkpoint.py
  --neural-freeze-id "$pore_neural_freeze_id"
  --architecture-role "$pore_architecture_role"
  --cell-index "$pore_task_index"
  --annotations results/step3_coco_dataset/pore_annotations.json
  --image-dir results/step3_coco_dataset/images
  --mask-dir results/step2_pore_classification/pore_classifications
  --split-manifest config/confirmatory_splits.json
  --device cuda
)

exec "${pore_evaluation_command[@]}"
