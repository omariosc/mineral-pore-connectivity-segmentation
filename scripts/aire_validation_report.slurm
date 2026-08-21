#!/bin/bash
#SBATCH --job-name=pore_val_report
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/pore_val_report_%j.out
#SBATCH --error=logs/pore_val_report_%j.err

# Sole supported AIRE entry point for post-freeze validation-screen reporting.
# This is a non-array, validation-only inference job. It fixes the selected
# lock, data roots, output root, device, and L40S allocation in source control.

set -eo pipefail

fail_with_message() {
  echo "$1" >&2
  exit 2
}

variable_is_set() {
  declare -p "$1" >/dev/null 2>&1
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  fail_with_message "validation reporting requires an active Slurm allocation"
fi
if [[ "${SLURM_JOB_PARTITION:-}" != "gpu" ]]; then
  fail_with_message "validation reporting requires the gpu partition"
fi
if variable_is_set SLURM_ARRAY_TASK_ID; then
  fail_with_message "validation reporting must not run as a Slurm array"
fi

# The reporter itself has no output override. Reject common aliases as well so
# an inherited research shell cannot appear to alter the canonical invocation.
for forbidden in \
  PORE_PROJECT_DIR PORE_SELECTED_METHOD_LOCK PORE_VALIDATION_REPORT_LOCK \
  PORE_CHECKPOINT PORE_EVALUATION_CHECKPOINT PORE_OUTPUT_DIR \
  PORE_REPORT_OUTPUT_DIR PORE_ANNOTATIONS PORE_IMAGE_DIR PORE_MASK_DIR \
  PORE_SPLIT_MANIFEST PORE_MODEL_TYPE PORE_DEVICE QUICK_TEST SINGLE_BATCH \
  DISABLE_AMP; do
  if variable_is_set "$forbidden"; then
    fail_with_message "validation reporting forbids $forbidden"
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
  fail_with_message "validation reporting requires an allocated NVIDIA L40S GPU"
fi
export PORE_ALLOCATED_GPU_NAME="$pore_gpu_name"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

exec python3 scripts/report_validation_screen_tiles.py \
  --selected-method-lock config/selected_method_lock.json \
  --device cuda
