#!/bin/bash
#SBATCH --job-name=pore_classical_eval
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pore_classical_eval_%j.out
#SBATCH --error=logs/pore_classical_eval_%j.err

# Sole supported AIRE entry point for the exactly-once classical held-out pass.
# One non-array allocation evaluates B0, B1, and B2 together from a previously
# frozen train-only campaign lock. No validation corpus is used.

set -eo pipefail

fail_with_message() {
  echo "$1" >&2
  exit 2
}

variable_is_set() {
  declare -p "$1" >/dev/null 2>&1
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  fail_with_message "classical held-out evaluation requires an active Slurm allocation"
fi
if [[ "${SLURM_JOB_PARTITION:-}" != "nodes" ]]; then
  fail_with_message "classical held-out evaluation requires the nodes partition"
fi
if variable_is_set SLURM_ARRAY_TASK_ID; then
  fail_with_message "classical held-out evaluation must not run as a Slurm array"
fi

pore_classical_fit_id="${PORE_CLASSICAL_FIT_ID:-}"
if [[ ! "$pore_classical_fit_id" =~ ^classical-fit-[0-9a-f]{16}$ ]]; then
  fail_with_message "PORE_CLASSICAL_FIT_ID must be the canonical content-addressed classical fit ID"
fi
pore_neural_freeze_id="${PORE_NEURAL_FREEZE_ID:-}"
if [[ ! "$pore_neural_freeze_id" =~ ^neural-freeze-[0-9a-f]{16}$ ]]; then
  fail_with_message "PORE_NEURAL_FREEZE_ID must be the canonical content-addressed neural-freeze ID"
fi

# Canonical paths, predictor identities, model digest, output location, split,
# and CPU execution are source-controlled. Environment overrides fail closed.
for forbidden in \
  PORE_PROJECT_DIR PORE_CLASSICAL_LOCK PORE_LOCK PORE_CHECKPOINT \
  PORE_MODEL PORE_MODEL_PATH PORE_OUTPUT_DIR PORE_EVALUATION_OUTPUT_DIR \
  PORE_ANNOTATIONS PORE_ANNOTATION_INDEX PORE_IMAGE_DIR PORE_MASK_DIR \
  PORE_DATA_DIR PORE_SPLIT_MANIFEST PORE_DEVICE PORE_COMPARATOR \
    PORE_THRESHOLD PORE_AREA_CUTOFF PORE_BOOTSTRAP_REPLICATES \
    PORE_NEURAL_FREEZE_MANIFEST PORE_NEURAL_FREEZE_PATH \
    PORE_CLASSICAL_CAMPAIGN_ID PORE_CLASSICAL_FIT_PATH \
    QUICK_TEST SINGLE_BATCH; do
  if variable_is_set "$forbidden"; then
    fail_with_message "classical held-out evaluation forbids $forbidden"
  fi
done

pore_project_dir="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"
cd "$pore_project_dir"

module purge
module load miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pore-seg
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

exec python3 scripts/evaluate_locked_classical_comparators.py \
  --fit-id "$pore_classical_fit_id" \
  --neural-freeze-id "$pore_neural_freeze_id"
