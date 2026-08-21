#!/bin/bash
#SBATCH --job-name=pore_classical_fit
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pore_classical_fit_%j.out
#SBATCH --error=logs/pore_classical_fit_%j.err

# Train-only B0/B1/B2 freeze job. This wrapper never invokes held-out
# evaluation. Submit the separate locked-evaluation job only after the neural
# selection and selected-winner retraining campaigns are frozen.

set -eo pipefail

fail_with_message() {
  echo "$1" >&2
  exit 2
}

variable_is_set() {
  declare -p "$1" >/dev/null 2>&1
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  fail_with_message "classical fitting requires an active Slurm allocation"
fi
if [[ "${SLURM_JOB_PARTITION:-}" != "nodes" ]]; then
  fail_with_message "classical fitting requires the nodes partition"
fi
if variable_is_set SLURM_ARRAY_TASK_ID; then
  fail_with_message "classical fitting must run once, not as a Slurm array"
fi

# A caller may name one prospective campaign explicitly. Otherwise the active
# Slurm job creates a unique, reproducible public-safe campaign identifier.
pore_classical_campaign="${PORE_CLASSICAL_CAMPAIGN_ID:-classical-fit-${SLURM_JOB_ID}}"
if [[ -z "$pore_classical_campaign" ]] ||
   (( ${#pore_classical_campaign} > 96 )) ||
   [[ ! "$pore_classical_campaign" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  fail_with_message "PORE_CLASSICAL_CAMPAIGN_ID must be a public-path-safe campaign identifier"
fi

# The fitter receives only the source-controlled split and canonical train
# roots. It has no annotation-index argument and authenticates all 74 training
# filenames plus their input/mask byte aggregates before fitting.
for forbidden in \
  PORE_PROJECT_DIR PORE_CLASSICAL_LOCK PORE_LOCK PORE_CHECKPOINT \
  PORE_MODEL PORE_MODEL_PATH PORE_OUTPUT_DIR PORE_EVALUATION_OUTPUT_DIR \
  PORE_ANNOTATIONS PORE_ANNOTATION_INDEX PORE_IMAGE_DIR PORE_MASK_DIR \
  PORE_DATA_DIR PORE_SPLIT_MANIFEST PORE_DEVICE PORE_COMPARATOR \
  PORE_THRESHOLD PORE_AREA_CUTOFF PORE_BOOTSTRAP_REPLICATES \
  PORE_CLASSICAL_FIT_ID PORE_CLASSICAL_FIT_PATH \
  PORE_NEURAL_FREEZE_ID PORE_NEURAL_FREEZE_MANIFEST \
  QUICK_TEST SINGLE_BATCH; do
  if variable_is_set "$forbidden"; then
    fail_with_message "classical fitting forbids $forbidden"
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

exec python3 scripts/fit_classical_comparators.py \
  --campaign-id "$pore_classical_campaign" \
  --split-manifest config/confirmatory_splits.json \
  --image-dir original_images \
  --mask-dir results/step2_pore_classification/pore_classifications
