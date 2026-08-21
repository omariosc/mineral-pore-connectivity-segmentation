#!/bin/bash
#SBATCH --job-name=pore_confirm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pore_confirm_%A_%a.out
#SBATCH --error=logs/pore_confirm_%A_%a.err

set -eo pipefail

pore_project_dir="${PORE_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}"
cd "$pore_project_dir"

module purge
module load cuda/12.6.2 miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pore-seg
set -u

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export BOOTSTRAP_FACTOR=1

pore_run_mode="${PORE_RUN_MODE:-}"
pore_array_index="${SLURM_ARRAY_TASK_ID:-}"
pore_screen_candidates=(R3 H3 C2-P C2-F C2-FP)
pore_seeds=(42 123 2025)

fail_with_message() {
  echo "$1" >&2
  exit 2
}

variable_is_set() {
  declare -p "$1" >/dev/null 2>&1
}

if [[ -z "$pore_run_mode" ]]; then
  fail_with_message "PORE_RUN_MODE must be set explicitly by an approved wrapper"
fi
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_JOB_ID:-}" || -z "$pore_array_index" ]]; then
  fail_with_message "protocol runs require an active Slurm array allocation"
fi
if [[ "${PORE_LAUNCH_WRAPPER:-}" != "$pore_run_mode" ]]; then
  fail_with_message "direct main-script launch is forbidden; use the mode-specific Slurm wrapper"
fi
pore_gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "$pore_gpu_name" != *L40S* ]]; then
  fail_with_message "protocol runs require an allocated NVIDIA L40S GPU"
fi
export PORE_ALLOCATED_GPU_NAME="$pore_gpu_name"

validate_experiment_prefix() {
  local prefix="$1"
  if (( ${#prefix} > 64 )) ||
     [[ ! "$prefix" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    fail_with_message "PORE_EXPERIMENT_PREFIX must be 1-64 characters using only letters, digits, '.', '_' or '-'"
  fi
}

configure_candidate() {
  local candidate="$1"
  pore_candidate="$candidate"
  pore_evaluation_patch_size=2048
  pore_evaluation_batch_size=1
  pore_dropout=0.2
  pore_conditional=0
  pore_reference_weights=0
  case "$candidate" in
    R3)
      pore_model_type="multiscale_attention_unet"
      pore_loss_type="focal_dice"
      pore_num_classes=3
      pore_training_patch_size=683
      pore_training_batch_size=4
      pore_reference_weights=1
      pore_candidate_slug="r3"
      ;;
    H3)
      pore_model_type="multiscale_attention_unet"
      pore_loss_type="hierarchical_pore_connectivity"
      pore_num_classes=3
      pore_training_patch_size=683
      pore_training_batch_size=4
      pore_candidate_slug="h3"
      ;;
    C2-P)
      pore_model_type="multiscale_attention_unet"
      pore_loss_type="conditional_pore_focal_dice"
      pore_num_classes=2
      pore_training_patch_size=683
      pore_training_batch_size=4
      pore_conditional=1
      pore_candidate_slug="c2-p"
      ;;
    C2-F)
      pore_model_type="multiscale_attention_unet"
      pore_loss_type="conditional_pore_focal_dice"
      pore_num_classes=2
      pore_training_patch_size=2048
      pore_training_batch_size=1
      pore_conditional=1
      pore_candidate_slug="c2-f"
      ;;
    C2-FP)
      pore_model_type="multiscale_attention_unet_pyramid"
      pore_loss_type="conditional_pore_focal_dice"
      pore_num_classes=2
      pore_training_patch_size=2048
      pore_training_batch_size=1
      pore_conditional=1
      pore_dropout=0.0
      pore_candidate_slug="c2-fp"
      ;;
    *)
      fail_with_message "candidate must be one of R3, H3, C2-P, C2-F, or C2-FP"
      ;;
  esac
}

pore_validation_only=0
pore_protocol_candidate=""
pore_protocol_run_role=""
pore_protocol_campaign_id=""
pore_protocol_cell_index=""
pore_selected_method_lock=""
pore_selected_architecture_role=""
pore_smoke_preflight_manifest=""
pore_max_batches=""

case "$pore_run_mode" in
  screen)
    for forbidden in PORE_SCREEN_CANDIDATE PORE_MODEL_TYPE PORE_EXPERIMENT_PREFIX PORE_MAX_BATCHES QUICK_TEST SINGLE_BATCH; do
      if variable_is_set "$forbidden"; then
        fail_with_message "scientific screen forbids $forbidden"
      fi
    done
    if [[ "${DISABLE_AMP:-0}" != "0" || "${DISABLE_TRANSFORMS:-0}" != "0" ]]; then
      fail_with_message "scientific screen requires AMP and conservative transforms"
    fi
    if variable_is_set PORE_EPOCHS && [[ "$PORE_EPOCHS" != "30" ]]; then
      fail_with_message "scientific screen fixes PORE_EPOCHS=30"
    fi
    if [[ "${PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE:-0}" != "1" ]]; then
      fail_with_message "scientific screen requires PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE=1"
    fi
    if [[ -z "${PORE_SMOKE_PREFLIGHT_MANIFEST:-}" ]]; then
      fail_with_message "scientific screen requires PORE_SMOKE_PREFLIGHT_MANIFEST"
    fi
    if (( pore_array_index < 0 || pore_array_index >= 15 )); then
      fail_with_message "screen SLURM_ARRAY_TASK_ID must be in 0..14"
    fi
    pore_candidate_index=$((pore_array_index / 3))
    pore_seed_index=$((pore_array_index % 3))
    pore_seed="${pore_seeds[$pore_seed_index]}"
    configure_candidate "${pore_screen_candidates[$pore_candidate_index]}"
    pore_epochs=30
    pore_validation_only=1
    pore_protocol_candidate="$pore_candidate"
    pore_protocol_run_role="validation_screen_cell"
    pore_protocol_campaign_id="$SLURM_ARRAY_JOB_ID"
    pore_protocol_cell_index="$pore_array_index"
    pore_smoke_preflight_manifest="$PORE_SMOKE_PREFLIGHT_MANIFEST"
    pore_experiment_prefix="screen_${SLURM_ARRAY_JOB_ID}_${pore_array_index}_${pore_candidate_slug}"
    ;;

  smoke)
    for forbidden in PORE_MAX_BATCHES PORE_SMOKE_MAX_BATCHES PORE_SMOKE_CANDIDATE QUICK_TEST SINGLE_BATCH PORE_EPOCHS; do
      if variable_is_set "$forbidden"; then
        fail_with_message "immutable smoke mode forbids $forbidden"
      fi
    done
    if [[ "${PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE:-0}" != "1" ]]; then
      fail_with_message "smoke campaign requires PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE=1"
    fi
    if (( pore_array_index < 0 || pore_array_index >= ${#pore_seeds[@]} )); then
      fail_with_message "smoke SLURM_ARRAY_TASK_ID must be 0, 1, or 2"
    fi
    pore_smoke_candidates=(R3 C2-F C2-FP)
    pore_seed=42
    configure_candidate "${pore_smoke_candidates[$pore_array_index]}"
    pore_max_batches=2
    pore_epochs=1
    pore_validation_only=1
    pore_protocol_candidate="$pore_candidate"
    pore_protocol_run_role="validation_smoke_cell"
    pore_protocol_campaign_id="$SLURM_ARRAY_JOB_ID"
    pore_protocol_cell_index="$pore_array_index"
    pore_experiment_prefix="smoke_${SLURM_ARRAY_JOB_ID}_${pore_array_index}_${pore_candidate_slug}"
    ;;

  selected_retrain)
    for forbidden in PORE_MODEL_TYPE PORE_EXPERIMENT_PREFIX PORE_MAX_BATCHES QUICK_TEST SINGLE_BATCH; do
      if variable_is_set "$forbidden"; then
        fail_with_message "selected retraining forbids $forbidden"
      fi
    done
    if [[ "${DISABLE_AMP:-0}" != "0" || "${DISABLE_TRANSFORMS:-0}" != "0" ]]; then
      fail_with_message "selected retraining requires AMP and conservative transforms"
    fi
    if variable_is_set PORE_EPOCHS && [[ "$PORE_EPOCHS" != "30" ]]; then
      fail_with_message "selected retraining fixes PORE_EPOCHS=30"
    fi
    if [[ -z "${PORE_SELECTED_METHOD_KEY:-}" || -z "${PORE_SELECTED_METHOD_LOCK:-}" || -z "${PORE_SELECTED_ARCHITECTURE_ROLE:-}" ]]; then
      fail_with_message "selected retraining requires method key, lock, and architecture role"
    fi
    if (( pore_array_index < 0 || pore_array_index >= ${#pore_seeds[@]} )); then
      fail_with_message "selected retraining SLURM_ARRAY_TASK_ID must be 0, 1, or 2"
    fi
    pore_seed="${pore_seeds[$pore_array_index]}"
    configure_candidate "$PORE_SELECTED_METHOD_KEY"
    case "$PORE_SELECTED_ARCHITECTURE_ROLE" in
      primary_multiscale)
        ;;
      plain_unet_comparator)
        pore_model_type="plain_unet"
        ;;
      *)
        fail_with_message "PORE_SELECTED_ARCHITECTURE_ROLE must be primary_multiscale or plain_unet_comparator"
        ;;
    esac
    if (( pore_conditional == 1 )) &&
       [[ "${PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE:-0}" != "1" ]]; then
      fail_with_message "conditional selected retraining requires PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE=1"
    fi
    pore_epochs=30
    pore_validation_only=1
    pore_protocol_candidate="$pore_candidate"
    pore_protocol_run_role="selected_winner_retraining"
    pore_protocol_campaign_id="$SLURM_ARRAY_JOB_ID"
    pore_protocol_cell_index="$pore_array_index"
    pore_selected_method_lock="$PORE_SELECTED_METHOD_LOCK"
    pore_selected_architecture_role="$PORE_SELECTED_ARCHITECTURE_ROLE"
    pore_experiment_prefix="selected_${SLURM_ARRAY_JOB_ID}_${pore_array_index}_${pore_candidate_slug}_${PORE_SELECTED_ARCHITECTURE_ROLE}"
    ;;

  *)
    fail_with_message "PORE_RUN_MODE must be screen, smoke, or selected_retrain"
    ;;
esac

validate_experiment_prefix "$pore_experiment_prefix"

pore_train_command=(
  python3 scripts/train_patches.py
  --epochs "$pore_epochs"
  --batch-size "$pore_training_batch_size"
  --workers 8
  --lr 5e-4
  --weight-decay 1e-4
  --patch-size "$pore_training_patch_size"
  --evaluation-patch-size "$pore_evaluation_patch_size"
  --evaluation-batch-size "$pore_evaluation_batch_size"
  --annotations-path results/step3_coco_dataset/pore_annotations.json
  --image-dir results/step3_coco_dataset/images
  --mask-dir results/step2_pore_classification/pore_classifications
  --split-manifest config/confirmatory_splits.json
  --num-classes "$pore_num_classes"
  --model-type "$pore_model_type"
  --loss-type "$pore_loss_type"
  --focal-gamma 2.0
  --dropout "$pore_dropout"
  --optimizer adamw
  --scheduler cosine
  --augmentation conservative
  --seed "$pore_seed"
  --early-stopping
  --early-stopping-patience 10
  --mixed-precision
  --gradient-clip-val 1.0
  --save-every-n-epochs 10
  --experiment-name "${pore_experiment_prefix}_seed${pore_seed}"
)

if (( pore_reference_weights == 1 )); then
  pore_train_command+=(--class-weights 3 2 1)
fi
if (( pore_validation_only == 1 )); then
  pore_train_command+=(--validation-only)
fi
if [[ -n "$pore_protocol_candidate" ]]; then
  pore_train_command+=(--protocol-candidate-key "$pore_protocol_candidate")
fi
if [[ -n "$pore_protocol_run_role" ]]; then
  pore_train_command+=(--protocol-run-role "$pore_protocol_run_role")
fi
if [[ -n "$pore_protocol_campaign_id" ]]; then
  pore_train_command+=(
    --protocol-campaign-id "$pore_protocol_campaign_id"
    --protocol-cell-index "$pore_protocol_cell_index"
  )
fi
if (( pore_conditional == 1 )); then
  pore_train_command+=(
    --conditional-pore-threshold 100
    --acknowledge-recovered-threshold-rule
  )
fi
if [[ -n "$pore_selected_method_lock" ]]; then
  pore_train_command+=(--selected-method-lock "$pore_selected_method_lock")
fi
if [[ -n "$pore_selected_architecture_role" ]]; then
  pore_train_command+=(
    --selected-architecture-role "$pore_selected_architecture_role"
  )
fi
if [[ -n "$pore_smoke_preflight_manifest" ]]; then
  pore_train_command+=(
    --smoke-preflight-manifest "$pore_smoke_preflight_manifest"
  )
fi
if [[ -n "$pore_max_batches" ]]; then
  pore_train_command+=(--max-batches "$pore_max_batches")
fi

set +e
"${pore_train_command[@]}"
pore_training_status=$?
set -e

if (( pore_training_status != 0 )); then
  if [[ "$pore_run_mode" == "screen" || "$pore_run_mode" == "smoke" ]]; then
    pore_failure_command=(
      python3 scripts/record_protocol_cell_failure.py
      --run-role "$pore_protocol_run_role"
      --campaign-id "$pore_protocol_campaign_id"
      --cell-index "$pore_protocol_cell_index"
      --candidate "$pore_protocol_candidate"
      --seed "$pore_seed"
      --exit-code "$pore_training_status"
    )
    if [[ -n "$pore_smoke_preflight_manifest" ]]; then
      pore_failure_command+=(
        --smoke-preflight-manifest "$pore_smoke_preflight_manifest"
      )
    fi
    if ! "${pore_failure_command[@]}"; then
      echo "Could not persist the immutable failed-cell outcome" >&2
    fi
  fi
  exit "$pore_training_status"
fi
