#!/bin/bash
#SBATCH --job-name=pore_selected
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/pore_selected_%A_%a.out
#SBATCH --error=logs/pore_selected_%A_%a.err

set -euo pipefail
export PORE_RUN_MODE=selected_retrain
export PORE_LAUNCH_WRAPPER=selected_retrain
exec bash scripts/aire_confirmatory.slurm
