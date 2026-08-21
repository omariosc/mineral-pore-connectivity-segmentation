#!/bin/bash
#SBATCH --job-name=pore_screen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --array=0-14%1
#SBATCH --output=logs/pore_screen_%A_%a.out
#SBATCH --error=logs/pore_screen_%A_%a.err

set -euo pipefail
export PORE_RUN_MODE=screen
export PORE_LAUNCH_WRAPPER=screen
exec bash scripts/aire_confirmatory.slurm
