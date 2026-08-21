#!/bin/bash
#SBATCH --job-name=pore_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/pore_smoke_%A_%a.out
#SBATCH --error=logs/pore_smoke_%A_%a.err

set -euo pipefail
export PORE_RUN_MODE=smoke
export PORE_LAUNCH_WRAPPER=smoke
exec bash scripts/aire_confirmatory.slurm
