#!/bin/bash
#SBATCH --job-name=uni_cell_seg
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# ============================================================
# UNI Cell Instance Segmentation — single-GPU training job (Phase 1 + Phase 2 + eval)
# Usage: sbatch slurm_train.sh        (submit from the project directory)
# Uses the same torchrun/DDP code path as multi-GPU runs, with one process.
# ============================================================

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uni_seg

# UNI is gated: run `hf auth login` once on a login node (token is stored in
# ~/.cache/huggingface/token), or export HF_TOKEN before sbatch.

# SLURM runs a spooled copy of this script, so locate the project via the submit dir
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
cd "$PROJECT_DIR"

echo "=============================="
echo "Job ID: ${SLURM_JOB_ID:-n/a} | Node: ${SLURM_NODELIST:-n/a}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start time: $(date) | Working dir: $PROJECT_DIR"
echo "=============================="

NPROC=1 NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8} bash scripts/train_ddp.sh

echo "Finished at $(date)"
