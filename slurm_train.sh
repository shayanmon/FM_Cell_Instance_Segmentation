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
# UNI Cell Instance Segmentation — Training Job
# Usage: sbatch slurm_train.sh
# ============================================================

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate uni_seg

# HuggingFace token — set this before submitting
# Option 1: export HF_TOKEN in your shell before sbatch
# Option 2: uncomment and set directly (less secure)
# export HF_TOKEN="hf_xxxxx"

# Cache directories — use local scratch if available
if [ -n "$SLURM_TMPDIR" ]; then
    export HF_HOME="${SLURM_TMPDIR}/hf_cache"
    export TORCH_HOME="${SLURM_TMPDIR}/torch_cache"
    mkdir -p "$HF_HOME" "$TORCH_HOME"
fi

# CUDA settings for V100
export CUDA_LAUNCH_BLOCKING=0

# Project directory
PROJECT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$PROJECT_DIR"

# Create output directories
mkdir -p logs checkpoints results

echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "Start time: $(date)"
echo "Working dir: $PROJECT_DIR"
echo "=============================="

# Run the notebook via papermill
# PHASE=1: Decoder-only training (frozen backbone)
# PHASE=2: LoRA fine-tuning
papermill cell_instance_segmentation.ipynb results/train_phase1_${SLURM_JOB_ID}.ipynb \
    -p MODE "train" \
    -p PHASE 1 \
    -p EPOCHS 100 \
    -p BATCH_SIZE 16 \
    -p LR 1e-3 \
    -p NUM_WORKERS 8 \
    -k uni_seg

echo "Phase 1 training completed at $(date)"

# Phase 2: LoRA fine-tuning
papermill cell_instance_segmentation.ipynb results/train_phase2_${SLURM_JOB_ID}.ipynb \
    -p MODE "train" \
    -p PHASE 2 \
    -p EPOCHS 50 \
    -p BATCH_SIZE 8 \
    -p LR 1e-4 \
    -p NUM_WORKERS 8 \
    -p CHECKPOINT_PATH "checkpoints/best_phase1.pth" \
    -k uni_seg

echo "Phase 2 training completed at $(date)"
echo "GPU utilization summary:"
nvidia-smi
