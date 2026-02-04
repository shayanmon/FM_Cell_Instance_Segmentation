#!/bin/bash
#SBATCH --job-name=uni_seg_ddp
#SBATCH --output=train_ddp_%j.out
#SBATCH --error=train_ddp_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1

# ============================================================
# UNI Cell Instance Segmentation — Multi-GPU DDP Training
# Usage: sbatch slurm_train_ddp.sh
# Requests 2 GPUs on 1 node, runs 1 process per GPU via srun
# ============================================================

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate uni_seg

# HuggingFace token
# export HF_TOKEN="hf_xxxxx"

# Cache directories
if [ -n "$SLURM_TMPDIR" ]; then
    export HF_HOME="${SLURM_TMPDIR}/hf_cache"
    export TORCH_HOME="${SLURM_TMPDIR}/torch_cache"
    mkdir -p "$HF_HOME" "$TORCH_HOME"
fi

export CUDA_LAUNCH_BLOCKING=0

# DDP environment variables (NCCL)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

PROJECT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results data/processed

echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Tasks (processes): $SLURM_NTASKS"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Start time: $(date)"
echo "Working dir: $PROJECT_DIR"
echo "=============================="

# Phase 1 with DDP (srun launches SLURM_NTASKS processes)
srun papermill cell_instance_segmentation.ipynb results/train_phase1_ddp_${SLURM_JOB_ID}.ipynb \
    -p MODE "train" \
    -p PHASE 1 \
    -p EPOCHS 100 \
    -p BATCH_SIZE 16 \
    -p LR 1e-3 \
    -p NUM_WORKERS 8 \
    -p USE_DDP True \
    -k uni_seg

echo "Phase 1 DDP training completed at $(date)"

# Phase 2 with DDP
srun papermill cell_instance_segmentation.ipynb results/train_phase2_ddp_${SLURM_JOB_ID}.ipynb \
    -p MODE "train" \
    -p PHASE 2 \
    -p EPOCHS 50 \
    -p BATCH_SIZE 8 \
    -p LR 1e-4 \
    -p NUM_WORKERS 8 \
    -p USE_DDP True \
    -p CHECKPOINT_PATH "checkpoints/best_phase1.pth" \
    -k uni_seg

echo "Phase 2 DDP training completed at $(date)"
echo "GPU utilization summary:"
nvidia-smi
