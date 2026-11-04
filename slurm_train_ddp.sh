#!/bin/bash
#SBATCH --job-name=uni_seg_ddp
#SBATCH --output=train_ddp_%j.out
#SBATCH --error=train_ddp_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1

# ============================================================
# UNI Cell Instance Segmentation — multi-GPU DDP training (single node)
# Usage: sbatch slurm_train_ddp.sh    (submit from the project directory)
# One SLURM task; torchrun starts one process per allocated GPU (NCCL backend).
# Global batch stays 16 (world x micro-batch x accumulation), so results are
# comparable with single-GPU runs.
# ============================================================

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uni_seg

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
cd "$PROJECT_DIR"

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "=============================="
echo "Job ID: ${SLURM_JOB_ID:-n/a} | Node: ${SLURM_NODELIST:-n/a} | GPUs: $NGPU"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Start time: $(date) | Working dir: $PROJECT_DIR"
echo "=============================="

# Workers per process: split the CPU allocation across ranks
NPROC=$NGPU NUM_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-16} / NGPU )) bash scripts/train_ddp.sh

echo "Finished at $(date)"
nvidia-smi
