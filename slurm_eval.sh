#!/bin/bash
#SBATCH --job-name=uni_eval
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

# ============================================================
# UNI Cell Instance Segmentation — evaluation job
# Usage: sbatch slurm_eval.sh [checkpoint] [tag]
# ============================================================

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uni_seg

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
cd "$PROJECT_DIR"

CKPT=${1:-checkpoints/best_phase2.pth}
TAG=${2:-uni_phase2_lora}
echo "Evaluating $CKPT (tag $TAG) on $(nvidia-smi --query-gpu=name --format=csv,noheader) at $(date)"
python evaluate.py --checkpoint "$CKPT" --tag "$TAG" --num-workers "${SLURM_CPUS_PER_TASK:-4}"
echo "Evaluation completed at $(date)"
