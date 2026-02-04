#!/bin/bash
#SBATCH --job-name=uni_eval
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

# ============================================================
# UNI Cell Instance Segmentation — Evaluation Job
# Usage: sbatch slurm_eval.sh
# ============================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate uni_seg

if [ -n "$SLURM_TMPDIR" ]; then
    export HF_HOME="${SLURM_TMPDIR}/hf_cache"
    export TORCH_HOME="${SLURM_TMPDIR}/torch_cache"
    mkdir -p "$HF_HOME" "$TORCH_HOME"
fi

PROJECT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$PROJECT_DIR"
mkdir -p results

echo "=============================="
echo "Evaluation Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================="

papermill cell_instance_segmentation.ipynb results/eval_${SLURM_JOB_ID}.ipynb \
    -p MODE "eval" \
    -p CHECKPOINT_PATH "checkpoints/best_phase2.pth" \
    -p NUM_WORKERS 4 \
    -k uni_seg

echo "Evaluation completed at $(date)"
