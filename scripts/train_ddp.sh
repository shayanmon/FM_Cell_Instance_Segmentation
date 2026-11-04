#!/bin/bash
# ============================================================
# UNI Cell Instance Segmentation — DDP training on Linux (workstation or inside a SLURM job)
# ============================================================
# Usage:  bash scripts/train_ddp.sh            # one process per visible GPU (NCCL)
#         NPROC=2 bash scripts/train_ddp.sh
#         BACKBONE=imagenet RUN_NAME=ablation_imagenet bash scripts/train_ddp.sh   # ImageNet ViT-L/16 ablation
# Global batch stays 16 for any NPROC. Re-running resumes from checkpoints/latest_phase*.pth.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
export NO_ALBUMENTATIONS_UPDATE=1
NPROC=${NPROC:-$(python -c "import torch; print(max(1, torch.cuda.device_count()))")}
PHASE1_EPOCHS=${PHASE1_EPOCHS:-100}
PHASE2_EPOCHS=${PHASE2_EPOCHS:-50}
NUM_WORKERS=${NUM_WORKERS:-8}
BACKBONE=${BACKBONE:-uni}            # uni | imagenet; also the prefix of the evaluation tags
RUN_NAME=${RUN_NAME:-}               # write to checkpoints/<RUN_NAME>, logs/<RUN_NAME>, results/<RUN_NAME>
SAVE_DIR=checkpoints; LOG_DIR=logs; RESULTS_DIR=results
if [ -n "$RUN_NAME" ]; then SAVE_DIR=checkpoints/$RUN_NAME; LOG_DIR=logs/$RUN_NAME; RESULTS_DIR=results/$RUN_NAME; fi
UNI_ARGS=()
if [ -n "${UNI_WEIGHTS:-}" ] && [ "$BACKBONE" = uni ]; then UNI_ARGS=(--uni-weights "$UNI_WEIGHTS"); fi
DIRS=(--save-dir "$SAVE_DIR" --log-dir "$LOG_DIR" --results-dir "$RESULTS_DIR")
mkdir -p "$LOG_DIR" "$SAVE_DIR" "$RESULTS_DIR"

echo "DDP processes: $NPROC | backbone: $BACKBONE | output: $SAVE_DIR, $LOG_DIR, $RESULTS_DIR"
if [ "$BACKBONE" = uni ]; then python scripts/check_uni_access.py ${UNI_ARGS[@]+"${UNI_ARGS[@]}"}; fi

torchrun --standalone --nproc_per_node="$NPROC" train.py --backbone "$BACKBONE" "${DIRS[@]}" --phase 1 \
    --epochs "$PHASE1_EPOCHS" --batch-size 16 --lr 1e-3 --num-workers "$NUM_WORKERS" ${UNI_ARGS[@]+"${UNI_ARGS[@]}"}

torchrun --standalone --nproc_per_node="$NPROC" train.py --backbone "$BACKBONE" "${DIRS[@]}" --phase 2 \
    --epochs "$PHASE2_EPOCHS" --batch-size 16 --lr-lora 1e-5 --lr-decoder 1e-4 --num-workers "$NUM_WORKERS" \
    ${UNI_ARGS[@]+"${UNI_ARGS[@]}"}

EVAL=(python evaluate.py --results-dir "$RESULTS_DIR")
"${EVAL[@]}" --checkpoint "$SAVE_DIR/best_phase1.pth" --tag "${BACKBONE}_phase1" ${UNI_ARGS[@]+"${UNI_ARGS[@]}"}
"${EVAL[@]}" --checkpoint "$SAVE_DIR/best_phase2.pth" --tag "${BACKBONE}_phase2_lora" ${UNI_ARGS[@]+"${UNI_ARGS[@]}"}
# Same checkpoints with the notebook's post-processing, for comparison with v1 numbers
"${EVAL[@]}" --checkpoint "$SAVE_DIR/best_phase1.pth" --tag "${BACKBONE}_phase1_notebook_pp" --postproc notebook \
    --num-vis 0 ${UNI_ARGS[@]+"${UNI_ARGS[@]}"}
"${EVAL[@]}" --checkpoint "$SAVE_DIR/best_phase2.pth" --tag "${BACKBONE}_phase2_lora_notebook_pp" --postproc notebook \
    --num-vis 0 ${UNI_ARGS[@]+"${UNI_ARGS[@]}"}
