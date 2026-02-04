# Changelog

All notable changes to the cell instance segmentation notebook and supporting files.

---

## [2026-02-04] — Phase 2 Robustness & OOM Fixes

### Fixed

- **Phase 2 checkpoint loading: fallback chain** (Cell 33): Now tries `best_phase1.pth` first, then falls back to `latest_phase1.pth` if best doesn't exist, instead of failing silently when only the latest checkpoint was saved
  - **Rationale:** Phase 1 early-stopped at epoch 78 but `best_phase1.pth` was not saved due to an `is_main_process()` gating issue. Only `latest_phase1.pth` existed. The old code hardcoded `best_phase1.pth` and printed a warning, then trained Phase 2 from an untrained model.

- **Phase 2 checkpoint loading: LoRA-aware state_dict matching** (Cell 33): Detects whether the checkpoint was saved with or without LoRA layers and whether the model already has LoRA applied, then handles all four combinations correctly
  - **Rationale:** Re-running notebook cells caused `RuntimeError: Missing key(s) in state_dict` because the model already had LoRA layers (`qkv.original.weight`, `qkv.lora_A`, `qkv.lora_B`) but the Phase 1 checkpoint had plain keys (`qkv.weight`). The fix inspects both sides and rebuilds the model fresh if needed.

- **Phase 2 `best_phase2.pth` save indentation bug** (Cell 33): `torch.save` for best checkpoint was outside the `if val_metrics < best_val_loss` block, causing it to save every epoch regardless of improvement
  - **Rationale:** The `if is_main_process():` guard was at the wrong indentation level — it paired with the `else` for patience counting instead of being nested inside the improvement check. This meant `best_phase2.pth` was overwritten every epoch, potentially with worse weights.

- **Phase 2 decoder LR coupled to global `LR` parameter** (Cell 33): Hardcoded `PHASE2_LR_DECODER = 1e-4` instead of `LR if PHASE == 2 else 1e-4`
  - **Rationale:** When running Phase 2 directly (setting `PHASE=2` in the parameters cell), the global `LR=1e-3` was used for the decoder — 10x higher than intended. This destabilizes the decoder weights that Phase 1 carefully converged. The decoder LR should always be 1e-4 in Phase 2 regardless of the global LR setting.

- **OOM on 12GB GPUs during Phase 2** (Cell 33): Reduced default `PHASE2_BS` from 8 to 4, and added `del model; torch.cuda.empty_cache()` before rebuilding the model in the LoRA mismatch path
  - **Rationale:** The model rebuild path allocated a second full copy of the 304M-parameter model before the old one was freed, exceeding 12GB VRAM. Phase 2 also requires more memory than Phase 1 because LoRA adds backward-pass activations for the previously-frozen backbone attention layers. Batch size 4 with gradient accumulation of 4 maintains the same effective batch size of 16.

- **Phase 2 rebuild uses `pretrained_backbone=False`** (Cell 33): The model rebuild no longer attempts to download UNI weights from HuggingFace
  - **Rationale:** The rebuild path creates a temporary model solely to load the Phase 1 checkpoint into — the `load_state_dict` call immediately overwrites all weights. Downloading UNI weights (or failing with a 403 on gated repos) was wasted time and a confusing error message.

### Added

- **`PATIENCE = 15` explicitly set in Phase 2** (Cell 33): Phase 2 now has its own early stopping patience variable instead of relying on an inherited value from Phase 1
- **`optimizer_state_dict` added to `best_phase2.pth` save** (Cell 33): Enables proper analysis and potential resume from the best Phase 2 checkpoint

---

## [Unreleased] — Classification Fix

**Status:** Retraining required (loss function and post-processing changed)

### Changed

- **CLASS_WEIGHTS** `[0.1, 0.3, 3.0, 5.0]` &rarr; `[0.05, 0.5, 5.0, 8.0]` (Cell 4)
  - **Rationale:** The NC (classification) head was predicting background (class 0) for nearly all pixels inside detected cells. With the old weight of 0.1 on background, the model could predict "background everywhere" and barely be penalized — since ~70% of pixels are background, this was the path of least resistance. Reducing background weight to 0.05 and boosting minority classes (WBC 3.0&rarr;5.0, Platelet 5.0&rarr;8.0) forces the CE loss to prioritize learning rare cell types.

- **NC loss weight** `1.0` &rarr; `2.0` to match HV weight (Cell 27, CombinedLoss)
  - **Rationale:** The combined loss was `1.0*NP + 2.0*HV + 1.0*NC`. With the classification branch at half the weight of spatial regression, the model optimized heavily for "find and separate cells" (NP + HV) but underinvested in "classify them" (NC). Equalizing the weight gives the NC head proportional gradient signal.

- **Post-processing class assignment** now uses averaged softmax probabilities instead of hard argmax with fallback (Cell 34, `post_process`)
  - **Rationale:** The old logic did `argmax` &rarr; filter out class 0 &rarr; majority vote. When the NC head predicted background for all pixels inside a cell instance, the filter produced an empty array, and the fallback `else 1` silently assigned every instance to RBC. The new approach averages the softmax probability vectors across all pixels in each instance and picks the highest non-background class, so classification is never silently defaulted.

### Pre-Fix Results (Phase 2 LoRA, epoch 49)

These results were obtained **before** the classification fixes above. The model detects and segments cells well but classifies everything as RBC:

| Metric | RBC | WBC | Platelet | Mean |
|--------|-----|-----|----------|------|
| IoU | 0.9211 | 0.0000 | 0.0000 | 0.3070 |
| Dice | 0.9589 | 0.0000 | 0.0000 | 0.3196 |

| Detection (IoU=0.5) | Value |
|---------------------|-------|
| Precision | 0.8844 |
| Recall | 0.5849 |
| F1 | 0.7041 |

| Instance Segmentation | Value |
|----------------------|-------|
| mAP@0.50 | 0.4829 |
| mAP@[.50:.95] | 0.3871 |

| Panoptic Quality | Value |
|-----------------|-------|
| PQ | 0.7377 |
| SQ | 0.8927 |
| RQ | 0.8264 |

**Analysis:** Detection and segmentation quality are strong (PQ=0.74, SQ=0.89), confirming the NP and HV branches work well. The zero WBC/Platelet IoU is purely a classification failure in the NC branch, caused by the three issues fixed above.

---

## [2026-02-03] — Checkpoint Resume System

### Added

- `RESUME_TRAINING = True` parameter (Cell 2) — auto-resume from latest checkpoint when interrupted
- `save_latest_checkpoint()` helper with atomic write (`tmp` + `os.replace`) — survives mid-write process death (Cell 5)
- `load_latest_checkpoint()` helper — restores model, optimizer, scheduler, scaler, epoch counter, best_val_loss, patience_counter, and full training history (Cell 5)
- Resume detection in Phase 1 (Cell 30) and Phase 2 (Cell 32) training loops
- `latest_phase1.pth` and `latest_phase2.pth` saved every epoch, overwriting previous

### Rationale

University VPN enforces a 12-hour session timeout, dropping SSH connections mid-training. Previously, if training was at epoch 15/50 when the VPN dropped, all progress was lost because only "best" checkpoints were saved (no optimizer/scheduler state for resume). Now resubmitting `sbatch slurm_train.sh` automatically picks up where it left off.

---

## [2026-02-03] — Bug Fixes

### Fixed

- **Phase 2 cell silent skip** (Cell 30, 32): Added `elif` branch that prints a clear message when `PHASE` doesn't match, so users know to change the parameter instead of seeing no output
  - **Rationale:** With `PHASE=1` by default, the Phase 2 cell's `if MODE == 'train' and PHASE == 2:` guard evaluated to False, silently skipping the entire cell body. Jupyter showed an execution number with no error, making it appear as if nothing happened.

- **`CHECKPOINT_DIR` undefined** in evaluation cell (Cell 37): Renamed to `SAVE_DIR`
  - **Rationale:** Cell 2 defines `SAVE_DIR = "checkpoints"` but the evaluation cell referenced `CHECKPOINT_DIR`, which was never defined — a variable name mismatch causing `NameError`.

- **Evaluation `TypeError`** (Cell 37): Fixed batch unpacking from `batch['image']` to `(images, targets)` tuple
  - **Rationale:** `BCCDInstanceDataset.__getitem__` returns `(image_tensor, targets_dict)`. The default DataLoader collates this into a tuple, but the evaluation loop treated it as a dict. The training loops correctly used `for images, targets in loader`; the evaluation loop was inconsistent.

- **Phase 2 best checkpoint** (Cell 32): Added `optimizer_state_dict` to `best_phase2.pth` save
  - **Rationale:** Phase 2 was saving model weights and history but not the optimizer state, making it impossible to properly resume or analyze training dynamics from the best checkpoint.

---

## [2026-02-03] — Documentation

### Added

- **README: "Why Instance Segmentation with a Semantic Foundation Model?"** — introductory paragraph explaining why UNI's semantic attention features need an instance-aware decoder for cell counting and classification
- **README: "Why Two Phases?"** — rationale for gradual unfreezing (Phase 1 decoder-only, Phase 2 LoRA) to prevent overfitting on 364 images
- **README: "Multi-GPU Training & Performance"** — DDP speedup analysis with actual math (gradient sync cost, overhead ratio, realistic speedup estimates) and explanation of why DataLoader multiprocessing crashes in Jupyter but DDP works

---

## [2026-02-03] — DDP & Training Infrastructure

### Added

- **DistributedDataParallel (DDP)** support — optional, auto-detected from SLURM/torchrun environment variables, graceful fallback to single-GPU (Cell 5)
- **`torch.compile()`** support via `USE_COMPILE` flag (Cell 2, 24) — P100-safe with `mode='default'`
- **Configurable gradient accumulation** via `GRAD_ACCUM_TARGET` (Cell 2, 32) — effective batch size independent of GPU memory
- **Epoch timers** in Phase 1 and Phase 2 training loops — prints wall-clock time per epoch
- **`slurm_train_ddp.sh`** — multi-GPU SLURM job script with `srun` prefix and DDP env vars
- **`HPC_SETUP_GUIDE.md`** — step-by-step HPC deployment guide

### Fixed

- **`NUM_WORKERS`** `4` &rarr; `0` (Cell 2) — fixes `AssertionError: can only test a child process` crash caused by Python `fork()` duplicating Jupyter kernel state into DataLoader worker processes. SLURM papermill overrides to 8 via `-p NUM_WORKERS 8`.
- **SLURM log paths** — removed `logs/` prefix from `#SBATCH --output` directives because SLURM processes these before any script commands execute (so `mkdir -p logs` hasn't run yet)
- **Redundant HV map pre-computation** removed from `process_and_save` (Cell 14) — HV maps are always recomputed on-the-fly after augmentation in `BCCDInstanceDataset.__getitem__`, making the saved files dead weight
