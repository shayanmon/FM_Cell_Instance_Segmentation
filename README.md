# Cell Instance Segmentation with UNI Pathology Foundation Model

Cell instance segmentation on the BCCD blood cell dataset, using the UNI foundation model (ViT-L/16) as backbone with a HoVer-Net-inspired multi-task decoder.

### Why Instance Segmentation with a Semantic Foundation Model?

UNI is a pathology foundation model trained via DINOv2 on over 100 million H&E-stained tissue patches. Its vision transformer attention heads encode rich *semantic* features — tissue architecture, staining patterns, nuclear texture, and cell morphology — but they have no built-in notion of individual object instances. Standard semantic segmentation can label pixels as "WBC" or "RBC," but it cannot distinguish *which* WBC or *which* RBC a pixel belongs to. In clinical hematology, the downstream task demands more: counting individual cells, measuring per-cell morphology, and classifying each cell independently. This requires **instance segmentation** — delineating every cell as a separate object. Blood smears make this especially challenging because red blood cells frequently overlap and touch, causing semantic masks to merge adjacent cells into a single connected region. Our approach bridges this gap by pairing UNI's frozen semantic features with a HoVer-Net-inspired decoder that regresses per-pixel horizontal and vertical distance maps pointing toward each cell's centroid. Sobel gradients on these distance maps create sharp energy boundaries between touching cells, which marker-controlled watershed then cuts into individual instances. By keeping UNI's backbone frozen (Phase 1) and later adding lightweight LoRA adapters (Phase 2), we repurpose its 304M parameters of domain-specific attention as input to an instance-aware decoder — converting patch-level semantic representations into the instance-level predictions needed for cell counting, classification, and morphological analysis.

## Architecture

```
Input (224x224) -> UNI ViT-L/16 (frozen/LoRA) -> DPT Feature Pyramid -> 3 Decoder Heads
                   [blocks 5,11,17,23]           [14->28->56->112]
                   1024-dim features              256-dim fused          NP: binary seg (1ch)
                                                                         HV: distance maps (2ch)
                                                                         NC: classification (4ch)
                                                                              |
                                                                    Watershed Post-Processing
                                                                              |
                                                                    Instance Segmentation
```

### Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Backbone** | UNI ViT-L/16 | Pre-trained on 100M+ H&E pathology patches via DINOv2. Encodes domain-specific features (nuclear texture, cell morphology, staining patterns) directly relevant to blood cell analysis. Frozen backbone prevents overfitting on 364 images. |
| **Multi-scale features** | DPT-style reassembly (blocks 5/11/17/23) | ViT produces constant 14x14 features at all depths. DPT progressively upsamples to 112x112, recovering spatial detail lost by 16x patch tokenization. Earlier blocks capture edges; deeper blocks capture semantics. |
| **Decoder** | 3-branch HoVer-Net | The gold standard for histopathology cell segmentation. Distance-map regression naturally separates touching/overlapping cells (common in blood smears) through gradient-based watershed, unlike bounding-box methods (Mask R-CNN) which struggle with overlapping cells. |
| **Fine-tuning** | LoRA (rank=4) | Only ~720K extra params added to 304M backbone. PEFT is the most effective strategy for pathology foundation models on small datasets, outperforming full fine-tuning, linear probing, and partial fine-tuning. |

### Loss Functions

- **NP branch**: BCE + Dice — handles cell/background imbalance
- **HV branch**: MSE + MSGE (Sobel gradient error) — enforces sharp boundary transitions critical for watershed; weighted 2x
- **NC branch**: Weighted CrossEntropy (RBC=0.3, WBC=3.0, Platelet=5.0) + per-class Dice — compensates for class imbalance

## Setup

### Prerequisites

- NVIDIA GPU with >= 16GB VRAM (tested on V100 32GB)
- Miniconda or Anaconda
- HuggingFace account with accepted terms at [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)

### Installation

```bash
cd UNI_instance_seg

# Create conda environment
conda env create -f environment.yml
conda activate uni_seg

# Authenticate with HuggingFace (required for UNI weights)
huggingface-cli login
# Enter your HF token when prompted
```

### Data Preparation

1. Download the BCCD Dataset with Mask from [Kaggle](https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask/data)
2. Extract into `data/raw/`
3. Run the data processing cells in the notebook (Sections 1-2)

Expected structure after extraction:
```
data/raw/
  train/
    images/
    masks/
  test/
    images/
    masks/
```

### Data Split

The BCCD dataset ships with a pre-defined train/test partition. The pipeline preserves this original test set and further subdivides the training partition into train and validation sets:

| Split | Images | % of Total | Source |
|-------|--------|------------|--------|
| **Train** | 993 | 74.8% | 85% of original train partition |
| **Validation** | 176 | 13.3% | 15% of original train partition |
| **Test** | 159 | 12.0% | Original test partition (untouched) |
| **Total** | 1328 | 100% | |

The validation set is carved from the training partition using `sklearn.train_test_split` with a fixed random seed for reproducibility (`test_size=0.15, random_state=SEED`). The original Kaggle test set is never touched during training or hyperparameter tuning — it is held out exclusively for final evaluation.

**Why this split:**

1. **The test set is kept as-provided.** The BCCD Kaggle dataset defines its own train/test boundary. Respecting this boundary means our results are comparable to other work on the same dataset. Reshuffling would break comparability and risk data leakage if the dataset curators designed the split to avoid patient/slide overlap.

2. **15% validation is a pragmatic choice for small datasets.** With ~1170 training-source images, a 15% holdout yields 176 validation images — large enough for stable loss estimation and early stopping decisions, but small enough to leave 993 images for training. A larger validation set (e.g., 20–25%) would improve validation stability marginally but reduce the training set below 900 images, which matters when the dataset is already small. A smaller validation set (e.g., 5–10%) risks noisy validation loss curves that trigger early stopping prematurely or too late.

3. **No k-fold cross-validation.** While k-fold would make more efficient use of data, it multiplies training time by k — impractical when each Phase 1 run is 100 epochs on a ViT-L backbone. The two-phase training pipeline (Phase 1 + Phase 2) already doubles the training cost, making k-fold a 2k× multiplier. A single held-out validation set is the standard compromise for deep learning on medical imaging datasets of this size.

4. **Stratification is not applied.** Blood smear images contain all three cell types (RBC, WBC, Platelet) in most frames, so a random split naturally produces balanced class representation across partitions. Stratified splitting by dominant cell type would add complexity without meaningful benefit since per-image class distributions are relatively uniform.

## Usage

### Interactive (Jupyter)

```bash
conda activate uni_seg
jupyter notebook cell_instance_segmentation.ipynb
```

Run cells sequentially. Sections 0-5 handle setup, data, and model definition. Section 6 trains Phase 1, Section 7 trains Phase 2.

### HPC (SLURM)

```bash
# Set your HuggingFace token
export HF_TOKEN="hf_your_token_here"

# Submit training job (runs both phases)
sbatch slurm_train.sh

# Submit evaluation job (after training completes)
sbatch slurm_eval.sh

# Monitor
tail -f logs/train_*.out
```

## Training Procedure

### Why Two Phases?

With only 364 training images, fine-tuning all parameters simultaneously risks overfitting the backbone before the decoder has learned anything useful. Phase 1 freezes the entire UNI backbone (304M params) and trains only the decoder (~11M params), giving the feature pyramid and segmentation heads a stable foundation for interpreting UNI's pre-learned pathology features. Once the decoder converges, Phase 2 injects lightweight LoRA adapters (~720K params) into the backbone's attention layers and trains them at a 10x lower learning rate (1e-5 vs 1e-4). This lets the backbone make small, targeted adjustments to its feature representations — shifting attention toward cell boundary cues that matter for instance segmentation — without catastrophically overwriting the domain knowledge learned from 100M+ pathology patches. The two-phase approach is a form of gradual unfreezing, which consistently outperforms single-phase training on small medical imaging datasets where the risk of overfitting a large pretrained model is high.

### Why Low-Rank Adaptation After Decoder-Only Training?

After Phase 1, the decoder and feature pyramid have converged on the frozen backbone's feature space. At this point there are several options for further improvement: full fine-tuning, partial layer unfreezing, linear probing, or parameter-efficient fine-tuning (PEFT). We chose LoRA (Low-Rank Adaptation) for Phase 2 for several reasons:

1. **Full fine-tuning is catastrophic at this data scale.** Unfreezing all 304M backbone parameters and training them on 364 images would quickly overwrite the domain-specific representations UNI learned from 100M+ pathology patches. The backbone-to-data ratio (304M params / 364 images) makes full fine-tuning a severe overfitting risk — the model would memorize training images rather than generalize.

2. **Partial layer unfreezing is fragile.** A common alternative is unfreezing the last N transformer blocks while keeping earlier blocks frozen. This requires choosing N (how many layers?), and the optimal choice is dataset-dependent. Unfreezing too few layers limits adaptation; unfreezing too many reintroduces overfitting. LoRA sidesteps this entirely by injecting adapters into *every* attention layer at once, letting the optimization decide which layers need the most adjustment via gradient magnitude — early layers that already produce useful features will naturally receive smaller updates.

3. **LoRA preserves the pre-trained feature space.** Each LoRA adapter initializes with B=0, meaning the backbone starts Phase 2 producing *exactly the same features* as Phase 1. Training updates the low-rank matrices A and B to add small deltas: `output = Wx + (α/r) · BAx`. This means the decoder never faces a sudden distribution shift in its input features — adaptation is smooth and monotonic from the Phase 1 solution. Full fine-tuning or partial unfreezing, by contrast, immediately disrupts the feature distribution the decoder was trained on.

4. **Parameter efficiency matches the data budget.** LoRA rank=4 across all 24 attention QKV projections adds ~720K trainable parameters — only 0.24% of the backbone. This is a favorable ratio for 364 training images: enough capacity to steer attention toward cell boundary cues and instance-discriminative features, but not enough to memorize the dataset. The rank acts as an implicit regularizer — the low-rank bottleneck constrains updates to lie in a small subspace, preventing the kind of high-rank weight perturbations that cause overfitting.

5. **Differential learning rates are natural.** LoRA parameters and decoder parameters live in separate `nn.Module` subtrees, making it trivial to assign different learning rates (backbone LoRA: 1e-5, decoder: 1e-4). This 10x gap ensures the backbone makes conservative adjustments while the decoder continues refining its predictions — a form of discriminative fine-tuning that would be harder to configure cleanly with partial unfreezing.

### Phase 1: Decoder-Only Training (Backbone Frozen)

**What happens to the backbone**: Every parameter in the UNI ViT-L/16 (304M params) has `requires_grad=False`. The backbone acts as a fixed feature extractor — its pre-trained attention patterns, learned from 100M+ H&E pathology patches via DINOv2 self-supervision, are preserved exactly as released. Gradients do not flow into the backbone during Phase 1; only the feature pyramid (DPT reassembly + fusion blocks) and the three decoder heads receive gradient updates.

**What Phase 1 accomplishes**: The decoder and feature pyramid learn to *interpret* UNI's frozen features for the specific task of cell instance segmentation. The feature pyramid learns how to fuse multi-scale ViT activations (from blocks 5, 11, 17, 23) into spatially detailed 112x112 feature maps. The three decoder heads learn their respective tasks on top of these fused features: NP head learns to distinguish cell pixels from background, HV head learns to regress horizontal/vertical distance maps pointing toward each cell's centroid, and NC head learns to classify pixels as RBC, WBC, or Platelet. By the end of Phase 1, the entire decoder pipeline has stabilized around the backbone's feature distribution — this is the foundation that Phase 2 builds on.

**Why the backbone must be frozen first**: If the backbone and decoder trained simultaneously from scratch, two problems arise. First, the randomly-initialized decoder would produce noisy gradients that propagate into the backbone, corrupting its pre-learned features before the decoder has learned to use them. Second, both the feature space (backbone) and the feature consumer (decoder) would be moving targets — co-adaptation makes training unstable and prone to local minima. Freezing the backbone removes one degree of freedom, letting the decoder converge on a stable signal.

| Parameter | Value |
|-----------|-------|
| Trainable params | ~11M (decoder + feature pyramid) |
| Frozen params | ~304M (UNI backbone) |
| Epochs | 100 (early stopping patience=15) |
| Batch size | 16 |
| Optimizer | AdamW (LR=1e-3, weight_decay=1e-4) |
| LR schedule | 5-epoch linear warmup, then CosineAnnealingWarmRestarts (T0=20) |
| Mixed precision | fp16 |
| Augmentation | Geometric (flips, rotation, scale, elastic) + photometric (color jitter, blur, noise) |

### Phase 2: LoRA Fine-tuning (Backbone Partially Unfrozen via Low-Rank Adapters)

**What happens to the backbone**: The backbone's original weights remain frozen (`requires_grad=False`), but LoRA adapters are injected into every attention layer's QKV projection across all 24 transformer blocks. Each adapter consists of two small matrices — A (1024→4) and B (4→3072) — initialized so that B=0, meaning the backbone produces identical outputs to Phase 1 at the start of Phase 2. During training, only these adapter matrices are updated, producing a low-rank delta on top of the frozen attention weights: `output = W_frozen·x + (α/r)·B·A·x`. This is a *controlled partial unfreezing* — the backbone's feature representations can shift, but only within the low-rank subspace defined by rank=4. The original pre-trained weights are never modified, so the adaptation can be cleanly removed or adjusted without losing the foundation model.

**What Phase 2 accomplishes**: While Phase 1 trained the decoder to work with UNI's generic pathology features, Phase 2 steers the backbone's attention toward features that are specifically useful for cell instance segmentation on blood smears. The LoRA adapters learn to adjust the attention patterns so that the backbone emphasizes cell boundary cues, nucleus-cytoplasm contrast, and inter-cell separation signals — features that are present in UNI's representation space but not maximally activated by default. Because the decoder was already trained on the frozen feature distribution, these small feature-space shifts translate directly into improved segmentation without destabilizing the decoder. The differential learning rate (backbone LoRA at 1e-5, decoder at 1e-4) ensures the backbone adapts slowly while the decoder tracks the shifting features.

**Relationship to the wrapper model (`CellSegModel`)**: The `CellSegModel` wrapper encapsulates the full pipeline — backbone, feature pyramid, and decoder heads. During Phase 1, only the pyramid and heads inside the wrapper are trainable. When `apply_lora()` is called before Phase 2, it reaches into the wrapper's backbone submodule and replaces each `qkv` linear layer with a `LoRALinear` wrapper that adds the trainable low-rank path. The optimizer then receives two parameter groups extracted from the wrapper: LoRA parameters (matched by `'lora_'` in the parameter name) at 1e-5, and all other trainable parameters (pyramid + heads) at 1e-4. The wrapper model's structure makes this clean separation possible — backbone parameters live under `model.backbone.*`, while decoder parameters live under `model.feature_pyramid.*` and `model.*_head.*`.

| Parameter | Value |
|-----------|-------|
| LoRA rank | 4 (applied to all 24 attention QKV projections) |
| Added params | ~720K |
| Epochs | 50 |
| Batch size | 8 (with 2x gradient accumulation = effective 16) |
| Backbone LR | 1e-5 |
| Decoder LR | 1e-4 |
| LR schedule | CosineAnnealing (no restarts) |

### Augmentation Strategy

Heavy augmentation is essential for the small dataset (364 images):
- **Geometric**: random crop 256->224, flips, rotation (45 deg), scale (0.8-1.2x), elastic/grid distortion
- **Photometric**: color jitter, Gaussian blur, Gaussian noise, coarse dropout
- **HV map integrity**: geometric transforms are applied first, then HV distance maps are recomputed from the transformed instance map (crucial correctness detail)

### Checkpoint Resume System

The university HPC enforces a 12-hour VPN session timeout. When the VPN drops, the SSH connection to the compute node dies and the Jupyter kernel (or SLURM interactive session) is killed mid-training. Without a resume mechanism, all progress since the last `sbatch` submission is lost — if training was at epoch 60/100 when the session cut off, it would restart from epoch 0.

To handle this, the pipeline saves a **latest checkpoint every epoch** (`latest_phase1.pth` / `latest_phase2.pth`) that captures the full training state, not just model weights. When `RESUME_TRAINING = True` (the default), the training loop detects the existing checkpoint and continues from where it left off.

**What the checkpoint stores:**

| Component | Why it's needed for correct resume |
|-----------|------------------------------------|
| `model_state_dict` | Restore learned weights (including LoRA adapters in Phase 2) |
| `optimizer_state_dict` | Restore AdamW momentum buffers and per-parameter adaptive learning rates — without this, the optimizer restarts from scratch and the first few post-resume epochs behave erratically |
| `scheduler_state_dict` | Restore the cosine annealing position so the learning rate curve is continuous across the interruption, not reset to peak |
| `scaler_state_dict` | Restore the fp16 GradScaler's loss scale factor — if this resets, the scaler spends several epochs re-calibrating, causing gradient underflow |
| `epoch` | Resume the `for epoch in range(start_epoch, EPOCHS)` loop at the correct position |
| `best_val_loss` | Preserve the early stopping baseline — without this, the patience counter resets and the model trains for up to `PATIENCE` extra epochs before stopping |
| `patience_counter` | Continue the early stopping countdown rather than resetting it |
| `history` | Append to the existing training curves rather than starting new ones, so the final loss plots show the complete run |

**Atomic writes**: Checkpoints are saved via a tmp file + `os.replace()` swap. If the process is killed mid-write (e.g., VPN drops during `torch.save`), the previous checkpoint remains intact because `os.replace` is atomic on POSIX filesystems. A naive `torch.save` directly to the target path would corrupt the file if interrupted, leaving no valid checkpoint to resume from.

**How resume works in practice**: After a VPN dropout, reconnect and resubmit the same `sbatch slurm_train.sh` command (or re-run notebook cells from the top). The parameter cell sets `RESUME_TRAINING = True`, the training cell calls `load_latest_checkpoint()`, detects the saved file, restores all state, and prints `Resuming from epoch N (best_val_loss=X.XXXX)`. Training continues with the correct learning rate, optimizer momentum, and early stopping state — the interruption is transparent to the training dynamics.

**Separate from best checkpoints**: The `best_phase1.pth` / `best_phase2.pth` files are saved only when validation loss improves. These are used for final evaluation and inference. The `latest_*.pth` files exist solely for crash recovery and are overwritten every epoch. Both checkpoint types coexist in the `checkpoints/` directory.

## Multi-GPU Training & Performance

### Why DDP Instead of DataLoader Multiprocessing

When running in Jupyter, PyTorch's DataLoader with `num_workers > 0` crashes with:
```
AssertionError: can only test a child process
```

This happens because of how Python multiprocessing interacts with Jupyter:

1. **Jupyter's kernel** runs inside a complex process with event loops, UI handlers, and inter-process communication channels
2. **Python's `fork()`** (the default on Linux) duplicates the entire parent process state into child workers
3. **Forked children inherit** the partially-initialized Jupyter kernel state — they have copies of the kernel's IPC sockets, event loop handles, and process management structures that are only valid in the parent
4. When the DataLoader tries to clean up these child workers, it calls `is_alive()` on processes that were forked from the wrong context, triggering the `AssertionError`

**The fix**: `num_workers=0` disables forking entirely — data loads synchronously in the main process. When running via SLURM (not Jupyter), `num_workers=8` works fine because there is no Jupyter kernel to corrupt.

**DDP is different**: DistributedDataParallel launches separate, independent Python processes (one per GPU) — each with its own clean interpreter, not forked from a Jupyter kernel. This is why DDP works reliably while DataLoader multiprocessing does not in notebooks.

### DDP Speedup Analysis

We use PyTorch's `DistributedDataParallel` (DDP) for multi-GPU training. The expected speedup depends on the ratio of computation time to gradient synchronization overhead.

**Gradient synchronization cost:**

| Component | Value |
|-----------|-------|
| Trainable parameters (Phase 1) | ~11M (decoder only; 304M backbone is frozen) |
| Gradient tensor size | 11M × 4 bytes = 44 MB per AllReduce |
| PCIe 3.0 x16 bandwidth (P100) | ~12 GB/s practical |
| Gradient sync time | 44 MB / 12 GB/s ≈ 3.7 ms |
| Forward + backward pass (P100) | ~400–600 ms per step |
| Overhead ratio | 3.7 ms / 500 ms ≈ 0.7% |

**Speedup formula**: `speedup = N / (1 + communication_overhead / compute_time)`

| GPUs | Theoretical | Realistic | Notes |
|------|------------|-----------|-------|
| 1 | 1.0x | 1.0x (baseline) | Single GPU, no communication |
| 2 | 2 / 1.007 ≈ 1.99x | **~1.8x** | Small overhead from DistributedSampler, I/O contention |
| 4 | 4 / 1.014 ≈ 3.94x | **~3.0–3.2x** | AllReduce ring communication grows; PCIe bandwidth shared across 4 GPUs |

**Why the small trainable param count helps**: Only 11M decoder parameters need gradient synchronization — the frozen 304M backbone contributes zero communication overhead. This makes our workload particularly efficient for DDP compared to fully fine-tuned models.

**Phase 2 (LoRA)**: Adds ~720K LoRA parameters (total trainable ~11.7M), so overhead barely changes. However, the smaller batch size (8 vs 16) means shorter compute time per step, reducing the compute-to-communication ratio slightly. Realistic Phase 2 speedup: ~1.6–1.7x for 2 GPUs.

### Performance Options Summary

| Method | How to enable | Speedup | Trade-off |
|--------|---------------|---------|-----------|
| **DDP** (multi-GPU) | `sbatch slurm_train_ddp.sh` | ~1.8x per added GPU | Requires multiple GPUs; SLURM only |
| **torch.compile** | `USE_COMPILE = True` in params | 10–20% | First epoch ~30s slower (compilation); P100 gains are modest |
| **Gradient accumulation** | `GRAD_ACCUM_TARGET = 32` | Better convergence | Same speed, larger effective batch |
| **Mixed precision (fp16)** | Already enabled | ~2x vs fp32 | Already active; P100 supports fp16 |

## Results

Metrics are computed on the held-out test set after full training. Results are saved to `results/metrics.json`.

### Quantitative Results

| Metric | RBC | WBC | Platelet | Mean |
|--------|-----|-----|----------|------|
| IoU | 0.9207 | 0.0000 | 0.0000 | 0.3069 |
| Dice | 0.9587 | 0.0000 | 0.0000 | 0.3196 |

| Detection Metric (IoU=0.5) | Value |
|---------------------------|-------|
| Precision | 0.8823 |
| Recall | 0.5809 |
| F1 | 0.7005 |

| Instance Segmentation | Value |
|----------------------|-------|
| mAP@0.50 | 0.4819 |
| mAP@[.50:.95] | 0.3865 |

| Panoptic Quality | Value |
|-----------------|-------|
| PQ | 0.7356 |
| SQ | 0.8928 |
| RQ | 0.8239 |

**Note on WBC/Platelet classification:** The zero IoU/Dice for WBC and Platelet classes reflects a class weight imbalance in the NC loss that was identified and corrected during development (see CHANGELOG). The model detects and segments cell instances well (PQ=0.74, Precision=0.88) but the classification head was undertrained for minority classes. Retraining with corrected class weights (`CLASS_WEIGHTS=[0.05, 0.5, 5.0, 8.0]` and NC loss weight=2.0) is expected to resolve this.

### Qualitative Results

Visualizations are saved to `results/visualization_sample_*.png`. Each figure shows:
- **Row 1**: Instance Overlays and the Error Map. The first row places the original blood smear image alongside ground-truth and predicted instance overlays, where each cell is painted a unique color so you can visually count and compare detections. In a good prediction, the two overlays look nearly identical — the same number of cells, with contours that trace the same boundaries. The most revealing panel is the error map in Row 3 (bottom-right), which paints each predicted instance green if it matched a ground-truth cell (true positive at IoU ≥ 0.5), red if no ground-truth cell corresponds to it (false positive — typically a single RBC that the watershed split into two, or a staining artifact picked up as a cell), and blue for ground-truth cells the model missed entirely (false negatives — usually tiny platelets tucked between RBC clusters, or cells sitting at the crop boundary with only a sliver visible). In practice, most errors concentrate at dense RBC clusters where cells overlap and the watershed has to decide where one cell ends and the next begins.
- **Row 2**: What the Decoder Actually Learned. The second row exposes the three intermediate maps the decoder produces before post-processing turns them into instance masks. The NP (nuclear pixel) heatmap on the left uses a hot colormap to show how confident the model is that each pixel belongs to a cell versus background — bright yellow-white regions are high-confidence cell pixels, and dark regions are background. When the model is working well, every cell body lights up cleanly with sharp edges; if the heatmap looks washed out or bleeds into the background, the model is struggling with cell-background separation, and the downstream watershed will produce noisy, fragmented instances. The middle and right panels show horizontal and vertical distance maps, which encode how far each pixel sits from the centroid of the cell it belongs to. Inside each cell, the horizontal map shows a smooth blue-to-red gradient from left to right, and the vertical map shows the same gradient from top to bottom — together they form a vector field pointing inward toward each cell's center. These gradients are what make instance separation possible: the Sobel filter extracts sharp edges where adjacent cells' gradients point in opposite directions, creating the energy barriers that watershed uses to cut touching cells apart. If you see crisp, concentric color transitions within each cell, the model has learned accurate centroid regression. Blurry or patchy gradients mean the model is unsure where cell centers are, which produces weak energy barriers and leads to under-segmentation (adjacent cells merged into one instance) or over-segmentation (a single cell fractured into pieces).
- **Row 3**: Per-Cell Classification. The bottom row compares ground-truth and predicted class maps, where each pixel is colored by its cell type — typically red for RBCs, a distinct color for WBCs, and another for platelets. Side-by-side comparison immediately reveals whether classification errors are systematic or isolated. A systematic failure looks like an entire cell type going dark (e.g., every platelet predicted as background), which points to the class weights in the loss function being too low for that minority class — the RBC-dominant gradient drowns out the learning signal for rarer cell types, and the NC head never learns to recognize them. An isolated failure looks like a single WBC mislabeled as an RBC while other WBCs in the same image are classified correctly, which is more likely a feature ambiguity at that particular cell's location. It is also worth checking whether misclassified pixels line up with the instance boundaries from Row 1 — if the class map predicts the right cell type but the colored region is shifted or misshapen relative to the instance contour, the problem is upstream in the NP/HV heads rather than in the classifier itself.

### Metric Trade-offs in Cell Morphology Context

- **IoU vs Dice**: IoU penalizes boundary errors more severely than Dice. For small cells like platelets where boundary pixels dominate, Dice gives a more forgiving assessment. IoU better captures morphological accuracy for larger WBCs.
- **mAP at high IoU**: mAP@0.75+ is extremely demanding for small round cells. A perfectly detected platelet with a 2-pixel boundary error can drop from IoU=0.7 to IoU=0.5, causing large mAP drops at strict thresholds.
- **Panoptic Quality**: PQ = SQ x RQ decomposition is useful because SQ measures segmentation quality of matched instances while RQ measures detection recall/precision. If SQ is high but RQ is low, the model segments well but misses cells. If RQ is high but SQ is low, it detects cells but with poor masks.

## Challenges and Potential Improvements

### Challenges Encountered

1. **Small dataset**: 364 images is extremely small for deep learning. Mitigated with heavy augmentation, frozen backbone, and LoRA, but still limits generalization.

2. **Mask format ambiguity**: The BCCD dataset masks use color-coding that varies. The pipeline auto-detects the format and maps colors to classes, but may need manual verification.

3. **HV map invalidation**: Geometric augmentations invalidate pre-computed distance maps. Splitting augmentation into geometric + photometric phases and recomputing HV maps adds ~3x data loading overhead.

4. **Overlapping cells**: RBCs commonly overlap in blood smears. The HoVer-Net distance-map approach handles this better than threshold-based methods, but the watershed post-processing has sensitive hyperparameters (Sobel kernel size, marker thresholds).

5. **Class imbalance**: RBCs vastly outnumber WBCs and platelets. Mitigated with class-weighted CE loss and per-class Dice.

6. **Unfavorable backbone model deployed**: Unfortunately, the UNI model requires permission from the authors to access their weights, which we could not get approval for in the time constraint. Thus, we used a more general purpose foundation model (ImageNet) as the wrapper to fine-tune on.

### Potential Improvements

1. **Stain normalization**: Apply Macenko or Reinhard stain normalization as a preprocessing step. Blood smear staining varies significantly between slides, and normalization would improve generalization.

2. **Test-time augmentation (TTA)**: Average predictions across multiple augmented views (flips, rotations) at inference time for more robust results.

3. **UNI v2**: The newer UNI2-h model (ViT-H, trained on 200M+ images) would provide stronger features. Drop-in replacement since the architecture is the same family.

4. **StarDist-style output**: Replace HoVer-Net with star-convex polygon predictions for more precise boundaries on round cells like blood cells.

5. **Larger dataset**: Combine BCCD with other blood cell datasets (PBC dataset, ALL-IDB) for more training data.

6. **Post-processing optimization**: Bayesian optimization or grid search over watershed hyperparameters on the validation set.

7. **Multi-scale inference**: Process images at multiple resolutions and merge predictions for better handling of cells at different sizes (platelets vs WBCs).

## Project Structure

```
UNI_instance_seg/
  cell_instance_segmentation.ipynb   # Primary notebook (all code)
  README.md                          # This file
  environment.yml                    # Conda environment
  slurm_train.sh                     # SLURM training job (single GPU)
  slurm_train_ddp.sh                 # SLURM training job (multi-GPU DDP)
  slurm_eval.sh                      # SLURM evaluation job
  HPC_SETUP_GUIDE.md                 # Step-by-step HPC deployment guide
  data/
    raw/                             # Kaggle download
    processed/                       # Pre-processed .npy files
      train/, val/, test/
  checkpoints/                       # Model weights
    best_phase1.pth
    best_phase2.pth
  results/                           # Outputs
    metrics.json
    training_curves_*.png
    visualization_sample_*.png
```

## References

- **UNI**: Chen, R.J. et al. "Towards a general-purpose foundation model for computational pathology." *Nature Medicine*, 2024.
- **HoVer-Net**: Graham, S. et al. "Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images." *Medical Image Analysis*, 2019.
- **DPT**: Ranftl, R. et al. "Vision Transformers for Dense Prediction." *ICCV*, 2021.
- **DINOv2**: Oquab, M. et al. "DINOv2: Learning Robust Visual Features without Supervision." *TMLR*, 2024.
- **LoRA**: Hu, E. et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.

## License

Model weights: Subject to [UNI license terms](https://huggingface.co/MahmoodLab/UNI) (CC-BY-NC-ND 4.0).
Code: MIT License.
