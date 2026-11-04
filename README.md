# Cell Instance Segmentation with UNI Pathology Foundation Model

Binary cell instance segmentation of blood smear images (Kaggle [BCCD Dataset with mask](https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask/data)). Every pixel is classified as cell or background, and every cell, including cells that touch, becomes its own instance. The encoder is [UNI](https://huggingface.co/MahmoodLab/UNI), a ViT-L/16 pathology foundation model, adapted with a HoVer-Net-style decoder and LoRA.

On 159 test images the final model reaches detection F1 0.910 and panoptic quality (PQ) 0.823. See [RESULTS.md](RESULTS.md).

## Architecture

```
Image (224x224)
  -> UNI ViT-L/16 (frozen in Phase 1, + LoRA on QKV in Phase 2)
  -> features from blocks 6, 12, 18, 24
  -> DPT-style feature pyramid (14 -> 28 -> 56 -> 112 px)
  -> NP head: cell probability (1 channel)
  -> HV head: horizontal / vertical distance to the cell centroid (2 channels)
  -> HoVer-Net marker-controlled watershed -> one label per cell
```

### Design choices

- **UNI backbone.** Pretrained with DINOv2-style self-supervision on more than 100M patches from more than 100,000 H&E slides. It is loaded strictly; if the weights cannot be downloaded the run stops (no ImageNet fallback).
- **NP + HV heads.** The masks are binary, so there is no cell-type head. The HV distance maps make the boundary between two touching cells explicit, which the watershed uses to split them.
- **Reference HoVer-Net post-processing.** A faithful port of HoVer-Net's watershed. The notebook's original version merged touching cells and is still available with `evaluate.py --postproc notebook`.

### Loss functions

- NP: binary cross-entropy + soft Dice (weight 1.0)
- HV: MSE + mean squared gradient error, on cell pixels only (weight 2.0)

## Setup

### Prerequisites

- An NVIDIA GPU (tested on one RTX 5000 Ada, 32 GB) and Python 3.11
- Approved access to [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) on Hugging Face

### Installation (Windows)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\hf.exe auth login                      # Hugging Face read token
.venv\Scripts\python.exe scripts\check_uni_access.py # optional: verify UNI weights
```

On Linux: `conda env create -f environment.yml && conda activate uni_seg && hf auth login`.

### Data preparation

Download the BCCD dataset with masks and place the processed data in `data/processed/` (not tracked in git):

```
data/processed/{train,val,test}/images/*.png        256x256 RGB
data/processed/{train,val,test}/instance_maps/*.npy uint16 instance ids
```

### Data split

| Split | Images | Source |
|---|--:|---|
| Train | 993 | Kaggle train partition (85%) |
| Val | 176 | Kaggle train partition (15%, `random_state=42`) |
| Test | 159 | Kaggle test partition |

## Usage

### Local (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_local_ddp.ps1 -NProc 1
```

This runs the UNI check, Phase 1, Phase 2 and the test-set evaluations, and resumes from the latest checkpoint if re-run. `-SkipPhase1` continues an interrupted Phase 2. `-Backbone imagenet -RunName ablation_imagenet` trains the ImageNet ViT-L/16 baseline.

### HPC (SLURM)

```bash
sbatch slurm_train.sh        # 1 GPU: both phases + evaluation
sbatch slurm_train_ddp.sh    # all GPUs on one node (DDP, NCCL)
sbatch slurm_eval.sh [checkpoint] [tag]
```

See [HPC_SETUP_GUIDE.md](HPC_SETUP_GUIDE.md).

## Training procedure

### Why two phases?

Phase 1 trains only the decoder, so its random initial gradients cannot disturb the pretrained features. Phase 2 then adapts UNI with low-rank adapters (LoRA). The adapters start as an exact no-op (B = 0), so Phase 2 begins from the Phase 1 model.

| | Phase 1: decoder only | Phase 2: LoRA |
|---|---|---|
| Trainable parameters | 11.8M (UNI frozen) | 12.2M (incl. 393K LoRA, rank 4 on QKV of all 24 blocks) |
| Optimizer | AdamW, lr 1e-3 | AdamW, LoRA lr 1e-5, decoder lr 1e-4 |
| Schedule | 5-epoch warmup, cosine with warm restarts | cosine |
| Max epochs / patience | 100 / 15 | 50 / 15 |
| Global batch | 16 | 16 (8 x 2 accumulation) |

### Augmentation

Random crop, flips, 90-degree rotations, shift-scale-rotate, elastic/grid distortion, color jitter, blur, noise and coarse dropout. HV targets are recomputed after every geometric transform.

### Distributed training

Training runs under `torchrun` / DistributedDataParallel: NCCL on Linux with one GPU per process, gloo otherwise. The global batch stays 16 for any number of processes. The reported run used one process on one GPU.

## Results

| Model | F1 | PQ | Foreground Dice |
|---|--:|--:|--:|
| UNI Phase 1 (frozen) | 0.909 | 0.818 | 0.957 |
| **UNI Phase 2 (LoRA)** | **0.910** | **0.823** | **0.959** |
| ImageNet ViT-L/16 Phase 2 (same code) | 0.907 | 0.818 | 0.958 |

Full metrics and sample segmentations: [RESULTS.md](RESULTS.md).

## Challenges and potential improvements

- **Touching cells.** The notebook's watershed merged neighbouring cells (PQ 0.634). The reference HoVer-Net post-processing fixed this (PQ 0.823) for the same model.
- **Small backbone gain.** UNI was pretrained on H&E tissue, while these are blood smears. A fairer ablation would use several seeds and each backbone's own input normalisation.
- **Recall.** The model misses more cells (recall 0.860) than it adds false ones (precision 0.967).
- **Cell types.** The masks have one foreground class, so cell typing (RBC / WBC / platelet) needs new labels.

## Project structure

```
train.py, evaluate.py         training (Phase 1 / 2) and test-set evaluation
uniseg/                       model, LoRA, losses, data, training loop, DDP, post-processing, metrics, plots
scripts/                      Windows and Linux pipelines, UNI access check, run comparison
slurm_*.sh                    SLURM jobs
results/                      metrics, curves and sample figures
requirements.txt, environment.yml
```

## References

- Chen et al. Towards a general-purpose foundation model for computational pathology (UNI). Nature Medicine, 2024.
- Graham et al. HoVer-Net: simultaneous segmentation and classification of nuclei. Medical Image Analysis, 2019.
- Ranftl et al. Vision Transformers for Dense Prediction (DPT). ICCV, 2021.
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.

## License

UNI weights are licensed CC-BY-NC-ND 4.0 (non-commercial academic use). Checkpoints written by this code contain only the decoder and LoRA weights, never UNI weights.
