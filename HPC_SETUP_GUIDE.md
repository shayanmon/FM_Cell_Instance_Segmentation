# HPC Setup & Run Guide

How to train and evaluate on a SLURM cluster. `<user>` and `<hpc_login_address>` are placeholders; adjust the `#SBATCH` partition and GPU lines in `slurm_*.sh` for your cluster.

## 1. Get the project onto the cluster

```bash
ssh <user>@<hpc_login_address>
git clone https://github.com/shayanmon/FM_Cell_Instance_Segmentation.git
```

The data is not in git. Copy `data/processed/` from your machine:

```bash
scp -r data/processed <user>@<hpc_login_address>:FM_Cell_Instance_Segmentation/data/
```

If a shell script fails with `$'\r': command not found`, convert it to LF: `sed -i 's/\r$//' slurm_*.sh scripts/train_ddp.sh`.

## 2. Create the conda environment

```bash
cd ~/FM_Cell_Instance_Segmentation
conda env create -f environment.yml
conda activate uni_seg
```

If `conda` is not found: `eval "$(~/miniconda3/bin/conda shell.bash hook)"`.

## 3. Log in to Hugging Face

Request access at https://huggingface.co/MahmoodLab/UNI, create a read token, then:

```bash
hf auth login
```

## 4. Check the GPU and UNI access on a compute node

```bash
srun --partition=gpu --gres=gpu:1 --time=00:15:00 --pty bash
conda activate uni_seg
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/check_uni_access.py
exit
```

## 5. Submit training

```bash
sbatch slurm_train.sh        # 1 GPU: UNI check, Phase 1, Phase 2, evaluation
sbatch slurm_train_ddp.sh    # all GPUs on one node (2 requested), NCCL
```

Options go on the command line, for example:

```bash
PHASE2_EPOCHS=30 sbatch slurm_train.sh
BACKBONE=imagenet RUN_NAME=ablation_imagenet sbatch slurm_train.sh   # ImageNet baseline
```

The global batch stays 16 for any number of GPUs. Re-submitting the same job resumes from `checkpoints/latest_phase*.pth`.

## 6. Monitor

```bash
squeue -u $USER
tail -f train_<jobid>.out
tail -f logs/train_phase1.log
```

## 7. Evaluate a checkpoint

```bash
sbatch slurm_eval.sh checkpoints/best_phase2.pth uni_phase2_lora
```

Metrics go to `results/metrics_<tag>.json`, figures to `results/<tag>_visualization_sample_*.png`.

## 8. Copy results back

From your local machine:

```bash
scp -r <user>@<hpc_login_address>:FM_Cell_Instance_Segmentation/results ./results_hpc
```

## Troubleshooting

- **`UNIAccessError`**: access to UNI is not approved yet, or `hf auth login` was not run on this account.
- **Out of memory in Phase 2**: add `--micro-batch 4` to the Phase 2 `train.py` line in `scripts/train_ddp.sh`; gradient accumulation keeps the global batch at 16.
- **`backend=gloo` on Linux**: the job started more processes than visible GPUs. It still trains correctly, just slower.
