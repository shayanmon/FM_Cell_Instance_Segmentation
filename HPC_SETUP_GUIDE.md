# HPC Setup & Run Guide

Step-by-step instructions for deploying and running the UNI Cell Instance Segmentation project on the Pioneer HPC cluster.

---

## 1. Upload Project to HPC

From your **local machine**:

```bash
rsync -avz --exclude '__pycache__' --exclude '.ipynb_checkpoints' \
  /Users/shayan/UNI_instance_seg/ \
  sxm1165@<hpc_login_address>:~/UNI_instance_seg/
```

This uploads the entire project folder including the dataset (`BCCD Dataset with mask/`), notebook, SLURM scripts, and environment file.

---

## 2. SSH into HPC

```bash
ssh sxm1165@<hpc_login_address>
```

You'll land on a **login node** (e.g., `hpc7`). Do NOT run training here.

---

## 3. Initialize Conda

If `conda` is not found, you need to initialize it for your shell:

```bash
# Find your miniconda installation
ls ~/miniconda3/bin/conda

# Initialize for current session
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# Make it permanent (one-time setup)
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

---

## 4. Create the Conda Environment

```bash
cd ~/UNI_instance_seg
conda env create -f environment.yml
```

This takes 5-20 minutes. Wait for it to finish — you'll see:
```
done
# To activate this environment, use
#     $ conda activate uni_seg
```

Activate it:
```bash
conda activate uni_seg
```

---

## 5. Fix NumPy Version Conflict

Conda may install NumPy 2.x, which is incompatible with PyTorch 2.1. Fix it:

```bash
pip install "numpy<2" --force-reinstall
```

The opencv warning that appears is harmless and can be ignored.

---

## 6. Login to HuggingFace

Required to download the UNI model weights (gated model):

```bash
python -c "from huggingface_hub import login; login()"
```

Paste your HuggingFace token when prompted. Get one at: https://huggingface.co/settings/tokens

Make sure you have access to the UNI model at: https://huggingface.co/MahmoodLab/uni

---

## 7. Create Output Directories

```bash
cd ~/UNI_instance_seg
mkdir -p logs checkpoints results data/processed
```

These must exist before running anything.

---

## 8. Request an Interactive GPU Node

**Do NOT run on the login node** — it has no GPU and jobs will be killed.

```bash
srun --partition=gpu --gres=gpu:1 --mem=64G --time=12:00:00 --cpus-per-task=8 --pty bash
```

Wait for the allocation. Once assigned, you'll see a new prompt like:
```
(uni_seg) [sxm1165@gput062 UNI_instance_seg]$
```

---

## 9. Verify GPU Access

On the compute node:

```bash
conda activate uni_seg
cd ~/UNI_instance_seg
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output (GPU type may vary):
```
True Tesla P100-PCIE-12GB
```

If you see `False` or errors, check:
- Is `pytorch-cuda` installed? (`pip list | grep torch`)
- Are you on a GPU node? (`nvidia-smi`)

---

## 10. Launch Jupyter Notebook

On the **compute node**:

```bash
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

It will print a URL with a token like:
```
http://gput062:8888/?token=abc123def456...
```

Note the **node name** (e.g., `gput062`) and the **token**.

---

## 11. SSH Tunnel from Local Machine

Open a **new terminal** on your local machine (keep the HPC terminal open):

```bash
ssh -L 8888:<node_name>:8888 sxm1165@<hpc_login_address>
```

Replace `<node_name>` with the compute node (e.g., `gput062`).

Then open in your browser:
```
http://localhost:8888
```

Enter the token from step 10 if prompted.

---

## 12. Run the Notebook

Open `cell_instance_segmentation.ipynb` in the Jupyter interface and run cells sequentially (Shift+Enter).

**Important notes:**
- If you get a P100 (12GB VRAM), you may need to reduce `BATCH_SIZE` in the parameters cell (try 8 for Phase 1, 4 for Phase 2)
- The notebook runs end-to-end: data processing, training (both phases), evaluation, and visualization
- Checkpoints are saved to `checkpoints/`, results to `results/`

---

## Alternative: Run via Command Line (No Jupyter)

If you prefer not to use Jupyter, you can execute the entire notebook from the command line on the compute node:

```bash
cd ~/UNI_instance_seg
conda activate uni_seg
jupyter nbconvert --to notebook --execute cell_instance_segmentation.ipynb \
  --output results/train_run.ipynb \
  --ExecutePreprocessor.timeout=-1 2>&1 | tee logs/run.log
```

---

## Alternative: SLURM Batch Job (Automated)

If you prefer non-interactive execution:

```bash
cd ~/UNI_instance_seg
mkdir -p logs checkpoints results data/processed
export HF_TOKEN="hf_your_token_here"
sbatch slurm_train.sh
```

Monitor with:
```bash
squeue -u $USER          # Check job status
tail -f train_*.out      # Watch stdout log
cat train_*.err          # Check for errors
```

After training completes, run evaluation:
```bash
sbatch slurm_eval.sh
```

---

## 13. Download Results to Local Machine

After training completes, from your **local machine**:

```bash
rsync -avz sxm1165@<hpc_login_address>:~/UNI_instance_seg/results/ \
  /Users/shayan/UNI_instance_seg/results/

rsync -avz sxm1165@<hpc_login_address>:~/UNI_instance_seg/checkpoints/ \
  /Users/shayan/UNI_instance_seg/checkpoints/
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `conda: command not found` | Run `eval "$(~/miniconda3/bin/conda shell.bash hook)"` |
| `huggingface-cli: command not found` | Use `python -c "from huggingface_hub import login; login()"` instead |
| NumPy 2.x crash with PyTorch | Run `pip install "numpy<2" --force-reinstall` |
| SLURM `mkdir: Permission denied` | Create directories manually before submitting: `mkdir -p logs checkpoints results data/processed` |
| SLURM job fails with no logs | Log files are in the project directory (`train_<jobid>.out`), not in `logs/`. Check with `ls ~/UNI_instance_seg/train_*.err` |
| Interactive session times out | Request more time: `--time=12:00:00` |
| CUDA out of memory | Reduce `BATCH_SIZE` in the notebook parameters cell (try 8 for Phase 1, 4 for Phase 2) |
| SSH tunnel doesn't connect | Make sure you use the correct compute node name in the tunnel command |
| VPN drops / `Connection refused` on tunnel | See **Reconnecting After VPN Dropout** section below |

---

## Reconnecting After VPN Dropout

If your VPN disconnects (you'll see `channel 3: open failed: connect failed: Connection refused`), the Jupyter kernel and any running cells will die. No files are lost — only the in-memory kernel state.

### Step 1: Reconnect VPN
Reconnect your CWRU VPN.

### Step 2: SSH back into HPC
```bash
ssh sxm1165@<hpc_login_address>
```

### Step 3: Check if your interactive job is still alive
```bash
squeue -u $USER
```

**If the job is listed** — note the node name (e.g., `gput062`) and skip to Step 4.

**If the job is gone** — request a new interactive session:
```bash
srun --partition=gpu --gres=gpu:1 --mem=64G --time=12:00:00 --cpus-per-task=8 --pty bash
```
Then start from Step 9 above (activate conda, verify GPU, launch Jupyter).

### Step 4: SSH into the compute node
```bash
ssh gput062
```
(Replace with your actual node name from `squeue`)

### Step 5: Check if Jupyter is still running
```bash
ps aux | grep jupyter
```

**If Jupyter is running** — skip to Step 6.

**If Jupyter is not running** — restart it:
```bash
conda activate uni_seg
cd ~/UNI_instance_seg
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

### Step 6: Re-create the SSH tunnel
On your **local machine** (new terminal):
```bash
ssh -L 8888:<node_name>:8888 sxm1165@<hpc_login_address>
```

Open `http://localhost:8888` in your browser.

### Step 7: Re-run notebook cells
Your notebook file is saved, but the kernel state (variables, loaded models) is lost. You'll need to **re-run cells from the top** (Kernel → Restart & Run All, or Shift+Enter through each cell).

If training was in progress, any completed checkpoints are still saved in `checkpoints/`. The training loop will start from scratch unless you modify it to resume from a checkpoint.

---

## Distributed Training (Multi-GPU with DDP)

The notebook supports optional DistributedDataParallel (DDP) for multi-GPU training. DDP is **disabled by default** — the notebook works on a single GPU in Jupyter without any changes.

### How it works

- When `USE_DDP = True` or SLURM environment variables (`SLURM_PROCID`) are detected, DDP initializes automatically
- Each GPU runs its own copy of the training process, splitting the data via `DistributedSampler`
- Gradients are synchronized across GPUs after each backward pass
- Only rank 0 (the main process) prints logs and saves checkpoints

### Running DDP via SLURM

```bash
cd ~/UNI_instance_seg
mkdir -p logs checkpoints results data/processed
sbatch slurm_train_ddp.sh
```

This requests **2 GPUs on 1 node** and launches 2 processes (one per GPU). Each process runs the full notebook via papermill with `-p USE_DDP True`.

Monitor:
```bash
squeue -u $USER
tail -f train_ddp_*.out
cat train_ddp_*.err
```

### Expected speedup

Estimates based on: only 11M trainable params → 44 MB gradient sync per step, PCIe 3.0 bandwidth ~12 GB/s → ~3.7 ms sync time, vs ~500 ms compute per step. Formula: `speedup = N / (1 + overhead/compute)`.

| Setup | Phase 1 (BS=16) | Phase 2 (BS=8) | Basis |
|-------|-----------------|-----------------|-------|
| 1 GPU | 1x (baseline) | 1x (baseline) | — |
| 2 GPUs (DDP) | ~1.8x faster | ~1.6–1.7x faster | 0.7% overhead ratio, minor I/O contention |
| 4 GPUs (DDP) | ~3.0–3.2x faster | ~2.5–2.8x faster | AllReduce ring grows on PCIe |

Phase 2 is slightly slower to scale because smaller batch sizes (8 vs 16) reduce compute time per step, worsening the compute-to-communication ratio.

To use 4 GPUs, edit `slurm_train_ddp.sh` and change `--gres=gpu:4` and `--ntasks-per-node=4`.

### Other speedup options (single GPU)

| Feature | How to enable | Expected effect |
|---------|---------------|-----------------|
| `torch.compile()` | Set `USE_COMPILE = True` in parameters cell | 10-20% faster (first epoch slower due to compilation) |
| Gradient accumulation | Set `GRAD_ACCUM_TARGET = 32` in parameters cell | Larger effective batch → better convergence, same speed |
| Mixed precision | Already enabled (fp16) | Already active |

### Important notes

- DDP is designed for SLURM batch jobs. For interactive Jupyter, keep `USE_DDP = False`
- Checkpoints saved in DDP mode are compatible with single-GPU loading (no `module.` prefix)
- `torch.compile` on P100 (compute cap 6.0) gives modest gains; more effective on A100/V100
