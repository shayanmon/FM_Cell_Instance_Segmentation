# ============================================================
# Training & Validation Loops (notebook Cells 29, 31, 33) — DDP-aware
# ============================================================
# Differences from the notebook loops, all needed for correct DDP:
#   - losses are accumulated as per-sample sums and all-reduced, so every rank
#     sees the same global train/val loss (and makes the same early-stopping call)
#   - gradient-accumulation micro-steps run under DDP no_sync()
#   - a trailing accumulation window shorter than `accum_steps` is averaged over
#     its real length
import math
import time
from collections import defaultdict

import torch

from .distributed import all_reduce_sum, no_sync_if

LOSS_KEYS = ('total', 'np_loss', 'hv_loss')


def autocast_ctx(device, amp_dtype):
    enabled = amp_dtype is not None and device.type == 'cuda'
    return torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=enabled)


def _move(targets, device):
    return {k: v.to(device, non_blocking=True) for k, v in targets.items()}


def _reduce_sums(sums, count, device):
    vec = torch.stack([sums[k] for k in LOSS_KEYS] + [count]).to(device, torch.float64)
    all_reduce_sum(vec)
    n = vec[-1].clamp(min=1)
    return {k: (vec[i] / n).item() for i, k in enumerate(LOSS_KEYS)}, int(vec[-1].item())


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_dtype,
                    accum_steps=1, max_grad_norm=1.0, max_steps=None, progress=None):
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    sums = {k: torch.zeros((), device=device, dtype=torch.float64) for k in LOSS_KEYS}
    count = torch.zeros((), device=device, dtype=torch.float64)

    n_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets) in enumerate(loader):
        if step >= n_steps:
            break
        images = images.to(device, non_blocking=True)
        targets = _move(targets, device)

        window_start = (step // accum_steps) * accum_steps
        window_len = min(accum_steps, n_steps - window_start)
        is_update_step = (step + 1 - window_start) == window_len

        with no_sync_if(model, not is_update_step):
            with autocast_ctx(device, amp_dtype):
                preds = model(images)
                losses = criterion(preds, targets)
            scaler.scale(losses['total'] / window_len).backward()

        if is_update_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = images.shape[0]
        for k in LOSS_KEYS:
            sums[k] += losses[k].detach().double() * bs
        count += bs
        if progress is not None:
            progress(step, n_steps, losses['total'].item())

    return _reduce_sums(sums, count, device)[0]


@torch.no_grad()
def validate(model, loader, criterion, device, amp_dtype):
    model.eval()
    sums = {k: torch.zeros((), device=device, dtype=torch.float64) for k in LOSS_KEYS}
    count = torch.zeros((), device=device, dtype=torch.float64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = _move(targets, device)
        with autocast_ctx(device, amp_dtype):
            preds = model(images)
            losses = criterion(preds, targets)
        bs = images.shape[0]
        for k in LOSS_KEYS:
            sums[k] += losses[k].detach().double() * bs
        count += bs

    metrics, n = _reduce_sums(sums, count, device)
    return metrics


# ------------------------------------------------------------
# LR schedules
# ------------------------------------------------------------

def warmup_cosine_restarts_lambda(warmup_epochs=5, t0=20, t_mult=2, base_lr=1e-3, eta_min=1e-6):
    """Phase 1 schedule as an epoch -> LR-factor function for LambdaLR.

    Linear warmup over `warmup_epochs`, then SGDR (CosineAnnealingWarmRestarts,
    T_0=20, T_mult=2). The notebook built a LambdaLR and a
    CosineAnnealingWarmRestarts on the same optimizer; the second constructor
    reset the LR to the peak, so epoch 0 actually ran at the full LR. A single
    LambdaLR gives the intended schedule and resumes cleanly.
    """
    floor = eta_min / base_lr

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = epoch - warmup_epochs
        period = t0
        while t >= period:
            t -= period
            period *= t_mult
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t / period))

    return lr_lambda


class EpochTimer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start
