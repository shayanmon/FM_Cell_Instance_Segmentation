"""
Train the UNI + HoVer-Net cell instance segmentation model with DistributedDataParallel.

Binary instance segmentation: NP head (cell vs background) + HV head (distance maps
that the watershed uses to split touching cells into individual instances).

Phase 1: UNI frozen, train feature pyramid + NP/HV heads.
Phase 2: load best Phase 1 weights, inject LoRA (rank 4) into every attention QKV
         projection of UNI, fine-tune LoRA (LR 1e-5) + decoder (LR 1e-4).

Single process:   python train.py --phase 1
DDP (any OS):     torchrun --standalone --nproc_per_node=<N> train.py --phase 1
Windows launcher: scripts\\train_local_ddp.ps1   (phases 1+2 and evaluation)

--batch-size is the GLOBAL effective batch (default 16, as in the notebook). It is
split as  batch_size = world_size x micro_batch x grad_accum_steps,  so the
optimisation recipe is the same whatever the number of processes.
"""
import argparse
import json
import os
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from uniseg.checkpoint import (atomic_save, has_legacy_nc_head, infer_backbone_source,
                               is_backbone_frozen_key, load_model_weights, trainable_state_dict)
from uniseg.data import BCCDInstanceDataset, build_eval_loader, build_train_loader
from uniseg.distributed import barrier, cleanup_distributed, gloo_host_allreduce_hook, init_distributed
from uniseg.engine import train_one_epoch, validate, warmup_cosine_restarts_lambda
from uniseg.lora import apply_lora
from uniseg.losses import CombinedLoss
from uniseg.model import CellSegModel, resolve_uni_weights
from uniseg.viz import plot_training_curves

AMP_DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'none': None}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase', type=int, choices=[1, 2], required=True)
    p.add_argument('--epochs', type=int, default=None, help='default: 100 (phase 1), 50 (phase 2)')
    p.add_argument('--batch-size', type=int, default=16, help='global effective batch size')
    p.add_argument('--micro-batch', type=int, default=None,
                   help='per-process batch per forward pass (default: phase 1 = batch/world, phase 2 = min(8, batch/world))')
    p.add_argument('--lr', type=float, default=1e-3, help='phase 1 learning rate')
    p.add_argument('--lr-decoder', type=float, default=1e-4, help='phase 2 decoder learning rate')
    p.add_argument('--lr-lora', type=float, default=1e-5, help='phase 2 LoRA learning rate')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup-epochs', type=int, default=5, help='phase 1 linear warmup')
    p.add_argument('--patience', type=int, default=15, help='early-stopping patience (epochs)')
    p.add_argument('--lora-rank', type=int, default=4)
    p.add_argument('--lora-alpha', type=float, default=4.0)
    p.add_argument('--backbone', choices=['uni', 'imagenet'], default='uni',
                   help="'imagenet' is only for ablations; the default and intended backbone is UNI")
    p.add_argument('--uni-weights', default=None, help='local UNI pytorch_model.bin (skips the Hub download)')
    p.add_argument('--init-checkpoint', default=None, help='phase 2 start point (default: <save-dir>/best_phase1.pth)')
    p.add_argument('--data-dir', default='data/processed')
    p.add_argument('--save-dir', default='checkpoints')
    p.add_argument('--results-dir', default='results')
    p.add_argument('--log-dir', default='logs')
    p.add_argument('--num-workers', type=int, default=4, help='DataLoader workers per process')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', choices=list(AMP_DTYPES), default='fp16')
    p.add_argument('--dist-backend', choices=['auto', 'gloo', 'nccl'], default='auto')
    p.add_argument('--no-resume', action='store_true', help='ignore latest_phase<N>.pth')
    p.add_argument('--max-steps-per-epoch', type=int, default=None, help='debug / smoke tests only')
    p.add_argument('--smoke-random-backbone', action='store_true',
                   help='debug only: random-init backbone (tests the DDP plumbing without UNI weights)')
    p.add_argument('--log-every', type=int, default=0, help='print a progress line every N steps (0 = off)')
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RankZeroLogger:
    """print() on rank 0 only, mirrored to a log file."""

    def __init__(self, env, path):
        self.enabled = env.is_main
        self.fh = None
        if self.enabled:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(path, 'a', encoding='utf-8')

    def __call__(self, msg=''):
        if not self.enabled:
            return
        print(msg, flush=True)
        self.fh.write(msg + '\n')
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def check_checkpoint_matches_run(ckpt, path, args, role='resume'):
    """Refuse to init/resume from a checkpoint trained on a different backbone, a smoke-test
    one, or the notebook's 3-head model (its decoder was trained with the 4-class NC loss).

    role: 'resume' (latest_phase<N>.pth) or 'init' (the Phase 2 starting point) — only the
    suggested remedy differs.
    """
    remedy = ('use --no-resume or a different --save-dir' if role == 'resume' else
              're-run Phase 1 with the current model, or pass --init-checkpoint pointing to a matching '
              'Phase 1 checkpoint')
    if has_legacy_nc_head(ckpt['model_state_dict']):
        raise SystemExit(f'{path} is a checkpoint of the old NP+HV+NC model; the binary NP+HV model '
                         f'cannot resume or start from it (evaluate.py can still evaluate it); {remedy}')
    meta = ckpt.get('backbone') or {}
    source = meta.get('name') or infer_backbone_source(ckpt['model_state_dict'])
    if source != args.backbone:
        raise SystemExit(f'{path} was trained with backbone={source!r}, but --backbone={args.backbone!r}; {remedy}')
    smoke = bool(meta.get('smoke_random_init'))
    if smoke != args.smoke_random_backbone:
        raise SystemExit(f'{path} has smoke_random_init={smoke} but --smoke-random-backbone='
                         f'{args.smoke_random_backbone}; {remedy}')


def build_optimizer(model, args):
    if args.phase == 1:
        # Only trainable params (decoder + feature pyramid)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup_cosine_restarts_lambda(args.warmup_epochs, t0=20, t_mult=2,
                                                     base_lr=args.lr, eta_min=1e-6))
        groups = {'decoder': sum(p.numel() for p in params)}
    else:
        # Separate param groups: LoRA params (backbone) vs decoder params
        lora_params, decoder_params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                (lora_params if 'lora_' in name else decoder_params).append(param)
        optimizer = torch.optim.AdamW([
            {'params': lora_params, 'lr': args.lr_lora},
            {'params': decoder_params, 'lr': args.lr_decoder},
        ], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        groups = {'lora': sum(p.numel() for p in lora_params), 'decoder': sum(p.numel() for p in decoder_params)}
    return optimizer, scheduler, groups


def main():
    args = parse_args()
    if args.epochs is None:
        args.epochs = 100 if args.phase == 1 else 50

    env = init_distributed(args.dist_backend)
    seed_everything(args.seed)   # identical on all ranks -> identical model init
    device = env.device
    amp_dtype = AMP_DTYPES[args.amp]
    for d in (args.save_dir, args.results_dir, args.log_dir):
        os.makedirs(d, exist_ok=True)
    log = RankZeroLogger(env, f'{args.log_dir}/train_phase{args.phase}.log')

    # --- Batch geometry: global batch = world x micro x accum ---
    world = env.world_size
    if args.batch_size % world:
        raise SystemExit(f'--batch-size {args.batch_size} is not divisible by world size {world}')
    per_rank = args.batch_size // world
    micro = args.micro_batch or (per_rank if args.phase == 1 else min(8, per_rank))
    if per_rank % micro:
        raise SystemExit(f'per-process batch {per_rank} is not divisible by --micro-batch {micro}')
    accum = per_rank // micro

    log('=' * 78)
    log(f'Phase {args.phase} | started {time.strftime("%Y-%m-%d %H:%M:%S")} | host {platform.node()}')
    log(f'torch {torch.__version__} (CUDA {torch.version.cuda}) | python {platform.python_version()} | {platform.platform()}')
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    log(f'DDP: active={env.active} backend={env.backend} world_size={world} '
        f'GPUs visible={len(gpu_names)} {gpu_names}')
    log(f'Batch: global={args.batch_size} = {world} proc x {micro} micro-batch x {accum} accum step(s)')

    # --- Data ---
    train_ds = BCCDInstanceDataset(f'{args.data_dir}/train', is_train=True, verbose=env.is_main)
    val_ds = BCCDInstanceDataset(f'{args.data_dir}/val', is_train=False, verbose=env.is_main)
    train_loader = build_train_loader(train_ds, micro, env, args.num_workers, args.seed)
    val_loader = build_eval_loader(val_ds, 16, env, args.num_workers)

    # --- Backbone weights: rank 0 downloads/validates first, the rest read the cache ---
    if args.backbone == 'uni' and not args.uni_weights and not args.smoke_random_backbone:
        if env.is_main:
            resolve_uni_weights()
        barrier()

    model = CellSegModel(backbone_source=args.backbone,
                         pretrained_backbone=not args.smoke_random_backbone, freeze_backbone=True,
                         uni_weights=args.uni_weights, verbose=env.is_main)
    backbone_meta = model.backbone.meta
    if args.smoke_random_backbone:
        log('WARNING: --smoke-random-backbone: backbone is RANDOMLY INITIALISED (plumbing test only).')
        backbone_meta = {'name': args.backbone, 'smoke_random_init': True}

    lora_cfg = None
    if args.phase == 2:
        init_path = args.init_checkpoint or f'{args.save_dir}/best_phase1.pth'
        if not os.path.exists(init_path):
            raise SystemExit(f'Phase 2 needs a Phase 1 checkpoint; not found: {init_path}')
        init_ckpt = torch.load(init_path, map_location='cpu', weights_only=False)
        check_checkpoint_matches_run(init_ckpt, init_path, args, role='init')
        ckpt_lora = load_model_weights(model, init_ckpt['model_state_dict'], lora_cfg=init_ckpt.get('lora'))
        log(f"Loaded Phase 2 starting point {init_path} (phase {init_ckpt.get('phase', '?')}, "
            f"epoch {init_ckpt.get('epoch')}, val_loss={init_ckpt.get('val_loss', float('nan')):.4f})")
        if ckpt_lora is None:
            apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, verbose=env.is_main)
            lora_cfg = {'rank': args.lora_rank, 'alpha': args.lora_alpha}
        else:
            # Continuing from a LoRA checkpoint: its adapters (and their rank/alpha) win
            lora_cfg = ckpt_lora
            if (ckpt_lora['rank'], ckpt_lora['alpha']) != (args.lora_rank, args.lora_alpha):
                log(f'NOTE: {init_path} already has LoRA {ckpt_lora}; ignoring --lora-rank/--lora-alpha')
        del init_ckpt

    model.to(device)
    params = model.count_parameters()
    log(f"Parameters: trainable={params['trainable']:,} frozen={params['frozen']:,} total={params['total']:,}")

    if env.active:
        # Frozen backbone weights are loaded identically on every rank from the same
        # file; skip DDP's initial 300M-parameter broadcast for them.
        frozen = [n for n, p in model.named_parameters() if is_backbone_frozen_key(n) and not p.requires_grad]
        if hasattr(DDP, '_set_params_and_buffers_to_ignore_for_model'):
            DDP._set_params_and_buffers_to_ignore_for_model(model, frozen)
        ddp_model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None,
                        find_unused_parameters=False)
        if env.backend == 'gloo' and device.type == 'cuda':
            ddp_model.register_comm_hook(None, gloo_host_allreduce_hook)
            log('DDP gradient all-reduce: gloo via host memory (comm hook)')
    else:
        ddp_model = model

    criterion = CombinedLoss(weight_np=1.0, weight_hv=2.0).to(device)
    optimizer, scheduler, groups = build_optimizer(model, args)
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16 and device.type == 'cuda'))
    log(f'Optimizer groups: {groups}')

    # --- Resume ---
    history = {'train': defaultdict(list), 'val': defaultdict(list), 'lr': [], 'epoch_time': []}
    best_val_loss, best_epoch, patience_counter, start_epoch = float('inf'), -1, 0, 0
    resumed_at_epochs = []   # every epoch this run was resumed at (crash/restart record)
    latest_path = f'{args.save_dir}/latest_phase{args.phase}.pth'
    best_path = f'{args.save_dir}/best_phase{args.phase}.pth'
    if not args.no_resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)
        check_checkpoint_matches_run(ckpt, latest_path, args)
        load_model_weights(model, ckpt['model_state_dict'], lora_cfg=ckpt.get('lora'))
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss, best_epoch = ckpt['best_val_loss'], ckpt['best_epoch']
        patience_counter = ckpt['patience_counter']
        resumed_at_epochs = list(ckpt.get('resumed_at_epochs', []))
        for split in ('train', 'val'):
            history[split] = defaultdict(list, ckpt['history'][split])
        history['lr'], history['epoch_time'] = ckpt['history']['lr'], ckpt['history']['epoch_time']
        log(f'Resumed from {latest_path}: next epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, '
            f'patience={patience_counter}')
        saved_epochs = (ckpt.get('args') or {}).get('epochs')
        if saved_epochs is not None and saved_epochs != args.epochs:
            log(f'WARNING: resuming with --epochs {args.epochs} but the run was started with {saved_epochs}; '
                f'the restored LR schedule still follows the original length.')
        if patience_counter >= args.patience:
            start_epoch = args.epochs   # run already early-stopped
        if start_epoch < args.epochs:   # re-entering a finished phase is not a resume
            resumed_at_epochs.append(start_epoch)
        del ckpt

    log(f'Phase {args.phase}: epochs={args.epochs} patience={args.patience} amp={args.amp} '
        f'lr={args.lr if args.phase == 1 else (args.lr_lora, args.lr_decoder)}')
    log(f'Train steps/epoch per process: {len(train_loader)} | val images: {len(val_ds)}')
    log('-' * 78)

    def progress(step, n_steps, loss):
        if env.is_main and args.log_every and (step + 1) % args.log_every == 0:
            print(f'    step {step + 1}/{n_steps} loss={loss:.4f}', flush=True)

    run_start = time.time()
    stopped_early = False
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)   # DDP: reshuffle per epoch

        train_m = train_one_epoch(ddp_model, train_loader, criterion, optimizer, scaler, device, amp_dtype,
                                  accum_steps=accum, max_steps=args.max_steps_per_epoch, progress=progress)
        val_m = validate(ddp_model, val_loader, criterion, device, amp_dtype)
        lr_now = [g['lr'] for g in optimizer.param_groups]
        scheduler.step()
        epoch_time = time.time() - epoch_start

        for k, v in train_m.items():
            history['train'][k].append(v)
        for k, v in val_m.items():
            history['val'][k].append(v)
        history['lr'].append(lr_now)
        history['epoch_time'].append(epoch_time)

        improved = val_m['total'] < best_val_loss
        if improved:
            best_val_loss, best_epoch, patience_counter = val_m['total'], epoch, 0
        else:
            patience_counter += 1
        stop = patience_counter >= args.patience

        lr_str = ' '.join(f'{x:.2e}' for x in lr_now)
        log(f"Epoch {epoch:03d} | Train {train_m['total']:.4f} (np={train_m['np_loss']:.3f} "
            f"hv={train_m['hv_loss']:.3f}) | Val {val_m['total']:.4f} "
            f"(np={val_m['np_loss']:.3f} hv={val_m['hv_loss']:.3f}) | "
            f"LR {lr_str} | {epoch_time:.1f}s{' | * best' if improved else ''}")

        if env.is_main:
            common = {
                'phase': args.phase,
                'epoch': epoch,
                'model_state_dict': trainable_state_dict(model),
                'backbone': backbone_meta,
                'lora': lora_cfg,
                'args': vars(args),
                'world_size': world,
            }
            if improved:
                atomic_save({**common, 'val_loss': best_val_loss, 'val_metrics': val_m}, best_path)
            atomic_save({**common,
                         'optimizer_state_dict': optimizer.state_dict(),
                         'scheduler_state_dict': scheduler.state_dict(),
                         'scaler_state_dict': scaler.state_dict(),
                         'best_val_loss': best_val_loss,
                         'best_epoch': best_epoch,
                         'patience_counter': patience_counter,
                         'resumed_at_epochs': resumed_at_epochs,
                         'history': {'train': dict(history['train']), 'val': dict(history['val']),
                                     'lr': history['lr'], 'epoch_time': history['epoch_time']}},
                        latest_path)
        barrier()
        if stop:
            log(f'Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)')
            stopped_early = True
            break

    wall = time.time() - run_start
    if env.is_main:
        hist = {'train': dict(history['train']), 'val': dict(history['val']),
                'lr': history['lr'], 'epoch_time': history['epoch_time']}
        with open(f'{args.results_dir}/history_phase{args.phase}.json', 'w') as f:
            json.dump(hist, f, indent=1)
        if hist['train'].get('total'):
            prefix = 'Phase 1 (UNI frozen): ' if args.phase == 1 else 'Phase 2 (UNI + LoRA): '
            plot_training_curves(hist, title_prefix=prefix,
                                 save_path=f'{args.results_dir}/training_curves_phase{args.phase}.png')
        n_epochs = len(hist['epoch_time'])
        run_info = {
            'phase': args.phase,
            'backbone': backbone_meta,
            'lora': lora_cfg,
            'distributed': {'active': env.active, 'backend': env.backend, 'world_size': world,
                            'gpus': gpu_names, 'platform': platform.platform()},
            'batch': {'global': args.batch_size, 'micro_batch_per_process': micro, 'grad_accum_steps': accum},
            'epochs_completed': n_epochs,
            'epochs_max': args.epochs,
            'stopped_early': stopped_early or patience_counter >= args.patience,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'mean_epoch_seconds': float(np.mean(hist['epoch_time'])) if n_epochs else None,
            'total_train_seconds': float(np.sum(hist['epoch_time'])) if n_epochs else None,
            'this_invocation_seconds': wall,
            'resumed_at_epochs': resumed_at_epochs,
            'versions': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version()},
            'args': vars(args),
        }
        with open(f'{args.results_dir}/run_info_phase{args.phase}.json', 'w') as f:
            json.dump(run_info, f, indent=2)
        log(f'Phase {args.phase} complete. Best val loss {best_val_loss:.4f} at epoch {best_epoch}. '
            f'{n_epochs} epochs, {run_info["total_train_seconds"] or 0:.0f}s of training.')
    log.close()
    barrier()
    cleanup_distributed()


if __name__ == '__main__':
    main()
