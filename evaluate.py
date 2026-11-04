"""
Evaluate a trained checkpoint on the held-out BCCD test set (159 images).

    python evaluate.py                                   # checkpoints/best_phase2.pth (else best_phase1.pth)
    python evaluate.py --checkpoint checkpoints/best_phase1.pth --tag uni_phase1

The checkpoint is rebuilt with the backbone it was trained with (UNI, or the
ImageNet ViT-L/16 of the original notebook run) and loaded strictly, with LoRA
adapters applied when present.

Instances are separated with the reference HoVer-Net post-processing by default;
--postproc notebook uses the notebook's version, which merges touching cells
(see uniseg/postprocess.py) and is kept to reproduce earlier numbers.

--legacy-notebook-eval reproduces the notebook's evaluation exactly, including
its bugs (LoRA adapters dropped by load_state_dict(strict=False), tanh on the
HV output, notebook post-processing). It exists only to show the old reported
numbers are reproducible.
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from uniseg.checkpoint import build_model_from_checkpoint
from uniseg.data import BCCDInstanceDataset
from uniseg.engine import autocast_ctx
from uniseg.metrics import InstanceSegEvaluator
from uniseg.postprocess import POSTPROC_CHOICES, decode_outputs, post_process
from uniseg.viz import plot_sample


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--data-dir', default='data/processed')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--results-dir', default='results')
    p.add_argument('--tag', default=None, help='suffix for metrics_<tag>.json and figure names')
    p.add_argument('--uni-weights', default=None)
    p.add_argument('--hv-activation', choices=['clip', 'tanh'], default='clip')
    p.add_argument('--postproc', choices=POSTPROC_CHOICES, default='hovernet',
                   help="watershed post-processing: 'hovernet' (reference) or 'notebook' (v1, merges touching cells)")
    p.add_argument('--legacy-notebook-eval', action='store_true')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--num-vis', type=int, default=5)
    return p.parse_args()


def notebook_strict_false_view(state_dict):
    """What the notebook's eval actually loaded: LoRA tensors dropped, frozen qkv kept."""
    out = {}
    for k, v in state_dict.items():
        if 'lora_' in k:
            continue
        out[k.replace('attn.qkv.original.', 'attn.qkv.')] = v
    return out


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = 'checkpoints/best_phase2.pth'
        if not os.path.exists(ckpt_path):
            ckpt_path = 'checkpoints/best_phase1.pth'
    if not os.path.exists(ckpt_path):
        raise SystemExit(f'Checkpoint not found: {ckpt_path}')

    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hv_activation, postproc = args.hv_activation, args.postproc
    if args.legacy_notebook_eval:
        ckpt['model_state_dict'] = notebook_strict_false_view(ckpt['model_state_dict'])
        ckpt['lora'] = None
        hv_activation, postproc = 'tanh', 'notebook'
        print('LEGACY MODE: reproducing the notebook evaluation (LoRA dropped, tanh on HV, '
              'notebook post-processing).')
    model, backbone_source, lora_cfg, legacy_nc = build_model_from_checkpoint(
        ckpt, device, uni_weights=args.uni_weights)

    dataset = BCCDInstanceDataset(f'{args.data_dir}/{args.split}', is_train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    tag = args.tag or os.path.splitext(os.path.basename(ckpt_path))[0]
    os.makedirs(args.results_dir, exist_ok=True)
    evaluator = InstanceSegEvaluator()
    vis_samples = []
    t0 = time.time()

    with torch.no_grad():
        for images, targets in loader:
            with autocast_ctx(device, torch.float16):
                out = model(images.to(device, non_blocking=True))
            np_b, hv_b = decode_outputs(out, hv_activation)
            gt_insts = targets['instance_map'].numpy()

            for i in range(images.shape[0]):
                inst_pred = post_process(np_b[i], hv_b[i], method=postproc)
                evaluator.add(gt_insts[i], inst_pred)
                if len(vis_samples) < args.num_vis:
                    vis_samples.append({
                        'image': images[i].numpy(), 'gt_inst': gt_insts[i], 'pred_inst': inst_pred,
                        'np_pred': np_b[i], 'hv_pred': hv_b[i],
                    })

    metrics = evaluator.compute_all()
    metrics['meta'] = {
        'checkpoint': os.path.abspath(ckpt_path),
        'split': args.split,
        'phase': ckpt.get('phase'),
        'epoch': ckpt.get('epoch'),
        'val_loss_at_save': ckpt.get('val_loss'),
        'backbone': ckpt.get('backbone') or {'name': backbone_source, 'inferred_from_checkpoint': True},
        'lora': lora_cfg,
        'legacy_nc_head_dropped': legacy_nc,
        'hv_activation': hv_activation,
        'postproc': postproc,
        'legacy_notebook_eval': args.legacy_notebook_eval,
        'eval_seconds': round(time.time() - t0, 1),
    }

    out_path = f'{args.results_dir}/metrics_{tag}.json'
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Per-image counts (same order as the sorted test images) for paired comparisons
    per_image_path = f'{args.results_dir}/per_image_{tag}.json'
    with open(per_image_path, 'w') as f:
        json.dump({'tag': tag, 'meta': metrics['meta'],
                   'images': [p.stem for p in dataset.image_files],
                   'stats': evaluator.per_image_stats()}, f, indent=1)

    m = metrics
    print('\n' + '=' * 60)
    print(f'TEST SET EVALUATION — {tag} (backbone={backbone_source}, '
          f"{'LoRA r=' + str(lora_cfg['rank']) if lora_cfg else 'no LoRA'}, {postproc} post-processing)")
    print('=' * 60)
    print(f"Foreground (cell vs bg)  IoU={m['foreground']['iou']:.4f}  Dice={m['foreground']['dice']:.4f}")
    d = m['detection']
    print(f"Detection @IoU0.5        P={d['precision']:.4f}  R={d['recall']:.4f}  F1={d['f1']:.4f}  "
          f"(TP={d['tp']} FP={d['fp']} FN={d['fn']})")
    print(f"mAP@0.50={m['mAP_50']:.4f}  mAP@[.50:.95]={m['mAP_50_95']:.4f}")
    pq = m['panoptic']
    print(f"Panoptic (standard)      PQ={pq['PQ']:.4f}  SQ={pq['SQ']:.4f}  RQ={pq['RQ']:.4f}")
    lg = m['legacy']
    print(f"Legacy notebook formulas mean_iou={lg['mean_iou']:.4f}  PQ={lg['panoptic']['PQ']:.4f}  "
          f"RQ={lg['panoptic']['RQ']:.4f}")
    c = m['counting']
    print(f"Counting                 GT/img={c['mean_gt_per_image']:.1f}  pred/img={c['mean_pred_per_image']:.1f}  "
          f"MAE={c['mae']:.2f}  MAPE={c['mean_abs_pct_error']:.1f}%")
    print(f'\nMetrics saved to {out_path} (per-image counts: {per_image_path})')

    for i, s in enumerate(vis_samples):
        plot_sample(s, f'{tag} — test sample {i + 1}',
                    f'{args.results_dir}/{tag}_visualization_sample_{i + 1}.png')
    if vis_samples:
        print(f'{len(vis_samples)} visualizations saved to {args.results_dir}/')


if __name__ == '__main__':
    main()
