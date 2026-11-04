# ============================================================
# Visualization & Analysis (notebook Cells 40-41), headless
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .config import IMAGENET_MEAN, IMAGENET_STD


def plot_training_curves(history, title_prefix='', save_path=None):
    """Plot training/validation loss curves for all branches.

    `history` = {'train': {'total': [...], 'np_loss': [...], ...}, 'val': {...}, 'lr': [...]}
    """
    panels = [('total', 'Total Loss'), ('np_loss', 'NP Branch (Cell vs Background)'),
              ('hv_loss', 'HV Branch (Distance Maps)')]
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    for ax, (key, title) in zip(axes.flat, panels):
        tr = history['train'].get(key, [])
        va = history['val'].get(key, [])
        epochs = np.arange(len(tr))
        ax.plot(epochs, tr, label='Train', linewidth=2)
        ax.plot(np.arange(len(va)), va, label='Val', linewidth=2)
        if key == 'total' and va:
            best = int(np.argmin(va))
            ax.axvline(best, color='gray', linestyle='--', linewidth=1, label=f'best (epoch {best})')
        ax.set_title(f'{title_prefix}{title}', fontsize=13)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def colorize_instances(inst_map):
    """Create an RGB visualization of an instance map, one random colour per cell."""
    h, w = inst_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        rng = np.random.RandomState(int(iid))
        vis[inst_map == iid] = tuple(rng.randint(50, 255, 3).tolist())
    return vis


def instance_boundaries(inst_map):
    """Cell pixels next to a different cell or to background (4-neighbourhood)."""
    edges = np.zeros(inst_map.shape, dtype=bool)
    edges[:-1, :] |= inst_map[:-1, :] != inst_map[1:, :]
    edges[1:, :] |= inst_map[1:, :] != inst_map[:-1, :]
    edges[:, :-1] |= inst_map[:, :-1] != inst_map[:, 1:]
    edges[:, 1:] |= inst_map[:, 1:] != inst_map[:, :-1]
    return edges & (inst_map > 0)


def create_error_map(pred_inst, gt_inst, iou_thresh=0.5):
    """
    Create TP/FP/FN error visualization.
    Green = TP (matched), Red = FP (no GT match), Blue = FN (missed GT).
    """
    h, w = pred_inst.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    pred_ids = [i for i in np.unique(pred_inst) if i > 0]
    gt_ids = [i for i in np.unique(gt_inst) if i > 0]
    matched_pred, matched_gt = set(), set()

    for pid in pred_ids:
        p_mask = (pred_inst == pid)
        best_iou, best_gid = 0, None
        for gid in gt_ids:
            g_mask = (gt_inst == gid)
            inter = (p_mask & g_mask).sum()
            if inter == 0:
                continue
            iou = inter / ((p_mask | g_mask).sum() + 1e-8)
            if iou > best_iou:
                best_iou, best_gid = iou, gid
        if best_iou >= iou_thresh:
            matched_pred.add(pid)
            matched_gt.add(best_gid)

    for pid in pred_ids:
        error_map[pred_inst == pid] = (0, 200, 0) if pid in matched_pred else (200, 0, 0)
    for gid in gt_ids:
        if gid not in matched_gt:
            error_map[gt_inst == gid] = (0, 0, 200)
    return error_map


def denormalize(image_chw):
    img = image_chw.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def plot_sample(sample, title, save_path):
    """3x3 panel: image / GT / prediction, NP + HV maps, cell masks with outlines + error map."""
    img = denormalize(sample['image'])
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    n_gt = len([x for x in np.unique(sample['gt_inst']) if x > 0])
    axes[0, 1].imshow(img)
    axes[0, 1].imshow(colorize_instances(sample['gt_inst']), alpha=0.5)
    axes[0, 1].set_title(f'GT Overlay ({n_gt} instances)')
    n_pred = len([x for x in np.unique(sample['pred_inst']) if x > 0])
    axes[0, 2].imshow(img)
    axes[0, 2].imshow(colorize_instances(sample['pred_inst']), alpha=0.5)
    axes[0, 2].set_title(f'Prediction Overlay ({n_pred} instances)')

    axes[1, 0].imshow(sample['np_pred'], cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('NP Prediction (Cell Probability)')
    axes[1, 1].imshow(sample['hv_pred'][:, :, 0], cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('Horizontal Distance Prediction')
    axes[1, 2].imshow(sample['hv_pred'][:, :, 1], cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 2].set_title('Vertical Distance Prediction')

    for ax, key, name in ((axes[2, 0], 'gt_inst', 'GT'), (axes[2, 1], 'pred_inst', 'Predicted')):
        inst = sample[key]
        mask_vis = np.repeat((inst > 0)[:, :, None], 3, axis=2).astype(np.uint8) * 255
        mask_vis[instance_boundaries(inst)] = (220, 50, 50)
        ax.imshow(mask_vis)
        ax.set_title(f'{name} Cell Mask + Instance Outlines')
    axes[2, 2].imshow(img)
    axes[2, 2].imshow(create_error_map(sample['pred_inst'], sample['gt_inst']), alpha=0.6)
    axes[2, 2].set_title('Error Map (Green=TP, Red=FP, Blue=FN)')

    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle(title, fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
