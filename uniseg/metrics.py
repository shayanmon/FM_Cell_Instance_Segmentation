# ============================================================
# Evaluation Metrics — InstanceSegEvaluator (notebook Cell 37)
# ============================================================
# Binary instance segmentation: every instance is a "cell", so the pixel
# metrics are foreground (cell vs background) IoU/Dice and everything else is
# instance-level. Same matching rules as the notebook (greedy one-to-one at
# IoU >= 0.5, instance area as the mAP confidence proxy), with two corrections
# reported next to the legacy numbers rather than silently replacing them:
#
# 1. Panoptic Quality. The notebook computed RQ as the harmonic mean of
#    TP/(TP+FP/2) and TP/(TP+FN/2), which overstates RQ. The standard
#    definition (Kirillov et al., 2019) is RQ = TP / (TP + FP/2 + FN/2),
#    i.e. exactly the detection F1 at IoU 0.5.  -> 'panoptic' (standard),
#    'legacy' -> 'panoptic' (notebook formula).
# 2. Class-wise pixel metrics. The notebook averaged IoU over RBC, WBC and
#    Platelet, but the processed BCCD masks are binary: every ground-truth
#    cell is class 1, so WBC/Platelet scored 0.0 ("absent", not "failed") and
#    the 3-class mean was foreground IoU / 3. That mean is kept only as
#    'legacy' -> 'mean_iou' so v1 numbers can be traced; 'foreground' is the
#    real pixel metric. (In every notebook-checkpoint evaluation its "RBC" IoU
#    equalled foreground IoU exactly: the NC head labelled every cell class 1.)
# A per-image cell-count error is added as well.
import numpy as np

from .config import LEGACY_CLASS_NAMES


def iou_matrix(gt_inst, pred_inst):
    """IoU between all GT and predicted instances via a contingency table.

    Returns (iou (G, P), gt_ids, pred_ids). Values equal the notebook's
    pairwise mask loop: inter / (union + 1e-8).
    """
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return np.zeros((len(gt_ids), len(pred_ids))), list(gt_ids), list(pred_ids)

    g = np.searchsorted(gt_ids, gt_inst.ravel())        # 0..G-1 for foreground pixels
    p = np.searchsorted(pred_ids, pred_inst.ravel())
    g_fg = gt_inst.ravel() > 0
    p_fg = pred_inst.ravel() > 0

    G, P = len(gt_ids), len(pred_ids)
    gt_area = np.bincount(g[g_fg], minlength=G).astype(np.float64)
    pred_area = np.bincount(p[p_fg], minlength=P).astype(np.float64)
    both = g_fg & p_fg
    inter = np.bincount(g[both] * P + p[both], minlength=G * P).reshape(G, P).astype(np.float64)
    union = gt_area[:, None] + pred_area[None, :] - inter
    return inter / (union + 1e-8), list(gt_ids), list(pred_ids)


def greedy_match(iou_mat, iou_thresh):
    """Greedy best-IoU-first one-to-one matching. Returns list of (gi, pi, iou)."""
    iou_mat = iou_mat.copy()
    matches = []
    if iou_mat.size == 0:
        return matches
    while True:
        idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        if iou_mat[idx] < iou_thresh:
            break
        gi, pi = idx
        matches.append((gi, pi, float(iou_mat[idx])))
        iou_mat[gi, :] = 0
        iou_mat[:, pi] = 0
    return matches


class InstanceSegEvaluator:
    """
    Evaluator for binary (cell vs background) instance segmentation.
    Computes pixel-level (foreground), object-level, and panoptic metrics.
    """

    def __init__(self):
        self.results = []

    def add(self, gt_inst, pred_inst):
        """
        Add one image's results.
        gt_inst, pred_inst: (H, W) instance maps (0=background, 1..N=cells)
        """
        iou_mat, gt_ids, pred_ids = iou_matrix(gt_inst, pred_inst)
        self.results.append({
            'gt_inst': gt_inst,
            'pred_inst': pred_inst,
            'iou': iou_mat,
            'gt_ids': gt_ids,
            'pred_ids': pred_ids,
        })

    def compute_pixel_metrics(self):
        """Foreground (cell vs background) IoU/Dice at pixel level, pooled over all images."""
        fg_tp = fg_fp = fg_fn = 0
        for r in self.results:
            gt_fg = r['gt_inst'] > 0
            pred_fg = r['pred_inst'] > 0
            fg_tp += np.logical_and(gt_fg, pred_fg).sum()
            fg_fp += np.logical_and(~gt_fg, pred_fg).sum()
            fg_fn += np.logical_and(gt_fg, ~pred_fg).sum()
        return {
            'iou': float(fg_tp / (fg_tp + fg_fp + fg_fn + 1e-8)),
            'dice': float(2 * fg_tp / (2 * fg_tp + fg_fp + fg_fn + 1e-8)),
        }

    def _match_counts(self, iou_thresh=0.5):
        total_tp = total_fp = total_fn = 0
        sum_iou = 0.0
        for r in self.results:
            matches = greedy_match(r['iou'], iou_thresh)
            total_tp += len(matches)
            total_fp += len(r['pred_ids']) - len(matches)
            total_fn += len(r['gt_ids']) - len(matches)
            sum_iou += sum(m[2] for m in matches)
        return total_tp, total_fp, total_fn, sum_iou

    def compute_detection_metrics(self, iou_thresh=0.5):
        """Object-level precision, recall, F1 at given IoU threshold."""
        tp, fp, fn, _ = self._match_counts(iou_thresh)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}

    def compute_map(self, iou_thresholds=None):
        """Compute mAP at specified IoU thresholds (COCO-style IoU sweep, 11-point AP).

        Confidence proxy = instance area (as in the notebook).
        """
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05)

        all_detections = []  # (confidence, best_iou)
        total_gt = 0
        for r in self.results:
            total_gt += len(r['gt_ids'])
            if not r['pred_ids']:
                continue
            areas = np.bincount(r['pred_inst'].ravel())
            for pi, pid in enumerate(r['pred_ids']):
                best_iou = r['iou'][:, pi].max() if r['iou'].size > 0 else 0
                all_detections.append((areas[pid], best_iou))

        if total_gt == 0 or not all_detections:
            return {'mAP_50': 0.0, 'mAP_50_95': 0.0}

        # Stable sort by confidence (descending), as list.sort in the notebook
        all_detections.sort(key=lambda x: -x[0])
        best = np.array([d[1] for d in all_detections])

        aps = []
        for thresh in iou_thresholds:
            is_tp = best >= thresh
            tp_cumsum = np.cumsum(is_tp)
            fp_cumsum = np.cumsum(~is_tp)
            recall_curve = tp_cumsum / total_gt
            precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

            # 11-point interpolation
            ap = 0
            for r_thresh in np.linspace(0, 1, 11):
                prec_at_r = precision_curve[recall_curve >= r_thresh]
                ap += (prec_at_r.max() if len(prec_at_r) > 0 else 0) / 11
            aps.append(ap)

        return {'mAP_50': float(aps[0]), 'mAP_50_95': float(np.mean(aps))}

    def compute_panoptic_quality(self, iou_thresh=0.5):
        """Standard PQ = SQ x RQ, plus the notebook's (inflated) RQ variant for comparison."""
        tp, fp, fn, sum_iou = self._match_counts(iou_thresh)
        sq = sum_iou / (tp + 1e-8)
        rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)

        rq_prec = tp / (tp + fp / 2 + 1e-8)
        rq_rec = tp / (tp + fn / 2 + 1e-8)
        rq_legacy = 2 * rq_prec * rq_rec / (rq_prec + rq_rec + 1e-8)

        return ({'PQ': float(sq * rq), 'SQ': float(sq), 'RQ': float(rq)},
                {'PQ': float(sq * rq_legacy), 'SQ': float(sq), 'RQ': float(rq_legacy)})

    def compute_counting(self):
        """Per-image cell-count error (cells counted after watershed vs GT instances)."""
        n_gt = np.array([len(r['gt_ids']) for r in self.results], dtype=np.float64)
        n_pred = np.array([len(r['pred_ids']) for r in self.results], dtype=np.float64)
        err = n_pred - n_gt
        return {
            'mean_gt_per_image': float(n_gt.mean()),
            'mean_pred_per_image': float(n_pred.mean()),
            'mae': float(np.abs(err).mean()),
            'mean_signed_error': float(err.mean()),
            'mean_abs_pct_error': float((np.abs(err) / np.maximum(n_gt, 1)).mean() * 100),
        }

    def per_image_stats(self, iou_thresh=0.5):
        """Raw per-image counts from which the pooled detection, PQ and foreground
        scores can be recomputed for any subset of images (used for paired bootstrap
        comparisons in scripts/compare_runs.py)."""
        stats = []
        for r in self.results:
            matches = greedy_match(r['iou'], iou_thresh)
            gt_fg = r['gt_inst'] > 0
            pred_fg = r['pred_inst'] > 0
            stats.append({
                'n_gt': len(r['gt_ids']),
                'n_pred': len(r['pred_ids']),
                'tp': len(matches),
                'fp': len(r['pred_ids']) - len(matches),
                'fn': len(r['gt_ids']) - len(matches),
                'sum_iou': float(sum(m[2] for m in matches)),
                'fg_tp': int(np.logical_and(gt_fg, pred_fg).sum()),
                'fg_fp': int(np.logical_and(~gt_fg, pred_fg).sum()),
                'fg_fn': int(np.logical_and(gt_fg, ~pred_fg).sum()),
            })
        return stats

    def compute_all(self):
        """Compute all metrics and return as dict."""
        fg = self.compute_pixel_metrics()
        det = self.compute_detection_metrics(iou_thresh=0.5)
        map_scores = self.compute_map()
        pq, pq_legacy = self.compute_panoptic_quality()

        # The notebook's 3-class pixel scores under binary labels: every cell is
        # class 1 ("RBC"), WBC and Platelet never occur and score 0.
        n_legacy = len(LEGACY_CLASS_NAMES)
        legacy_iou = {name: (fg['iou'] if i == 0 else 0.0) for i, name in enumerate(LEGACY_CLASS_NAMES)}
        legacy_dice = {name: (fg['dice'] if i == 0 else 0.0) for i, name in enumerate(LEGACY_CLASS_NAMES)}

        return {
            'task': 'binary',
            'num_images': len(self.results),
            'foreground': fg,
            'detection': det,
            **map_scores,
            'panoptic': pq,
            'counting': self.compute_counting(),
            'legacy': {
                'note': ('Notebook formulas, for tracing v1 numbers only: mean IoU/Dice over RBC/WBC/Platelet '
                         'with every cell counted as RBC (= foreground / 3); non-standard RQ.'),
                'pixel_iou': legacy_iou,
                'pixel_dice': legacy_dice,
                'mean_iou': fg['iou'] / n_legacy,
                'mean_dice': fg['dice'] / n_legacy,
                'panoptic': pq_legacy,
            },
        }
