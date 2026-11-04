# ============================================================
# Post-Processing — Watershed Instance Separation
# ============================================================
# Two ways to turn the NP (cell probability) and HV (distance) maps into cells:
#
# 'hovernet' (default): the reference HoVer-Net post-processing, a port of
#   __proc_np_hv in https://github.com/vqdang/hover_net (models/hovernet/post_proc.py,
#   MIT licence; Graham et al., Medical Image Analysis 2019) with its fixed values.
#   The SIGNED Sobel gradient of each HV map is min-max normalised and inverted,
#   so a high value means "boundary in this direction"; np.maximum then flags a
#   pixel as boundary when EITHER direction jumps.
#
# 'notebook': the original notebook's version (Cell 35), kept to reproduce v1 and
#   earlier v2 numbers. It inverts the ABSOLUTE gradient, so a high value means
#   "flat in this direction", but still combines the two directions with
#   np.maximum: a pixel counts as cell interior if EITHER direction is flat. A
#   contact between two side-by-side cells is a jump in the horizontal map only,
#   so it is lost and touching cells merge. On the validation split (UNI Phase 2
#   checkpoint) it found 41.6 cells/image against 60.5 in the ground truth;
#   'hovernet' found 55.1 (recall 0.615 -> 0.876, PQ 0.655 -> 0.833).
import cv2
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, label as scipy_label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

POSTPROC_CHOICES = ('hovernet', 'notebook')


def post_process(np_pred, hv_pred, method='hovernet'):
    """
    Convert model outputs into instance segmentation via marker-controlled watershed.

    Args:
        np_pred: (H, W) float [0, 1], cell (foreground) probability
        hv_pred: (H, W, 2) float, h/v distance predictions (decode_outputs clips them to [-1, 1])
        method: 'hovernet' (reference algorithm) or 'notebook' (v1 behaviour)
    Returns:
        instance_map: (H, W) uint16, 0 = background, 1..N = individual cells
    """
    if method == 'hovernet':
        return post_process_hovernet(np_pred, hv_pred)
    if method == 'notebook':
        return post_process_notebook(np_pred, hv_pred)
    raise ValueError(f'unknown post-processing {method!r}; choose from {POSTPROC_CHOICES}')


def post_process_hovernet(np_pred, hv_pred):
    """HoVer-Net reference post-processing (__proc_np_hv), same thresholds and kernel sizes."""
    blb = (np.asarray(np_pred, dtype=np.float32) >= 0.5).astype(np.int32)
    blb = scipy_label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(np.asarray(hv_pred[..., 0], dtype=np.float32), None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(np.asarray(hv_pred[..., 1], dtype=np.float32), None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)
    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    # cells form mountains, so invert to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = (overall >= 0.4).astype(np.int32)
    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = scipy_label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    return watershed(dist, markers=marker, mask=blb).astype(np.uint16)


def post_process_notebook(np_pred, hv_pred, np_thresh=0.5, min_size=10, ksize=21, combine=np.maximum):
    """The notebook's post-processing (see the module comment for why it merges cells).

    `combine` merges the two per-direction flatness maps; np.maximum is the notebook's
    choice, np.minimum was tested only as a diagnostic (see RESULTS.md).
    """
    binary = (np_pred > np_thresh).astype(np.uint8)
    binary = binary_fill_holes(binary).astype(np.uint8)
    binary = remove_small_objects(binary.astype(bool), min_size=min_size).astype(np.uint8)

    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=np.uint16)

    h_map = ((hv_pred[:, :, 0] + 1) / 2).astype(np.float64)
    v_map = ((hv_pred[:, :, 1] + 1) / 2).astype(np.float64)

    h_grad = cv2.Sobel(h_map, cv2.CV_64F, 1, 0, ksize=ksize)
    v_grad = cv2.Sobel(v_map, cv2.CV_64F, 0, 1, ksize=ksize)
    h_grad = 1 - cv2.normalize(np.abs(h_grad), None, 0, 1, cv2.NORM_MINMAX)
    v_grad = 1 - cv2.normalize(np.abs(v_grad), None, 0, 1, cv2.NORM_MINMAX)

    overall = combine(h_grad, v_grad)
    overall = overall - (1 - binary.astype(np.float64))
    overall[overall < 0] = 0

    marker_mask = ((np_pred > 0.7) & (overall > 0.4)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_OPEN, kernel)
    markers, _ = scipy_label(marker_mask)
    for mid in np.unique(markers):
        if mid > 0 and (markers == mid).sum() < 5:
            markers[markers == mid] = 0
    markers, _ = scipy_label(markers > 0)

    if markers.max() == 0:
        markers, _ = scipy_label(binary)

    energy = -cv2.GaussianBlur(overall, (3, 3), 0)
    inst_map = watershed(energy, markers, mask=binary).astype(np.uint16)

    for iid in np.unique(inst_map):
        if iid > 0 and (inst_map == iid).sum() < min_size:
            inst_map[inst_map == iid] = 0

    return inst_map


def decode_outputs(out, hv_activation='clip'):
    """Batch model outputs -> per-image numpy maps (np prob, hv in [-1, 1]).

    The HV head is trained with plain MSE against targets in [-1, 1] (no output
    activation). The notebook applied tanh at inference only, which compresses
    the predicted distances (tanh(1) = 0.76); 'clip' matches training.
    'tanh' reproduces the notebook exactly.
    """
    np_prob = torch.sigmoid(out['np'][:, 0].float())
    hv = out['hv'].float()
    hv = torch.tanh(hv) if hv_activation == 'tanh' else hv.clamp(-1, 1)
    return np_prob.cpu().numpy(), hv.permute(0, 2, 3, 1).cpu().numpy()
