# ============================================================
# Checkpoints: atomic save, LoRA-aware strict loading
# ============================================================
# New checkpoints store only what training changes: the feature pyramid, the
# NP and HV heads (incl. BatchNorm statistics) and, in Phase 2, the LoRA matrices.
# The frozen UNI weights are re-loaded from the Hugging Face cache instead, which
# keeps checkpoints at ~50 MB (vs 1.3 GB) and avoids redistributing UNI weights
# (CC-BY-NC-ND 4.0) inside checkpoint files.
#
# Old notebook checkpoints (full state dict, ImageNet ViT backbone, 4-class NC
# head) still load for evaluation; their NC head is dropped explicitly.
import os

import torch

from .config import LEGACY_NC_PREFIX
from .lora import LoRALinear, apply_lora, has_lora
from .model import CellSegModel

BACKBONE_PREFIX = 'backbone.'


def is_backbone_frozen_key(key):
    return key.startswith(BACKBONE_PREFIX) and 'lora_' not in key


def trainable_state_dict(model):
    """State dict minus the frozen backbone weights (keeps LoRA A/B and all decoder buffers)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not is_backbone_frozen_key(k)}


def atomic_save(obj, path):
    """torch.save to <path>.tmp then os.replace — an interrupted write never corrupts `path`."""
    tmp_path = str(path) + '.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def checkpoint_lora_config(state_dict, default_alpha=4.0, saved=None):
    """Return {'rank', 'alpha'} if the state dict contains LoRA tensors, else None."""
    a_keys = [k for k in state_dict if k.endswith('lora_A')]
    if not a_keys:
        return None
    rank = state_dict[a_keys[0]].shape[0]
    alpha = (saved or {}).get('alpha', default_alpha)
    return {'rank': int(rank), 'alpha': float(alpha)}


def has_legacy_nc_head(state_dict):
    """True for checkpoints of the notebook's 3-head model (NP + HV + 4-class NC)."""
    return any(k.startswith(LEGACY_NC_PREFIX) for k in state_dict)


def infer_backbone_source(state_dict):
    """Old checkpoints carry the whole backbone; UNI's ViT has LayerScale (`ls1`), timm's ImageNet ViT-L does not."""
    backbone_keys = [k for k in state_dict if is_backbone_frozen_key(k)]
    if not backbone_keys:
        return None
    return 'uni' if any('.ls1.' in k for k in backbone_keys) else 'imagenet'


def load_model_weights(model, state_dict, lora_cfg=None, strict_backbone=None):
    """Load a (trainable-only or full) state dict and fail on anything unexpected.

    - Applies LoRA first when the checkpoint has LoRA tensors and the model does not.
    - Refuses checkpoints with LoRA when lora_cfg is None (never silently drops adapters —
      the notebook's `strict=False` load did exactly that during evaluation).
    - Missing keys are only allowed for frozen backbone weights, and only when the
      checkpoint carries no backbone weights at all.
    """
    ckpt_lora = checkpoint_lora_config(state_dict, saved=lora_cfg)
    if ckpt_lora and not has_lora(model):
        apply_lora(model, rank=ckpt_lora['rank'], alpha=ckpt_lora['alpha'], verbose=False)
    elif has_lora(model) and not ckpt_lora:
        raise RuntimeError("Model has LoRA adapters but the checkpoint has none — load before apply_lora().")
    elif ckpt_lora:
        # Model already has adapters (resume): rank/alpha must match the checkpoint's,
        # otherwise the trained B@A update would be silently rescaled.
        model_cfgs = {(m.rank, float(m.alpha)) for m in model.modules() if isinstance(m, LoRALinear)}
        expected = (ckpt_lora['rank'], ckpt_lora['alpha'])
        alpha_known = lora_cfg is not None and 'alpha' in lora_cfg
        if any(r != expected[0] or (alpha_known and a != expected[1]) for r, a in model_cfgs):
            raise RuntimeError(f"Checkpoint LoRA (rank, alpha)={expected} != model {sorted(model_cfgs)}; "
                               f"re-run with --lora-rank {expected[0]} --lora-alpha {expected[1]} or --no-resume.")

    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys in checkpoint: {result.unexpected_keys[:10]} ...")
    ckpt_has_backbone = any(is_backbone_frozen_key(k) for k in state_dict)
    if strict_backbone is None:
        strict_backbone = ckpt_has_backbone
    bad_missing = [k for k in result.missing_keys
                   if strict_backbone or not is_backbone_frozen_key(k)]
    if bad_missing:
        raise RuntimeError(f"Checkpoint is missing trainable keys: {bad_missing[:10]} ...")
    return ckpt_lora


def build_model_from_checkpoint(ckpt, device, uni_weights=None, verbose=True):
    """Rebuild the exact model a checkpoint was trained with and load it strictly.

    Returns (model, backbone source, LoRA config, whether a legacy NC head was dropped).
    """
    state = ckpt['model_state_dict']
    meta = ckpt.get('backbone') or {}
    if meta.get('smoke_random_init'):
        raise RuntimeError('This checkpoint was trained on a randomly initialised backbone '
                           '(--smoke-random-backbone, a plumbing test); it cannot be evaluated.')
    inferred = infer_backbone_source(state)
    source = meta.get('name') or inferred or 'uni'
    full = inferred is not None   # checkpoint carries backbone weights itself

    # The NC head never fed the instance map (NP + HV + watershed do), so dropping it
    # leaves every instance, detection and foreground metric unchanged.
    legacy_nc = has_legacy_nc_head(state)
    if legacy_nc:
        state = {k: v for k, v in state.items() if not k.startswith(LEGACY_NC_PREFIX)}

    model = CellSegModel(backbone_source=source,
                         pretrained_backbone=not full, freeze_backbone=True,
                         uni_weights=uni_weights, verbose=verbose)
    lora_cfg = load_model_weights(model, state, lora_cfg=ckpt.get('lora'))
    model.to(device).eval()
    if verbose:
        lora_str = f"LoRA rank={lora_cfg['rank']}" if lora_cfg else 'no LoRA'
        print(f"Rebuilt model: backbone={source} ({'weights from checkpoint' if full else 'pretrained weights'}), "
              f"{lora_str}, epoch={ckpt.get('epoch', '?')}")
        if legacy_nc:
            print("  legacy checkpoint: dropped its 4-class NC head (binary labels; it never "
                  "affected the instance map)")
    return model, source, lora_cfg, legacy_nc
