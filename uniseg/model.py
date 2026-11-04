# ============================================================
# Model: UNI backbone -> DPT feature pyramid -> HoVer-Net heads
# (notebook Cells 20-22)
# ============================================================
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EXTRACT_INDICES, UNI_REPO_ID, UNI_WEIGHTS_FILE

BACKBONE_CHOICES = ('uni', 'imagenet')


class UNIAccessError(RuntimeError):
    pass


def build_uni_vit():
    """UNI v1 architecture (MahmoodLab/UNI config.json): ViT-L/16, LayerScale init 1e-5, no head."""
    return timm.create_model(
        'vit_large_patch16_224',
        pretrained=False,
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )


def resolve_uni_weights(weights_path=None):
    """Return (local path to UNI pytorch_model.bin, metadata dict).

    Uses an explicit local file if given, otherwise downloads (or reuses the
    cached copy of) the gated MahmoodLab/UNI weights from the Hugging Face Hub.
    """
    if weights_path:
        path = Path(weights_path)
        if not path.is_file():
            raise UNIAccessError(f"--uni-weights file not found: {path}")
        return path, {'name': 'uni', 'repo_id': UNI_REPO_ID, 'revision': None,
                      'weights_file': str(path), 'size_bytes': path.stat().st_size}

    from huggingface_hub import hf_hub_download
    try:
        path = Path(hf_hub_download(UNI_REPO_ID, UNI_WEIGHTS_FILE))
    except Exception as e:  # gated / not logged in / offline
        raise UNIAccessError(
            f"Could not download UNI weights from https://huggingface.co/{UNI_REPO_ID} ({type(e).__name__}: {e}).\n"
            "  1. Log in to Hugging Face and click 'Agree and access repository' on the UNI model page.\n"
            "  2. Authenticate on this machine: `hf auth login` (or set HF_TOKEN).\n"
            "  3. Or pass --uni-weights /path/to/pytorch_model.bin if you already have the file.\n"
            "There is deliberately no silent fallback to an ImageNet backbone."
        ) from e
    # Cache layout: .../models--MahmoodLab--UNI/snapshots/<commit sha>/pytorch_model.bin
    return path, {'name': 'uni', 'repo_id': UNI_REPO_ID, 'revision': path.parent.name,
                  'weights_file': UNI_WEIGHTS_FILE, 'size_bytes': path.stat().st_size}


class UNIBackbone(nn.Module):
    """
    Wraps the UNI ViT-L/16 model for multi-scale feature extraction.

    UNI v1: vit_large_patch16_224, embed_dim=1024, 24 blocks, 16 heads.
    For 224x224 input -> 14x14 = 196 patches per layer.

    Extracts features from blocks [5, 11, 17, 23] (0-indexed).
    Each feature: (B, 1024, 14, 14) after spatial reshape.

    source='uni'      -> UNI weights, loaded with strict key matching. Fails loudly
                         if the weights are unavailable (no fallback).
    source='imagenet' -> timm's ImageNet ViT-L/16. Only used explicitly, e.g. to
                         re-evaluate the original run that fell back to it.
    pretrained=False builds the architecture only (weights come from a checkpoint).
    """

    EXTRACT_INDICES = EXTRACT_INDICES

    def __init__(self, source='uni', pretrained=True, frozen=True, weights_path=None, verbose=True):
        super().__init__()
        if source not in BACKBONE_CHOICES:
            raise ValueError(f"Unknown backbone source {source!r}; choose from {BACKBONE_CHOICES}")
        self.source = source
        self.meta = {'name': source}

        if source == 'uni':
            self.model = build_uni_vit()
            if pretrained:
                path, self.meta = resolve_uni_weights(weights_path)
                state = torch.load(path, map_location='cpu', weights_only=True)
                self.model.load_state_dict(state, strict=True)   # raises on any key/shape mismatch
                if verbose:
                    n = sum(p.numel() for p in self.model.parameters())
                    print(f"Loaded UNI backbone from {UNI_REPO_ID} "
                          f"(revision {self.meta['revision']}, {n / 1e6:.1f}M params, strict load).")
        else:
            self.model = timm.create_model('vit_large_patch16_224', pretrained=pretrained,
                                           dynamic_img_size=True)
            self.meta = {'name': 'imagenet', 'timm_model': 'vit_large_patch16_224'}
            if verbose:
                print("Using ImageNet ViT-L/16 backbone (explicitly requested; this is NOT UNI).")

        self.embed_dim = self.model.embed_dim  # 1024
        self.patch_size = 16

        if frozen:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x):
        """
        Args:  x: (B, 3, H, W)
        Returns: list of 4 tensors, each (B, 1024, H/16, W/16)
        """
        return self.model.forward_intermediates(
            x,
            indices=self.EXTRACT_INDICES,
            intermediates_only=True,
        )

    def train(self, mode=True):
        # Keep backbone in eval mode always (frozen LayerNorm; LoRA has no dropout)
        super().train(mode)
        self.model.eval()
        return self


# =======================================================================
# Dense Prediction Transformer-Style Multi-Resolution Feature Pyramid
# =======================================================================

class Reassemble(nn.Module):
    """Project ViT features and upsample to a target resolution."""

    def __init__(self, in_channels=1024, out_channels=256, upsample_factor=1):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        layers = []
        current = 1
        while current < upsample_factor:
            layers.extend([
                nn.ConvTranspose2d(out_channels, out_channels,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ])
            current *= 2
        self.upsample = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x):
        return self.upsample(self.project(x))


class FusionBlock(nn.Module):
    """RefineNet-style residual fusion: upsample coarse + add fine + refine."""

    def __init__(self, channels=256):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, coarse, fine):
        coarse_up = F.interpolate(coarse, size=fine.shape[2:],
                                  mode='bilinear', align_corners=False)
        fused = coarse_up + fine
        return fused + self.refine(fused)


class FeaturePyramid(nn.Module):
    """
    Creates a multi-scale feature pyramid from 4 same-resolution ViT features.

    Layer 23 (deepest)  -> 14x14   (no upsample)
    Layer 17            -> 28x28   (2x upsample)
    Layer 11            -> 56x56   (4x upsample)
    Layer 5  (earliest) -> 112x112 (8x upsample)

    Then fuses coarse-to-fine using RefineNet blocks.
    Output: (B, 256, 112, 112)
    """
    PYRAMID_CHANNELS = 256

    def __init__(self, embed_dim=1024):
        super().__init__()
        C = self.PYRAMID_CHANNELS
        # Reassemble: deepest first
        self.reassemble = nn.ModuleList([
            Reassemble(embed_dim, C, upsample_factor=1),  # 14x14
            Reassemble(embed_dim, C, upsample_factor=2),  # 28x28
            Reassemble(embed_dim, C, upsample_factor=4),  # 56x56
            Reassemble(embed_dim, C, upsample_factor=8),  # 112x112
        ])
        self.fusions = nn.ModuleList([
            FusionBlock(C),  # 14->28
            FusionBlock(C),  # 28->56
            FusionBlock(C),  # 56->112
        ])

    def forward(self, features):
        """
        Args: features = [layer5, layer11, layer17, layer23], each (B, 1024, 14, 14)
        Returns: (B, 256, 112, 112)
        """
        # Reassemble in reverse order (deepest first)
        r = [self.reassemble[i](features[3 - i]) for i in range(4)]
        # r[0]=(B,256,14,14), r[1]=(B,256,28,28), r[2]=(B,256,56,56), r[3]=(B,256,112,112)

        fused = r[0]
        for i, fusion in enumerate(self.fusions):
            fused = fusion(fused, r[i + 1])
        return fused  # (B, 256, 112, 112)


# ============================================================
# Decoder Heads + Full Model
# ============================================================

class DecoderHead(nn.Module):
    """
    Takes fused features (B, 256, 112, 112) and produces output at
    original resolution (B, out_ch, 224, 224).
    """

    def __init__(self, in_channels=256, mid_channels=128, out_channels=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.head(x)


class CellSegModel(nn.Module):
    """
    Complete cell instance segmentation model.

    UNI ViT-L/16 -> Feature Pyramid -> 2 decoder heads (HoVer-Net without the type branch):
      - np_head: (B, 1, 224, 224)  binary cell-vs-background logits
      - hv_head: (B, 2, 224, 224)  h/v distance to each cell's centroid, which
                                   the watershed uses to split touching cells

    Trainable params (backbone frozen): ~11.8M
    Frozen params (backbone):           ~304M
    """

    def __init__(self, backbone_source='uni', pretrained_backbone=True,
                 freeze_backbone=True, uni_weights=None, verbose=True):
        super().__init__()
        self.backbone = UNIBackbone(source=backbone_source, pretrained=pretrained_backbone,
                                    frozen=freeze_backbone, weights_path=uni_weights, verbose=verbose)
        self.feature_pyramid = FeaturePyramid(embed_dim=self.backbone.embed_dim)

        C = FeaturePyramid.PYRAMID_CHANNELS  # 256
        self.np_head = DecoderHead(C, 128, 1)            # Binary segmentation
        self.hv_head = DecoderHead(C, 128, 2)            # H/V distance regression

    def forward(self, x):
        features = self.backbone(x)
        fused = self.feature_pyramid(features)
        return {
            'np': self.np_head(fused),
            'hv': self.hv_head(fused),
        }

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {'trainable': trainable, 'frozen': frozen, 'total': trainable + frozen}
