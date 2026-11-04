"""
Verify that the real UNI weights are available before training.

    python scripts/check_uni_access.py [--uni-weights /path/to/pytorch_model.bin]

Downloads (or reuses the cached) MahmoodLab/UNI pytorch_model.bin, loads it into
the UNI ViT-L/16 architecture with strict key matching, runs a forward pass
through the four feature taps used by the decoder, and prints the Hub revision
and SHA-256 of the weight file. Exits non-zero, with instructions, if UNI is
not accessible — training never falls back to another backbone.
"""
import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from uniseg.model import UNIAccessError, UNIBackbone, resolve_uni_weights


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--uni-weights', default=None)
    args = p.parse_args()

    t0 = time.time()
    try:
        path, meta = resolve_uni_weights(args.uni_weights)
    except UNIAccessError as e:
        print(f'UNI ACCESS FAILED\n{e}', file=sys.stderr)
        sys.exit(1)
    print(f'UNI weights: {path}')
    print(f'  revision : {meta["revision"]}')
    print(f'  size     : {meta["size_bytes"] / 1e9:.2f} GB')
    print(f'  sha256   : {sha256(path)}')

    backbone = UNIBackbone(source='uni', pretrained=True, frozen=True, weights_path=args.uni_weights)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    backbone.to(device)
    with torch.no_grad():
        feats = backbone(torch.randn(2, 3, 224, 224, device=device))
    shapes = [tuple(f.shape) for f in feats]
    assert shapes == [(2, 1024, 14, 14)] * 4, shapes
    print(f'  features : blocks {backbone.EXTRACT_INDICES} -> {shapes[0]} x4')
    print(f'UNI OK ({time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
