# ============================================================
# Dataset, augmentation and HV-map targets (notebook Sections 2-3)
# ============================================================
import os
import random
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')   # no network check per DataLoader worker

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import center_of_mass
from skimage.measure import label as skimage_label
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import albumentations as A

from .config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


def generate_hv_map(instance_map):
    """
    Generate horizontal/vertical distance maps from instance segmentation.

    For each instance: compute normalized distance from each pixel to
    the instance centroid, in range [-1, +1].

    Returns: (H, W, 2) float32 array [horizontal, vertical]
    """
    hv_map = np.zeros((*instance_map.shape, 2), dtype=np.float32)
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]

    for inst_id in instance_ids:
        inst_mask = (instance_map == inst_id)
        # Bounding box for efficiency
        rows = np.any(inst_mask, axis=1)
        cols = np.any(inst_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        pad = 2
        rmin_p = max(0, rmin - pad)
        rmax_p = min(instance_map.shape[0], rmax + pad + 1)
        cmin_p = max(0, cmin - pad)
        cmax_p = min(instance_map.shape[1], cmax + pad + 1)

        crop_mask = inst_mask[rmin_p:rmax_p, cmin_p:cmax_p]
        cy, cx = center_of_mass(crop_mask)

        y_grid, x_grid = np.mgrid[0:crop_mask.shape[0], 0:crop_mask.shape[1]]
        x_dist = (x_grid - cx).astype(np.float32)
        y_dist = (y_grid - cy).astype(np.float32)

        x_dist[~crop_mask] = 0
        y_dist[~crop_mask] = 0

        x_valid = x_dist[crop_mask]
        y_valid = y_dist[crop_mask]

        # Normalize to [-1, 1]
        if len(x_valid) > 0:
            x_min_abs = abs(x_valid.min()) if x_valid.min() < 0 else 1
            x_max_abs = x_valid.max() if x_valid.max() > 0 else 1
            x_dist_norm = x_dist.copy()
            neg_mask = crop_mask & (x_dist < 0)
            pos_mask = crop_mask & (x_dist > 0)
            if neg_mask.any() and x_min_abs > 0:
                x_dist_norm[neg_mask] = x_dist[neg_mask] / (x_min_abs + 1e-8)
            if pos_mask.any() and x_max_abs > 0:
                x_dist_norm[pos_mask] = x_dist[pos_mask] / (x_max_abs + 1e-8)
            x_dist_norm[~crop_mask] = 0

            y_min_abs = abs(y_valid.min()) if y_valid.min() < 0 else 1
            y_max_abs = y_valid.max() if y_valid.max() > 0 else 1
            y_dist_norm = y_dist.copy()
            neg_mask = crop_mask & (y_dist < 0)
            pos_mask = crop_mask & (y_dist > 0)
            if neg_mask.any() and y_min_abs > 0:
                y_dist_norm[neg_mask] = y_dist[neg_mask] / (y_min_abs + 1e-8)
            if pos_mask.any() and y_max_abs > 0:
                y_dist_norm[pos_mask] = y_dist[pos_mask] / (y_max_abs + 1e-8)
            y_dist_norm[~crop_mask] = 0

            hv_map[rmin_p:rmax_p, cmin_p:cmax_p, 0] += x_dist_norm
            hv_map[rmin_p:rmax_p, cmin_p:cmax_p, 1] += y_dist_norm

    return hv_map


def relabel_instances(inst_map, min_size=3):
    """Re-split instances into connected components after a geometric transform."""
    new_inst = np.zeros_like(inst_map)
    new_id = 1
    for uid in np.unique(inst_map):
        if uid == 0:
            continue
        region = (inst_map == uid)
        labeled, n = skimage_label(region, return_num=True)
        for j in range(1, n + 1):
            comp = (labeled == j)
            if comp.sum() >= min_size:
                new_inst[comp] = new_id
                new_id += 1
    return new_inst


# ------------------------------------------------------------
# Augmentation (same recipe as the notebook)
# ------------------------------------------------------------
# The notebook used the albumentations 1.3 argument names (value=, var_limit=,
# max_holes=...). Those are spelled out here with the equivalent current
# arguments so the behaviour is identical:
#   GaussNoise(var_limit=(10, 50)) on uint8  -> std in [sqrt(10), sqrt(50)] / 255
#   CoarseDropout(max_holes=8, max_height=16, max_width=16) -> exactly 8 holes of 16x16

def build_geometric_transform():
    """Applied jointly to image and instance_map. HV maps must be recomputed afterwards."""
    return A.Compose([
        A.RandomCrop(INPUT_SIZE, INPUT_SIZE),   # 256 -> 224
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7
        ),
        A.OneOf([
            A.ElasticTransform(alpha=80, sigma=80 * 0.05, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ], p=0.3),
    ])


def build_photometric_transform():
    """Applied to the image only."""
    return A.Compose([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(std_range=(np.sqrt(10.0) / 255.0, np.sqrt(50.0) / 255.0), p=0.3),
        A.CoarseDropout(num_holes_range=(8, 8), hole_height_range=(16, 16),
                        hole_width_range=(16, 16), fill=0, p=0.3),
    ])


def build_val_transform():
    """Validation/test: deterministic center crop only."""
    return A.Compose([
        A.CenterCrop(INPUT_SIZE, INPUT_SIZE),
    ])


class BCCDInstanceDataset(Dataset):
    """
    PyTorch Dataset for BCCD cell instance segmentation.

    Expects the notebook's processed layout:
        <data_dir>/images/*.png, instance_maps/*.npy (uint16)
    (the class_maps/*.npy files it also wrote are not read: they hold only 0/1 and
    equal instance_map > 0 except for 56 isolated pixels across all 1,328 images)

    Returns:
        image: (3, H, W) float32 tensor, ImageNet-normalized
        targets: dict with:
            'np_map':       (1, H, W) binary cell mask (instance_map > 0)
            'hv_map':       (2, H, W) horizontal/vertical distance maps
            'instance_map': (H, W) int tensor (for eval only, not used in loss)
    """

    def __init__(self, data_dir, is_train=True, verbose=True):
        self.data_dir = Path(data_dir)
        self.is_train = is_train

        self.image_files = sorted(self.data_dir.glob('images/*.png'))
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.data_dir / 'images'}")
        if verbose:
            print(f"BCCDInstanceDataset: {len(self.image_files)} samples from {data_dir}")

        if is_train:
            self.geometric_transform = build_geometric_transform()
            self.photometric_transform = build_photometric_transform()
        else:
            self.val_transform = build_val_transform()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        stem = self.image_files[idx].stem

        # Load pre-processed instance map
        img = np.array(Image.open(self.image_files[idx]).convert('RGB'))      # (H, W, 3) uint8
        inst_map = np.load(self.data_dir / 'instance_maps' / f'{stem}.npy')  # (H, W) uint16

        if self.is_train:
            # Phase 1: Geometric transforms (invalidates HV maps)
            augmented = self.geometric_transform(image=img, mask=inst_map.astype(np.int32))
            img = augmented['image']
            inst_map = augmented['mask'].astype(np.uint16)

            # Re-label instances after geometric transform
            inst_map = relabel_instances(inst_map)

            # Recompute HV map from transformed instance map
            hv_map = generate_hv_map(inst_map)

            # Phase 2: Photometric transforms (image only)
            img = self.photometric_transform(image=img)['image']
        else:
            # Validation: deterministic center crop
            augmented = self.val_transform(image=img, mask=inst_map.astype(np.int32))
            img = augmented['image']
            inst_map = augmented['mask'].astype(np.uint16)

            # Recompute HV map for the cropped region
            hv_map = generate_hv_map(inst_map)

        # Image: normalize to ImageNet stats
        img_float = img.astype(np.float32) / 255.0
        img_float = (img_float - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)  # (3, H, W)

        np_map = (inst_map > 0).astype(np.float32)
        targets = {
            'np_map': torch.from_numpy(np_map).unsqueeze(0),                  # (1, H, W)
            'hv_map': torch.from_numpy(hv_map).permute(2, 0, 1),               # (2, H, W)
            'instance_map': torch.from_numpy(inst_map.astype(np.int32)),      # (H, W)
        }
        return img_tensor, targets


class ShardedEvalSampler(Sampler):
    """Deterministic, non-overlapping split for validation: rank r gets indices r, r+W, r+2W, ...

    Unlike DistributedSampler this never pads with duplicates, so per-sample
    sums all-reduced across ranks give the exact validation loss.
    """

    def __init__(self, dataset_len, rank=0, world_size=1):
        self.indices = list(range(rank, dataset_len, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def reseed_augmentations(dataset, seed):
    """Give the dataset's augmentation pipelines their own seed-derived RNG streams.

    albumentations 1.4.x keeps private generators inside every transform, created
    from OS entropy when the pipeline is built and then copied unchanged into each
    DataLoader worker. Reseeding the global `random`/`np.random` does not reach
    them, so without this every worker would replay identical augmentations.
    """
    children = np.random.SeedSequence(seed).spawn(2)
    for name, child in zip(('geometric_transform', 'photometric_transform'), children):
        transform = getattr(dataset, name, None)
        if transform is not None:
            s = int(child.generate_state(1)[0])
            transform.set_random_state(np.random.default_rng(s), random.Random(s))


def seed_worker(worker_id):
    """Give every DataLoader worker (on every rank) its own random streams.

    torch.initial_seed() in a worker is base_seed + worker_id, where base_seed
    comes from the loader's generator (seeded with seed + rank).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        reseed_augmentations(info.dataset, worker_seed)


def build_train_loader(dataset, micro_batch, dist_env, num_workers, seed):
    if num_workers == 0:
        reseed_augmentations(dataset, seed + dist_env.rank)   # main-process loading
    sampler = None
    if dist_env.active:
        sampler = DistributedSampler(dataset, num_replicas=dist_env.world_size, rank=dist_env.rank,
                                     shuffle=True, seed=seed, drop_last=True)
    generator = torch.Generator()
    generator.manual_seed(seed + dist_env.rank)
    return DataLoader(
        dataset, batch_size=micro_batch,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=generator,
        persistent_workers=num_workers > 0,
    )


def build_eval_loader(dataset, batch_size, dist_env, num_workers):
    sampler = ShardedEvalSampler(len(dataset), dist_env.rank, dist_env.world_size) if dist_env.active else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        worker_init_fn=seed_worker, persistent_workers=num_workers > 0,
    )
