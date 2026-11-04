# Changelog

## [2.0.0] - UNI backbone, binary model, DDP training

### Changed

- **Notebook ported to a Python package** (`uniseg/`, `train.py`, `evaluate.py`).
- **UNI loaded strictly.** v1 silently fell back to an ImageNet ViT-L/16 when the gated UNI download failed, so the v1 numbers came from ImageNet weights. v2 stops with an error instead.
- **Binary instance segmentation.** The masks only mark cells, so the 4-class cell-type head was removed. The model has NP (cell probability) and HV (distance map) heads.
- **Reference HoVer-Net post-processing by default.** The notebook's watershed merged touching cells; it is still available as `--postproc notebook`.
- **Training under torchrun / DistributedDataParallel** with resumable checkpoints and a fixed global batch of 16.

### Fixed

- LoRA adapters were dropped at evaluation (`strict=False` load).
- Panoptic quality used a non-standard formula that inflated it.
- Two stacked schedulers skipped the Phase 1 warmup.
- Data-loader workers replayed identical augmentation streams.

### Added

- ImageNet ViT-L/16 baseline trained with the same code (`-Backbone imagenet`).
- Paired-bootstrap comparison of two evaluations (`scripts/compare_runs.py`).
- UNI access check (`scripts/check_uni_access.py`).

## [1.0.0]

- Jupyter notebook with UNI + HoVer-Net decoder and LoRA fine-tuning (first commits of this repository).
