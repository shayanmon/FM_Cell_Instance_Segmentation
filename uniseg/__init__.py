"""Cell instance segmentation on BCCD with the UNI pathology foundation model.

Script/package port of the original `cell_instance_segmentation.ipynb`
(v1, in the first commits of this repository). The backbone taps, DPT decoder, NP/HV branches
and their losses, augmentation recipe and watershed post-processing follow the
notebook. What changed:
  - the task is binary instance segmentation: the BCCD masks only mark cells,
    so the notebook's 4-class NC head and its CE + Dice loss term were removed
    (NP + HV only; loss values are not comparable with the notebook's);
  - the UNI backbone is loaded strictly (no silent ImageNet fallback);
  - training runs under torchrun / DistributedDataParallel and can resume;
  - evaluation applies LoRA, clips HV instead of tanh, and reports standard PQ.
"""

__version__ = "2.0.0"
