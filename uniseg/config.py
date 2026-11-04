# ============================================================
# Shared constants (were the Cell 2 / Cell 4 globals in the notebook)
# ============================================================
import numpy as np

INPUT_SIZE = 224
RESIZE_SIZE = 256           # Processed images are 256x256; random/center crop to INPUT_SIZE

# Binary task: the processed BCCD masks only say "cell" vs "background" (no
# RBC/WBC/Platelet labels), so the model predicts a binary cell mask (NP head)
# and separates it into individual cells with the HV distance maps. The
# notebook's 4-class NC head (bg, RBC, WBC, Platelet) was dropped: with binary
# labels it only ever predicted class 1 and duplicated the NP head.
# Old notebook checkpoints still carry that NC head; evaluation ignores it.
LEGACY_NC_PREFIX = 'nc_head.'
LEGACY_CLASS_NAMES = ('RBC', 'WBC', 'Platelet')

# ImageNet normalization constants (UNI was pretrained with these)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ViT blocks tapped for the DPT feature pyramid (1/4, 1/2, 3/4, full depth)
EXTRACT_INDICES = [5, 11, 17, 23]

# UNI v1 on the Hugging Face Hub (gated: accept terms + `hf auth login`)
UNI_REPO_ID = "MahmoodLab/UNI"
UNI_WEIGHTS_FILE = "pytorch_model.bin"
