# ============================================================
# LoRA for Phase 2 Fine-Tuning (notebook Cell 23)
# ============================================================
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation for nn.Linear.
    Output = Wx + (alpha/rank) * B @ A @ x

    Only A and B are trainable; original W is frozen.
    State-dict keys (`...qkv.original.weight`, `...qkv.lora_A`, `...qkv.lora_B`)
    match the notebook so old checkpoints stay loadable.
    """

    def __init__(self, original_linear, rank=4, alpha=4.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha

        in_f = original_linear.in_features
        out_f = original_linear.out_features

        # Freeze original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # Trainable low-rank matrices (B=0 -> adapter starts as an exact no-op)
        device = original_linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device))

    def forward(self, x):
        orig_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return orig_out + (self.alpha / self.rank) * lora_out


def has_lora(model):
    return any(isinstance(m, LoRALinear) for m in model.modules())


def apply_lora(model, rank=4, alpha=4.0, verbose=True):
    """Apply LoRA to all attention QKV projections in the backbone."""
    lora_params = 0
    blocks = model.backbone.model.blocks
    for block in blocks:
        original_qkv = block.attn.qkv
        lora_layer = LoRALinear(original_qkv, rank=rank, alpha=alpha)
        block.attn.qkv = lora_layer
        lora_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
    if verbose:
        print(f"Applied LoRA (rank={rank}, alpha={alpha}) to {len(blocks)} blocks. "
              f"Added {lora_params:,} trainable parameters.")
    return model
