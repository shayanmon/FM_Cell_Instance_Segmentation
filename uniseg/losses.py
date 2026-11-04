# ============================================================
# Multi-task loss (notebook Cell 27)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Loss for the 2-branch (NP + HV) HoVer-Net decoder.

    L_total = w_np * (BCE + Dice) + w_hv * (MSE + MSGE)

    The notebook added w_nc * (CE + Dice) for a 4-class NC head; with binary
    masks that term was a second copy of the NP objective and is gone.
    """

    def __init__(self, weight_np=1.0, weight_hv=2.0):
        super().__init__()
        self.w_np = weight_np
        self.w_hv = weight_hv

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

        # Sobel kernels for MSGE (registered as buffers so they move with .to())
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    @staticmethod
    def dice_loss(pred, target, smooth=1.0):
        pred_flat = pred.contiguous().view(-1)
        tgt_flat = target.contiguous().view(-1)
        inter = (pred_flat * tgt_flat).sum()
        return 1 - (2. * inter + smooth) / (pred_flat.sum() + tgt_flat.sum() + smooth)

    def msge_loss(self, pred_hv, target_hv, focus_mask):
        """Mean Squared Gradient Error — sharpens boundary predictions."""
        loss = torch.tensor(0.0, device=pred_hv.device)
        mask = focus_mask.unsqueeze(1).float()
        num_pixels = mask.sum().clamp(min=1)

        for ch in range(2):
            p = pred_hv[:, ch:ch+1]
            t = target_hv[:, ch:ch+1]
            # Sobel gradients
            pgx = F.conv2d(p, self.sobel_x, padding=1)
            pgy = F.conv2d(p, self.sobel_y, padding=1)
            tgx = F.conv2d(t, self.sobel_x, padding=1)
            tgy = F.conv2d(t, self.sobel_y, padding=1)
            # Masked MSE on gradients
            loss = loss + ((pgx - tgx) ** 2 * mask).sum() / num_pixels
            loss = loss + ((pgy - tgy) ** 2 * mask).sum() / num_pixels

        return loss / 4.0

    def forward(self, preds, targets):
        # --- NP branch ---
        np_bce = self.bce(preds['np'], targets['np_map'])
        np_dice = self.dice_loss(torch.sigmoid(preds['np']), targets['np_map'])
        np_loss = np_bce + np_dice

        # --- HV branch (masked to cell regions) ---
        focus = (targets['np_map'].squeeze(1) > 0.5)
        mask_f = focus.unsqueeze(1).float()
        hv_mse = self.mse(preds['hv'] * mask_f, targets['hv_map'] * mask_f)
        hv_msge = self.msge_loss(preds['hv'], targets['hv_map'], focus)
        hv_loss = hv_mse + hv_msge

        total = self.w_np * np_loss + self.w_hv * hv_loss

        return {
            'total': total,
            'np_loss': np_loss.detach(),
            'hv_loss': hv_loss.detach(),
        }
