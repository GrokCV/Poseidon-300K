import torch
import torch.nn as nn
import torch.nn.functional as F


from mmdet.registry import MODELS

@MODELS.register_module()
class HistLoss(nn.Module):
    def __init__(self,
                 loss_type='mse',
                 num_bins = 256,
                 loss_weight=0.05,
                 activated=False):
        super().__init__()
        assert loss_type in ['l1', 'mse']
        self.num_bins = num_bins
        self.loss_type = loss_type
        self.loss_weight = loss_weight

    @staticmethod
    def _min_max_per_sample_channel(x):
        # x: [B, C, H, W] → per (B,C) 的 min/max，避免跨通道干扰
        B, C, H, W = x.shape
        x_ = x.view(B, C, -1)
        x_min = x_.min(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        x_max = x_.max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        return x_min, x_max
    def _interp_lut(self, x01, bins, lut):
        """
        手写线性插值（支持 autograd，兼容老 PyTorch）
        x01: [B,C,H,W] in [0,1]
        bins/lut: [K]
        将输入数据 x01 映射到查找表（LUT）中
        """
        K = bins.numel()
        pos = x01 * (K - 1)
        idx_lo = torch.clamp(pos.floor().long(), 0, K-1)
        idx_hi = torch.clamp(idx_lo + 1, 0, K-1)
        w_hi = (pos - idx_lo).to(lut.dtype)
        v_lo = torch.take(lut, idx_lo)
        v_hi = torch.take(lut, idx_hi)
        # v_lo = lut[idx_lo]
        # v_hi = lut[idx_hi]
        #return (1 - w_hi) * v_lo + w_hi * v_hi
        return torch.lerp(v_lo, v_hi, w_hi)

    def forward(self, input_img, hist_img, reduction='mean'):
        K = self.num_bins
        B, C, H, W = input_img.shape
        device, dtype = input_img.device, input_img.dtype
        x_min, x_max = self._min_max_per_sample_channel(input_img)
        input_img_min_max = (input_img - x_min) / (x_max - x_min + 1e-6)
        input_flat = input_img.view(B, C, -1)
        hist_flat = hist_img.view(B, C, -1)
        cdf_avg = torch.linspace(0, 1, K,device=device, dtype=dtype).view(1,1,K)
        input_img_interp = self._interp_lut(input_img_min_max, cdf_avg, cdf_avg)
        out = input_img_interp * (x_max - x_min) + x_min

        # normalize to [0,1] 方便直方图比较
        input_flat = (input_flat - input_flat.min(dim=-1, keepdim=True)[0]) / \
                     (input_flat.max(dim=-1, keepdim=True)[0] - input_flat.min(dim=-1, keepdim=True)[0] + 1e-6)
        hist_flat = (hist_flat - hist_flat.min(dim=-1, keepdim=True)[0]) / \
                    (hist_flat.max(dim=-1, keepdim=True)[0] - hist_flat.min(dim=-1, keepdim=True)[0] + 1e-6)

        if self.loss_type == 'l1':
            loss = F.l1_loss(hist_img, out, reduction=reduction)
        else:
            loss = F.mse_loss(hist_img, input_img, reduction=reduction)
        return loss * self.loss_weight