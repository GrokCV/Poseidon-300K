from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
import time
import numpy as np
import math
from mmcv.cnn import ConvModule
class MonotonicLUT1D(nn.Module):
    def __init__(self, num_bins=512, group = 1,hidden=128):
        super().__init__()
        self.num_bins = num_bins
        self.group = group
        self.hidden = hidden
        self.net = nn.Sequential(
            ConvModule(self.group, hidden, 5, padding=2, groups=1, norm_cfg=None),
            ConvModule(hidden, hidden, 5, padding=2, groups=hidden, norm_cfg=None),  # depth-wise
            
            ConvModule(hidden, self.group, 1, norm_cfg=None, act_cfg=None)
        )
        
        self.register_buffer('init_cdf', torch.linspace(0, 1, num_bins))

    def forward(self, hist_cdf):
        delta = self.net(hist_cdf)                      
        delta = F.softplus(delta)                       
        cdf = torch.cumsum(delta, dim=-1)               
        cdf = cdf / (cdf[...,-1:] + 1e-6)               
        identity = self.init_cdf.view(1,1,-1)
        return  cdf +identity
import pandas as pd
@MODELS.register_module()
class LearnableHistEq(nn.Module):
    def __init__(self,
                 
                 num_bins: int = 64,
                 downsample: int = 32,
                 detach_hist: bool = True,
                 group: int = 16,
                 alpha_init: float = 0.25,
                 profile:bool = False,
                 
                 ):
        super().__init__()
        self.num_bins = num_bins
        self.downsample = downsample
        self.detach_hist = detach_hist
        self.profile = profile
        self.group = group
        self.alpha = nn.Parameter(torch.tensor(alpha_init).float())
        self.lut_net = MonotonicLUT1D(num_bins,group)
        
    @staticmethod
    def _min_max_per_sample_channel(x):
        B, C, H, W = x.shape
        x_ = x.view(B, C, -1)
        x_min = x_.min(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        x_max = x_.max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        return x_min, x_max

    def _build_monotonic_lut(self, device, dtype):
        deltas = F.softplus(self.delta).to(device=device, dtype=dtype) 
        cdf = torch.cumsum(deltas, dim=0)    
        cdf = cdf / (cdf[-1] + 1e-6)  # [0,1]
        bins = torch.linspace(0, 1, self.num_bins, device=device, dtype=dtype)
        return bins, cdf  # [K], [K]

    @torch.no_grad()
    def _estimate_cdf_shift(self, x_norm_small, bins):
        B, C, h, w = x_norm_small.shape
        K = bins.numel()
        G = self.group
        x_group = x_norm_small.view(B, G, -1, h, w).mean(dim=2) 
        idx = torch.clamp((x_group * (K - 1)).round().long(), 0, K - 1)  
        hist = torch.zeros(B, G, K, device=x_group.device, dtype=torch.float32)
        hist.scatter_add_(dim=-1, index=idx.view(B, G, -1), src=torch.ones_like(idx, dtype=torch.float32).view(B, G, -1))

        pdf = hist / (hist.sum(dim=-1, keepdim=True) + 1e-6)  
        cdf = torch.cumsum(pdf, dim=-1)  # [B,C,K]
       
        cdf_mean = cdf.mean(dim=(0), keepdim=True) 
       
        delta =cdf_mean.view(1,G,1,K)  
        return delta
        
       
    def _interp_lut(self, x01, bins, lut):
        B, C, H, W = x01.shape
        G = self.group
        K = bins.numel()
    
        pos = x01 * (K - 1)
        idx_lo = torch.clamp(pos.floor().long(), 0, K - 1)
        idx_hi = torch.clamp(idx_lo + 1, 0, K - 1)
        w_hi = (pos - idx_lo).to(lut.dtype)
        group_size = C // G

        device = x01.device
        group_ids = torch.arange(C, device=device) // group_size  
        lut = lut.squeeze(2)
        lut_group = lut[0, group_ids, :] 

        idx_lo_flat = idx_lo.view(B, C, -1)  
        idx_hi_flat = idx_hi.view(B, C, -1)
        lut_expand = lut_group.unsqueeze(0).expand(B, -1, -1) 

        v_lo = lut_expand.gather(2, idx_lo_flat)
        v_hi = lut_expand.gather(2, idx_hi_flat)

        w_hi_flat = w_hi.view(B, C, -1)
        out_flat = torch.lerp(v_lo, v_hi, w_hi_flat)

        out = out_flat.view(B, C, H, W)
        return out
    

    def forward_single(self, x,level):
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype   
        
        i = level
        bins = torch.linspace(0, 1, self.num_bins, device=device, dtype=dtype)
        x_min, x_max = self._min_max_per_sample_channel(x)
        x01 = (x - x_min) / (x_max - x_min + 1e-6)
        if self.downsample is not None and min(H, W) > self.downsample:
            x_small = F.adaptive_avg_pool2d(x01, self.downsample)
        else:
            x_small = x01
        
        if self.detach_hist:
            x_small_detached = x_small.detach()
        else:
            x_small_detached = x_small
       
        with torch.no_grad():
            delta_cdf = self._estimate_cdf_shift(x_small_detached,bins).to(dtype=dtype)  # [1,1,K]
            
        adapt_cdf = self.lut_net(delta_cdf)        
        x_eq01 = self._interp_lut(x01, bins, adapt_cdf)  # [B,C,H,W]
        
        alpha = torch.sigmoid(self.alpha)  # 0~1
        
        out01 = alpha * x_eq01 + (1 - alpha) * x01
        out = out01 * (x_max - x_min) + x_min
        return out

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            outputs = []
            for i, x in enumerate(inputs):
                if i < 4:
                    outputs.append(self.forward_single(x,i))
                else:
                    outputs.append(x)  
            return outputs
        else:
            return self.forward_single(inputs)
    