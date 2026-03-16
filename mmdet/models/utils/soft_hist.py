import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import numpy as np
from mmdet.registry import MODELS

@MODELS.register_module()
class FeatureHE(nn.Module):
    """
    可微直方图均衡化（Soft Histogram Equalization）
    - 对每个通道的特征做 0-1 归一化 → soft-hist → CDF → 核回归式映射 → 还原到原范围
    - 使用高斯核对 CDF 做加权和，避免离散索引带来的不可导问题
    """
    def __init__(self,
                 num_bins: int = 64,
                 init_sigma: float = 0.02,
                 init_tau: float = 0.02,
                 alpha_init: float = 0.0,
                 per_level_alpha: bool = False):
        super().__init__()
        assert num_bins >= 8, 'num_bins 太小会不稳定，建议 >= 32'
        self.num_bins = num_bins

        # 固定 bin 位置（[0,1] 均匀），作为 buffer 不参与梯度
        self.register_buffer('bins', torch.linspace(0., 1., num_bins))

        # 用可学习正参数（softplus）保证数值稳定
        self.log_sigma = nn.Parameter(torch.log(torch.exp(torch.tensor(init_sigma)) - 1.0))
        self.log_tau   = nn.Parameter(torch.log(torch.exp(torch.tensor(init_tau)) - 1.0))

        # 可学习的增强强度 α：共享或分层
        self.per_level_alpha = per_level_alpha
        if per_level_alpha:
            # 延后在第一次 forward 根据层数初始化
            self.alpha_list = nn.ParameterList()
            self._alpha_ready = False
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

#正确初始化alpha_list
    def _ensure_alpha_list(self, num_levels: int, device):
        if getattr(self, '_alpha_ready', False):   #如果 _alpha_ready 为 False，则继续执行初始化操作
            return
        for _ in range(num_levels):   #每一层初始化一个 alpha 参数。
            self.alpha_list.append(nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device)))
        self._alpha_ready = True

    @staticmethod
    def _minmax_norm(x):
        B, C, H, W = x.shape
        xv = x.view(B, C, -1)
        x_min = xv.min(dim=-1, keepdim=True)[0]
        x_max = xv.max(dim=-1, keepdim=True)[0]
        x_norm = (xv - x_min) / (x_max - x_min + 1e-6)
        return x_norm.view(B, C, H, W), x_min, x_max
    
    # def _soft_hist(self, x_norm, bins, sigma):
    #     """
    #     x_norm: [B,C,H,W] in [0,1]
    #     返回每个通道的 soft 直方图 [B,C,num_bins]
    #     """
    #     B, C, H, W = x_norm.shape
    #     x_flat = x_norm.view(B, C, -1)                # [B,C,N]
    #     # 高斯加权到各个 bin
    #     # 计算距离并做核密度加权求和
    #     # w_ik = exp(- (x_i - b_k)^2 / (2 sigma^2))
    #     dist2 = (x_flat.unsqueeze(-1) - bins.view(1, 1, 1, -1)) ** 2    # [B,C,N,K]
    #     w = torch.exp(-0.5 * dist2 / (sigma * sigma + 1e-12))           # [B,C,N,K]相当于每个像素点对bins的权重
    #     hist = w.sum(dim=2)                                             # [B,C,K]把权重求和，得到每个 bins 的频数
    #     hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-6)           #得到每个 bins 的频率
    #     return hist
    
    def _soft_hist(self, x_norm, bins, sigma, chunk_size=512):
        """
        x_norm: [B,C,H,W] in [0,1]
        返回每个通道的 soft 直方图 [B,C,num_bins]（支持分块计算以节省显存）
        """
        B, C, H, W = x_norm.shape
        N = H * W
        x_flat = x_norm.view(B, C, -1)  # [B, C, N]
        K = bins.shape[0]
        hist_total = torch.zeros(B, C, K, device=x_norm.device, dtype=x_norm.dtype)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x_flat[:, :, start:end]  # [B, C, chunk]

            # [B, C, chunk, K] - 高斯核权重
            dist2 = (x_chunk.unsqueeze(-1) - bins.view(1, 1, 1, -1)) ** 2
            w_chunk = torch.exp(-0.5 * dist2 / (sigma * sigma + 1e-12))

            # 求和到 hist_total
            hist_total += w_chunk.sum(dim=2)  # 累加每个 bin 的权重

        # 归一化成频率
        hist_total = hist_total / (hist_total.sum(dim=-1, keepdim=True) + 1e-6)
        return hist_total
    def _cdf_from_hist(self, hist):
        cdf = torch.cumsum(hist, dim=-1)                                # [B,C,K]得到每个bins 的累计概率密度
        # 归一化到 [0,1]
        cdf = (cdf - cdf[..., :1]) / (cdf[..., -1:] - cdf[..., :1] + 1e-6)
        return cdf
    '''
    def _kernel_regression_map(self, x_norm, bins, cdf, tau):
        """
        用核回归方式把每个像素映射到 cdf 上：
        y(x) = sum_k soft_w_k(x) * cdf_k
        soft_w_k(x) 对 x 连续可导（高斯核 + 归一化）
        """
        B, C, H, W = x_norm.shape
        x_flat = x_norm.view(B, C, -1)                                  # [B,C,N]
        dist2 = (x_flat.unsqueeze(-1) - bins.view(1,1,1,-1)) ** 2       # [B,C,N,K]
        w = torch.exp(-0.5 * dist2 / (tau * tau + 1e-12))               # [B,C,N,K]
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)                    # softmax-like让每个像素的所有 bin 权重和为 1
        y_flat = (w * cdf.unsqueeze(2)).sum(dim=-1)                     # [B,C,N]得到每个像素位置的 CDF 值
        return y_flat.view(B, C, H, W)
        '''   
    def _kernel_regression_map(self, x_norm, bins, cdf):
        B, C, H, W = x_norm.shape
        x_flat = x_norm.view(B, C, -1)
        x_interp = torch.zeros_like(x_flat)
        bins_cpu = bins.cpu().numpy()
        
        for b in range(B):
            for c in range(C):
                x_flat_cpu = x_flat[b, c].detach().cpu().numpy()  # 转移到 CPU 并转换为 NumPy 数组
                cdf_cpu = cdf[b, c].detach().cpu().numpy() 
                x_interp[b, c] = torch.tensor(np.interp(x_flat_cpu, bins_cpu, cdf_cpu), device=x_norm.device)
                #x_interp[b, c] = np.interp(x_flat[b, c], bins, cdf[b, c])
        return x_interp.view(B, C, H, W)
    def forward_single(self, x, alpha_param):
        # 1) 归一化
        x_norm, x_min, x_max = self._minmax_norm(x)
        # 2) soft-hist & CDF
        sigma = F.softplus(self.log_sigma) + 1e-6
        #tau   = F.softplus(self.log_tau)   + 1e-6
        bins  = self.bins.to(x.device, dtype=x.dtype)
        hist  = self._soft_hist(x_norm, bins, sigma)
        cdf   = self._cdf_from_hist(hist)
        # 3) 连续可导的核回归映射
        x_eq  = self._kernel_regression_map(x_norm, bins, cdf)
        # 4) 还原到原动态范围
        B, C, H, W = x.shape
        x_eq = x_eq * (x_max - x_min + 1e-6).view(B, C, 1, 1) + x_min.view(B, C, 1, 1)
        # 5) α 融合
        alpha = torch.sigmoid(alpha_param)
        out = alpha * x_eq + (1.0 - alpha) * x
        #print("out gradient:", out.grad)
        #print("log_sigma gradient:", self.log_sigma.grad)
        return out

    def forward(self, inputs):
        # 多尺度或单尺度均可
        if isinstance(inputs, (list, tuple)):
            feats = list(inputs)
            if self.per_level_alpha:
                self._ensure_alpha_list(len(feats), device=feats[0].device)
                outs = [self.forward_single(f, self.alpha_list[i]) for i, f in enumerate(feats)]
            else:
                outs = [self.forward_single(f, self.alpha) for f in feats]
            return outs
        else:
            alpha_param = self.alpha_list[0] if self.per_level_alpha else self.alpha
            return self.forward_single(inputs, alpha_param)


    
