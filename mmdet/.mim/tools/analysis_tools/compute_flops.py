import torch
from fvcore.nn import FlopCountAnalysis
from mmdet.models import build_detector
from mmcv import Config

# 加载你的配置文件（请根据实际路径修改）
cfg = Config.fromfile('work_dirs/tood_r50_fpn_1x_poseidon/config.py')

# 构建模型
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

# 将模型转到设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 创建一个 dummy 输入来计算 FLOPs
dummy_input = torch.randn(1, 3, 512, 512).to(device)  # 输入尺寸请根据模型的实际需求调整

# 计算 FLOPs 和参数量
flops = FlopCountAnalysis(model, dummy_input)
print(f"FLOPs: {flops.total()}")

# 计算参数数量
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {params / 1e6:.2f}M")
