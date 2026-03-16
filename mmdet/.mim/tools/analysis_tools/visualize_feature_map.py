import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ======== 生成热力图 ========
def get_heatmap(feat):
    """
    feat: Tensor [1, C, H, W]
    return: numpy [H, W], normalized [0,1]
    """
    fmap = feat[0]  # [C,H,W]
    heatmap = fmap.mean(dim=0)  # 通道平均
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-6
    return heatmap.cpu().numpy()
# ======== 叠加热力图 ========
def overlay_heatmap(img, heatmap, alpha=0.5):
    """
    img: numpy [H,W,3] RGB 0-255
    heatmap: numpy [H,W] 0-1
    alpha: 热力图透明度
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# ======== 可选：在图上绘制网格 ========
def draw_grid(img, grid_size=4, color=(255,255,255)):
    h, w, _ = img.shape
    step_h = h // grid_size
    step_w = w // grid_size
    for i in range(1, grid_size):
        cv2.line(img, (0, i*step_h), (w, i*step_h), color, 1)
        cv2.line(img, (i*step_w, 0), (i*step_w, h), color, 1)
    return img

def overlay_heatmap(img, heatmap, alpha=0.5):
    """
    img: numpy [H,W,3] RGB 0-255
    heatmap: numpy [H,W] 0-1
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# ======== 可视化 Backbone vs HE ========
def visualize_heatmap_comparison(img, feat_backbone, feat_he, layer_name='C3', save_dir='work_dirs/feature_vis', grid=False):
    os.makedirs(save_dir, exist_ok=True)

    heat_b = get_heatmap(feat_backbone)
    heat_h = get_heatmap(feat_he)

    overlay_b = overlay_heatmap(img, heat_b)
    overlay_h = overlay_heatmap(img, heat_h)

    if grid:
        overlay_b = draw_grid(overlay_b.copy())
        overlay_h = draw_grid(overlay_h.copy())

    # 拼接左右对比
    combined = np.concatenate([overlay_b, overlay_h], axis=1)

    save_path = os.path.join(save_dir, f'{layer_name}_heatmap_overlay.png')
    cv2.imwrite(save_path, combined[:, :, ::-1])  # RGB->BGR for cv2
    print(f"✅ Saved heatmap overlay comparison: {save_path}")

# ======== 主函数 ========
def main():
    # 配置
    config_file = 'work_dirs/tood_r50_fpn_1x_poseidon/20251018_195916/vis_data/config.py'
    checkpoint_file = 'work_dirs/tood_r50_fpn_1x_poseidon/20251018_195916/epoch_12.pth'
    img_path = '15226.jpg'
    save_dir = 'work_dirs/feature_vis'
    use_grid = True  # 是否在图上加网格

    print("🚀 Loading model...")
    model = init_detector(config_file, checkpoint_file, device=device)
    model.eval()

    # 读取原图
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转 Tensor 并归一化
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # 提取特征
    print("🔍 Extracting features...")
    with torch.no_grad():
        feats_neck, feats_backbone, feats_he = model.extract_feat(img_tensor)

    # 可视化热力图
    layer_names = ['C3', 'C4', 'C5']
    for i, name in enumerate(layer_names):
        visualize_heatmap_comparison(img_rgb, feats_backbone[i], feats_he[i],
                                     layer_name=name, save_dir=save_dir, grid=use_grid)

    print(f"✅ All heatmaps saved to: {save_dir}")

if __name__ == '__main__':
    main()


'''
import os
import torch
import cv2
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmdet.registry import MODELS
from mmengine.runner import load_checkpoint
import mmdet.models

def register_hook(model, layer_name, feature_dict, key_name):
    """注册 forward hook 捕获特征"""
    def hook_fn(module, input, output):
        feature_dict[key_name] = output.detach()
    layer = dict(model.named_modules())[layer_name]
    layer.register_forward_hook(hook_fn)


def visualize_feature_maps(feat_before, feat_after, save_dir, prefix='compare'):
    """保存前后对比特征图"""
    os.makedirs(save_dir, exist_ok=True)
    num_channels = min(8, feat_before.shape[1])
    for i in range(num_channels):
        f1 = feat_before[0, i].cpu().numpy()
        f2 = feat_after[0, i].cpu().numpy()

        # 归一化
        f1 = (f1 - f1.min()) / (f1.max() - f1.min() + 1e-5)
        f2 = (f2 - f2.min()) / (f2.max() - f2.min() + 1e-5)

        # 组合成对比图
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(f1, cmap='viridis')
        axs[0].set_title('Before HistEq')
        axs[0].axis('off')
        axs[1].imshow(f2, cmap='viridis')
        axs[1].set_title('After HistEq')
        axs[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_ch{i}.png'),
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    print(f"✅ Saved feature comparison maps to: {save_dir}")


def load_model(cfg_path, checkpoint_path=None, device='cuda:0'):
    """加载模型"""
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def main():
    # === 根据你自己的路径修改 ===
    img_path = '15226.jpg'
    cfg_path = 'work_dirs/tood_r50_fpn_1x_poseidon/20251018_195916/vis_data/config.py'
    checkpoint_path = 'work_dirs/tood_r50_fpn_1x_poseidon/20251018_195916/epoch_12.pth'
    save_dir = './feature_maps_compare'

    model = load_model(cfg_path, checkpoint_path)
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        print(name)
    # 捕获点：ResNet50 输出层 + 直方图均衡化模块输出层
    feature_dict = {}
    register_hook(model, 'backbone.layer4', feature_dict, 'before_histeq')
    register_hook(model, 'backbone.histeq_module', feature_dict, 'after_histeq')  # 注意这里改成你模块的名字

    # 读取图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)

    data = {'inputs': [img_tensor], 'data_samples': None}
    inputs = model.data_preprocessor(data, False)

    # 前向传播
    with torch.no_grad():
        _ = model(inputs['inputs'], inputs['data_samples'], mode='tensor')

    feat_before = feature_dict['before_histeq']
    feat_after = feature_dict['after_histeq']

    visualize_feature_maps(feat_before, feat_after, save_dir, prefix='HistEqCompare')


if __name__ == '__main__':
    main()
'''

'''
import os
import torch
import matplotlib.pyplot as plt
from mmdet.apis import init_detector

from mmcv.image import imread

def visualize_feature_map_comparison(feat1, feat2, num_channels=4, layer_name='C3', save_dir='work_dirs/feature_vis'):
    """可视化 HE 前后特征图"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))

    for i in range(num_channels):
        plt.subplot(2, num_channels, i + 1)
        plt.imshow(feat1[0, i].detach().cpu(), cmap='magma')
        plt.title(f'{layer_name}-Before HE')
        plt.axis('off')

        plt.subplot(2, num_channels, i + num_channels + 1)
        plt.imshow(feat2[0, i].detach().cpu(), cmap='magma')
        plt.title(f'{layer_name}-After HE')
        plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{layer_name}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved visualization: {save_path}")

def visualize_feature_distribution(feat1, feat2, layer_name='C3', save_dir='work_dirs/feature_vis'):
    """绘制特征直方图分布对比"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(feat1.cpu().flatten(), bins=100, alpha=0.6, label='Before HE')
    plt.hist(feat2.cpu().flatten(), bins=100, alpha=0.6, label='After HE')
    plt.legend()
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.title(f'{layer_name}: Feature Value Distribution')
    save_path = os.path.join(save_dir, f'{layer_name}_hist.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved histogram: {save_path}")

def main():
    # -----------------------------
    # 1️⃣ 配置与模型加载
    # -----------------------------
    config_file = 'work_dirs/tood_r50_fpn_1x_poseidon/20251018_195916/vis_data/config.py'  # ← 修改为你的配置文件路径
    checkpoint_file = 'work_dirs/tood_r50_fpn_1x_poseidon/20251018_195916/epoch_12.pth'    # ← 修改为你的权重文件路径
    img_path = '15226.jpg'                # ← 修改为你的测试图像路径

    print("🚀 Loading model...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model.eval()

    # -----------------------------
    # 2️⃣ 读取图像并预处理
    # -----------------------------
    img = imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
# 封装成 MMDet 需要的输入格式
    data = {'inputs': [img_tensor], 'data_samples': None}
    inputs = model.data_preprocessor(data, False)

    inputs = {k: v.unsqueeze(0).to('cuda') for k, v in inputs.items()}

    # -----------------------------
    # 3️⃣ 前向传播，提取 backbone 与 HE 特征
    # -----------------------------
    print("🔍 Extracting feature maps...")
    with torch.no_grad():
        _, feats_backbone, feats_he = model(**inputs)

    # -----------------------------
    # 4️⃣ 可视化若干层（例如 C3, C4, C5）
    # -----------------------------
    layer_names = ['C3', 'C4', 'C5']
    for i, name in enumerate(layer_names):
        visualize_feature_map_comparison(feats_backbone[i], feats_he[i], num_channels=4, layer_name=name)
        visualize_feature_distribution(feats_backbone[i], feats_he[i], layer_name=name)

    print("✅ All visualizations saved to: work_dirs/feature_vis")

if __name__ == '__main__':
    main()
'''