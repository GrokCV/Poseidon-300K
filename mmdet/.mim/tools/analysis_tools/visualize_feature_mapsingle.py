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
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# ======== 可选：绘制网格 ========
def draw_grid(img, grid_size=4, color=(255,255,255)):
    h, w, _ = img.shape
    step_h = h // grid_size
    step_w = w // grid_size
    for i in range(1, grid_size):
        cv2.line(img, (0, i*step_h), (w, i*step_h), color, 1)
        cv2.line(img, (i*step_w, 0), (i*step_w, h), color, 1)
    return img

# ======== 可视化单模块热力图 ========
def visualize_single_heatmap(img, feat, layer_name='C3', save_dir='work_dirs/feature_vis', grid=False):
    os.makedirs(save_dir, exist_ok=True)
    heat = get_heatmap(feat)
    overlay = overlay_heatmap(img, heat)
    if grid:
        overlay = draw_grid(overlay.copy())
    save_path = os.path.join(save_dir, f'{layer_name}_heatmap.png')
    cv2.imwrite(save_path, overlay[:, :, ::-1])  # RGB->BGR for cv2
    print(f"✅ Saved heatmap: {save_path}")

# ======== 主函数 ========
def main():
    # 配置
    config_file = 'work_dirs/tood_r50_fpn_1x_poseidon/20251017_151752/vis_data/config.py'
    checkpoint_file = 'work_dirs/tood_r50_fpn_1x_poseidon/20251017_151752/epoch_12.pth'
    img_path = '15226.jpg'
    save_dir = 'work_dirs/feature_vis1'
    use_grid = True  # 是否加网格

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

    # 可视化单模块热力图（可选 Backbone 或 HE）
    layer_names = ['C3', 'C4', 'C5']
    for i, name in enumerate(layer_names):
        # 这里选择 HE 模块的热力图
        visualize_single_heatmap(img_rgb, feats_neck[i], layer_name=name, save_dir=save_dir, grid=use_grid)

    print(f"✅ All heatmaps saved to: {save_dir}")

if __name__ == '__main__':
    main()
