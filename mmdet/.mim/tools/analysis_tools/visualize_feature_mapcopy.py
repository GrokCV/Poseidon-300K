import os
import cv2
import torch
import matplotlib.pyplot as plt
from mmdet.apis import init_detector
import random
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns
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
def overlay_heatmap(img, heatmap, alpha=0.3):
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
#======== 可视化单模块热力图 ========
def visualize_single_heatmap(img, feat, layer_name='C3', save_dir='work_dirs/feature_vis', grid=False):
    os.makedirs(save_dir, exist_ok=True)
    heat = get_heatmap(feat)
    overlay = overlay_heatmap(img, heat)
    
    save_path = os.path.join(save_dir, f'{layer_name}_heatmap.png')
    cv2.imwrite(save_path, overlay[:, :, ::-1])  # RGB->BGR for cv2
    print(f"✅ Saved heatmap: {save_path}")
# def visualize_single_heatmap(img, feat, layer_name='C3', save_dir='work_dirs/feature_vis', grid=False):
#     os.makedirs(save_dir, exist_ok=True)

#     # 获取 heatmap（0~1）
#     heat = get_heatmap(feat)

#     # ----------- 保存叠加视觉图 (无 colorbar，和你之前一样) -----------
#     overlay = overlay_heatmap(img, heat)
#     save_path_overlay = os.path.join(save_dir, f'{layer_name}_heatmap_overlay.png')
#     cv2.imwrite(save_path_overlay, overlay[:, :, ::-1])
#     print(f"✅ Saved overlay heatmap: {save_path_overlay}")

#     # ----------- 新增：带 colorbar 的热力图可视化 -----------
#     plt.figure(figsize=(6, 5))
#     plt.imshow(heat, cmap='jet')
#     plt.title(f'{layer_name} Heatmap (0~1 normalized)')
#     plt.colorbar(label='Heat Value')  # 🔥 关键：热力值范围

#     plt.axis('off')
#     save_path_cb = os.path.join(save_dir, f'{layer_name}_heatmap_colorbar.png')
#     plt.savefig(save_path_cb, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"🎨 Saved heatmap with colorbar: {save_path_cb}")


def visualize_feature_map_comparison(feat1, feat2, num_channels=4, layer_name='C3', save_dir='work_dirs/feature_vis'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for i in range(num_channels):
        plt.subplot(2, num_channels, i + 1)
        plt.imshow(feat1[0, i].detach().cpu(), cmap='magma')
        plt.title(f'{layer_name}-Before Feature HE')
        plt.axis('off')

        plt.subplot(2, num_channels, i + num_channels + 1)
        plt.imshow(feat2[0, i].detach().cpu(), cmap='magma')
        plt.title(f'{layer_name}-After Feature HE')
        plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{layer_name}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved visualization: {save_path}")

def visualize_feature_distribution(feat1, feat2, layer_name='C3', save_dir='work_dirs/feature_vis', num_channels=16,bins = 128,y_min_threshold=0.5,density_crop_ratio=0.99):
    #平滑版
    # os.makedirs(save_dir, exist_ok=True)
    # sns.set_style("white")  # 去掉背景网格

    # # ===== 随机选择通道并取平均 =====
    # os.makedirs(save_dir, exist_ok=True)
    # sns.set_style("white")

    # # ===== 随机选择通道并取平均 =====
    # C = feat1.shape[1]
    # channels = random.sample(range(C), min(num_channels, C))
    # feat1_avg = feat1[0, channels, :, :].mean(dim=0).cpu().flatten().numpy()
    # feat2_avg = feat2[0, channels, :, :].mean(dim=0).cpu().flatten().numpy()

    # kde1 = gaussian_kde(feat1_avg)
    # kde2 = gaussian_kde(feat2_avg)

    # xmin = min(feat1_avg.min(), feat2_avg.min())
    # xmax = max(feat1_avg.max(), feat2_avg.max())
    # xs = np.linspace(xmin, xmax, 400)

    # y1 = kde1(xs)
    # y2 = kde2(xs)

    # # ===== 自动裁剪横坐标范围（只保留累积分布高的部分） =====
    # cdf1 = np.cumsum(y1)
    # cdf1 /= cdf1[-1]
    # cdf2 = np.cumsum(y2)
    # cdf2 /= cdf2[-1]

    # # 找到累计概率达到指定阈值的右端位置
    # x1_right = xs[np.searchsorted(cdf1, density_crop_ratio)]
    # x2_right = xs[np.searchsorted(cdf2, density_crop_ratio)]

    # # 取两者中较大的范围，保证两图横坐标一致
    # x_right = max(x1_right, x2_right)
    # x_left = xmin

    # ymax = max(y1.max(), y2.max())

    # # ===== 绘图 =====
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    # fig.suptitle(f'{layer_name}: KDE Feature Distribution (avg of {channels})', fontsize=11)

    # # --- Before HE ---
    # axes[0].plot(xs, y1, color='steelblue', linewidth=2.2)
    # axes[0].fill_between(xs, y1, color='steelblue', alpha=0.25)
    # axes[0].set_title('Before HE')
    # axes[0].set_xlabel('Feature Value')
    # axes[0].set_ylabel('Density')
    # axes[0].set_xlim(x_left, x_right)
    # axes[0].set_ylim(0, ymax * 1.05)
    # axes[0].spines['top'].set_visible(True)
    # axes[0].spines['right'].set_visible(True)

    # # --- After HE ---
    # axes[1].plot(xs, y2, color='darkorange', linewidth=2.2)
    # axes[1].fill_between(xs, y2, color='darkorange', alpha=0.25)
    # axes[1].set_title('After HE')
    # axes[1].set_xlabel('Feature Value')
    # axes[1].set_xlim(x_left, x_right)
    # axes[1].set_ylim(0, ymax * 1.05)
    # axes[1].spines['top'].set_visible(True)
    # axes[1].spines['right'].set_visible(True)

    # plt.tight_layout()
    # save_path = os.path.join(save_dir, f'{layer_name}_kde_split_cropped.png')
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"✅ Saved split KDE comparison with cropped x-axis: {save_path}")
    

    #随机选择几个取平均通道绘制直方图
    os.makedirs(save_dir, exist_ok=True)

    C = feat1.shape[1]
    channels = random.sample(range(C), min(num_channels, C))

    # 跨通道平均 -> [H, W]
    feat1_avg = feat1[0, channels, :, :].mean(dim=0)
    feat2_avg = feat2[0, channels, :, :].mean(dim=0)

    # 展平为 numpy
    f1 = feat1_avg.detach().cpu().numpy().ravel()
    f2 = feat2_avg.detach().cpu().numpy().ravel()

    # # 统一横坐标范围 (xmin, xmax)
    # xmin = float(min(f1.min(), f2.min()))
    # xmax = float(max(f1.max(), f2.max()))
    x_range=(0, 0.3)
    xmin, xmax = x_range
    if xmin == xmax:
        # 极端情况：所有值相同，扩一点范围以便显示
        xmin -= 1e-3
        xmax += 1e-3

    # 用 numpy 计算 density（PDF）以便取得正确的 ymax
    hist1, bin_edges = np.histogram(f1, bins=bins, range=(xmin, xmax), density=True)
    hist2, _ = np.histogram(f2, bins=bins, range=(xmin, xmax), density=True)

    ymax = max(hist1.max(), hist2.max()) * 1.1
    ymin = min(y_min_threshold, ymax * 0.9) if ymax < y_min_threshold else y_min_threshold


    # ymax = max(hist1.max() if hist1.size else 0.0,
    #            hist2.max() if hist2.size else 0.0)
    # # 给一点 margin（10%），防止顶端被裁到
    # ymax *= 1.1

    # 绘图
    # plt.figure(figsize=(10, 4))
    # x_vals = np.linspace(xmin, xmax, 500)
    # plt.subplot(1, 2, 1)
    # plt.hist(f1, bins=bins, range=(xmin, xmax), density=True, alpha=0.3, color='steelblue')
    # kde1 = gaussian_kde(f1)
    # plt.plot(x_vals, kde1(x_vals), color="#2973F1", linewidth=1.5)
    # plt.title(f'{layer_name} - Before HE')
    # plt.xlabel('Feature Value')
    # plt.ylabel('Density (PDF)')
    # plt.xlim(xmin, xmax)
    # plt.ylim(0, ymax)
                                          
    # plt.subplot(1, 2, 2)
    # plt.hist(f2, bins=bins, range=(xmin, xmax), density=True, alpha=0.3, color='darkorange')
    
    
    # kde2 = gaussian_kde(f2)
    
    # plt.plot(x_vals, kde2(x_vals), color="#F3562B", linewidth=1.5)
    # plt.title(f'{layer_name} - After HE')
    # plt.xlabel('Feature Value')
    # plt.ylabel('Density (PDF)')
    # plt.xlim(xmin, xmax)
    # plt.ylim(0, ymax)

    # plt.suptitle(f'{layer_name}: PDF (avg of channels {channels})', fontsize=11)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])

    # save_path = os.path.join(save_dir, f'{layer_name}_random_avg_hist_aligned.png')
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"✅ Saved histogram comparison with same PDF scale: {save_path}")
    #=======================================第二版绘图==============================================
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.linewidth': 0.8,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300
    })

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    x_vals = np.linspace(xmin, xmax, 500)

    # 定义颜色
    color_before = "#0070C0"  # 柔和蓝
    color_after  = "#F40C0C"  # 柔和红橙

    # ---- BEFORE ----
    ax = axes[0]
    kde1 = gaussian_kde(f1)
    ax.fill_between(x_vals, kde1(x_vals), color="#B6C7EA", alpha=0.4)
    ax.plot(x_vals, kde1(x_vals), color=color_before, linewidth=1)
    ax.set_title('Before Feature HistEqDet', pad=8,fontsize=16)
    ax.set_xlabel('Feature Value',fontsize=16)
    ax.set_ylabel('Density',fontsize=16)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.grid(False)

    # ---- AFTER ----
    ax = axes[1]
    kde2 = gaussian_kde(f2)
    ax.fill_between(x_vals, kde2(x_vals), color="#F5B7BF", alpha=0.4)
    ax.plot(x_vals, kde2(x_vals), color=color_after, linewidth=1)
    ax.set_title('After Feature HistEqDet', pad=8,fontsize=16)
    ax.set_xlabel('Feature Value',fontsize=16)
    ax.set_ylabel('Density',fontsize=16)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.grid(False)

    # ---- 美化外框 ----
    for ax in axes:
    # 保持四边边框可见且样式一致
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)

        # 只启用 bottom 和 left 的刻度线，并设置为向内
        ax.tick_params(axis='x',
                    which='both',
                    direction='in',   # 刻度线向内
                    bottom=True, top=False,
                    length=4, width=0.8,
                    pad=6,            # 数字与轴线距离（正值使数字在外侧）
                    labelsize=9)

        ax.tick_params(axis='y',
                    which='both',
                    direction='in',   # 刻度线向内
                    left=True, right=False,
                    length=4, width=0.8,
                    pad=6,            # 数字与轴线距离（正值使数字在外侧）
                    labelsize=9)

        # 确保上、右侧没有刻度线或刻度标签
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # 关闭网格（保持干净）
        ax.grid(False)
             

    # ---- 总标题 ----
    fig.suptitle(f'{layer_name}: Feature Distribution (avg of {channels})',
                 fontsize=11, y=1.02)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{layer_name}_paper_style.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved paper-style figure: {save_path}")

    # os.makedirs(save_dir, exist_ok=True)

    # C = feat1.shape[1]
    # # 随机选择 num_channels 个通道
    # channels = random.sample(range(C), min(num_channels, C))
    
    # # 对选中的通道取平均
    # feat1_avg = feat1[0, channels, :, :].mean(dim=0)
    # feat2_avg = feat2[0, channels, :, :].mean(dim=0)

    # plt.figure(figsize=(6, 4))
    # plt.hist(feat1_avg.cpu().flatten(), bins=128, alpha=0.6, label='Before HE',density=True)
    # plt.hist(feat2_avg.cpu().flatten(), bins=128, alpha=0.6, label='After HE',density=True)

    # plt.legend(fontsize=8)
    # plt.xlabel('Feature Value')
    # plt.ylabel('Density')
    # plt.title(f'{layer_name}: Feature Value Distribution (avg of channels {channels})')

    # save_path = os.path.join(save_dir, f'{layer_name}_random_avg_hist.png')
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"✅ Saved random averaged histogram: {save_path}")
    
    
    #随机选择几个通道绘制直方图
    # os.makedirs(save_dir, exist_ok=True)
    # C = feat1.shape[1]

    # # 随机选择几个通道
    # channels = random.sample(range(C), min(num_channels, C))

    # plt.figure(figsize=(6, 4))
    
    # for ch in channels:
    #     plt.hist(feat1[0, ch].cpu().flatten(), bins=128, alpha=0.4, label=f'Before HE ch{ch}')
    #     plt.hist(feat2[0, ch].cpu().flatten(), bins=128, alpha=0.4, label=f'After HE ch{ch}')
    
    # plt.legend(fontsize=8)
    # plt.xlabel('Feature Value')
    # plt.ylabel('Density')
    # plt.title(f'{layer_name}: Feature Value Distribution (subset of channels)')
    
    # save_path = os.path.join(save_dir, f'{layer_name}_subset_hist.png')
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"✅ Saved subset histogram: {save_path}")
    #所有通道计算总的
    # os.makedirs(save_dir, exist_ok=True)
    # plt.figure(figsize=(6, 4))
    # plt.hist(feat1.cpu().flatten(), bins=512, alpha=0.6, label='Before HE')
    # plt.hist(feat2.cpu().flatten(), bins=512, alpha=0.6, label='After HE')
    # plt.legend()
    # plt.xlabel('Feature Value')
    # plt.ylabel('Density')
    # plt.title(f'{layer_name}: Feature Value Distribution')
    # save_path = os.path.join(save_dir, f'{layer_name}_hist.png')
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"✅ Saved histogram: {save_path}")

def main():
    # -----------------------------
    # 1️⃣ 配置与模型加载
    # -----------------------------
    config_file = 'work_dirs/tood_r50_fpn_1x_poseidon/20251111_204403/vis_data/config.py'
    checkpoint_file = 'work_dirs/tood_r50_fpn_1x_poseidon/20251111_204403/epoch_12.pth'
    img_path = '14581.jpg'

    print("🚀 Loading model...")
    model = init_detector(config_file, checkpoint_file, device=device)
    model.eval()

    # -----------------------------
    # 2️⃣ 读取图像并转换为 tensor
    # -----------------------------
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (512, 512))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # -----------------------------
    # 3️⃣ 前向提取特征
    # -----------------------------
    print("🔍 Extracting feature maps...")
    with torch.no_grad():
        feats_neck, feats_backbone, feats_he = model.extract_feat(img_tensor)
        #feats_neck = model.extract_feat(img_tensor)
        

    # -----------------------------
    # 4️⃣ 可视化若干层
    # -----------------------------
    layer_names = ['C3', 'C4', 'C5']
    #layer_names = ['P3', 'P4', 'P5']
    for i, name in enumerate(layer_names):
        visualize_feature_map_comparison(feats_backbone[i], feats_he[i], num_channels=3, layer_name=name)
        visualize_feature_distribution(feats_backbone[i], feats_he[i], layer_name=name)
        visualize_single_heatmap(img_rgb, feats_backbone[i], layer_name=name, save_dir='work_dirs/feature_vis')


    print("✅ All visualizations saved to: work_dirs/feature_vis")

if __name__ == '__main__':
    main()
