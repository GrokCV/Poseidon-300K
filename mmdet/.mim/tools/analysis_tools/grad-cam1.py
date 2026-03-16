import torch
import cv2
import numpy as np
import os
from mmengine.dataset import Compose, pseudo_collate
from mmdet.apis import init_detector, inference_detector
# 使用 GradCAMPlusPlus，比普通 GradCAM 更适合目标检测的多目标场景
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==============================================================================
# 1. 配置区域 (请确认路径)
# ==============================================================================


# CONFIG_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251028_145925/vis_data/config.py'
# CHECKPOINT_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251028_145925/epoch_12.pth'
CONFIG_FILE ='work_dirs/tood_r50_fpn_1x_poseidon/20251221_184712/vis_data/config.py'
CHECKPOINT_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251221_184712/epoch_12.pth'
IMG_PATH = 'datasets/Poseidon-300K/images/test/51236.jpg'
DEVICE = 'cuda:3'
CONF_THR = 0.3  # 只显示置信度 > 0.3 的目标
INPUT_SIZE = (512, 512) # 必须与你 Config 中的 scale 保持一致

# ==============================================================================
# 2. 修复版 Wrapper (解决 RuntimeError)
# ==============================================================================
class MMDetGradCAMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.extract_feat(x)
        outs = self.model.bbox_head(feats)
        cls_scores = outs[0]
        
        flat_cls_scores = []
        for s in cls_scores:
            # s shape: [Batch, Num_Classes, H, W]
            # 1. Permute 到 [Batch, H, W, Num_Classes]
            permuted = s.permute(0, 2, 3, 1)
            
            # 2. Reshape 到 [Batch, N, Num_Classes]
            # [核心修复]: 使用 s.size(1) (即类别数) 作为最后一维，而不是 s.size(-1) (宽度)
            # 这样无论 P3 还是 P7，最后一维都是 19 (类别数)，可以拼接
            reshaped = permuted.reshape(permuted.size(0), -1, s.size(1))
            flat_cls_scores.append(reshaped)
            
        return torch.cat(flat_cls_scores, dim=1)

# ==============================================================================
# 3. Target 定义
# ==============================================================================
class ConfidenceTarget:
    def __init__(self, category_index, score_thr=0.3):
        self.category_index = category_index
        self.score_thr = score_thr

    def __call__(self, model_output):
        # model_output: [Batch, Total_Anchors, Num_Classes]
        if model_output.ndim == 3:
            target_scores = model_output[0, :, self.category_index]
        else:
            target_scores = model_output[:, self.category_index]
        
        # 聚合所有高置信度点的梯度
        mask = target_scores > self.score_thr
        if mask.sum() == 0: return target_scores.sum() * 0.0
        return target_scores[mask].sum()

# ==============================================================================
# 4. 图像反归一化工具 (Tensor -> 512x512 Image)
# ==============================================================================
def denormalize_image(tensor, mean, std):
    # 1. 维度变换 & 分离梯度 (解决 RuntimeError: Can't call numpy() on Tensor that requires grad)
    img = tensor.permute(1, 2, 0).detach().cpu().numpy() # [H, W, C]
    
    # 2. 形状对齐
    mean = np.array(mean).flatten().reshape(1, 1, 3)
    std = np.array(std).flatten().reshape(1, 1, 3)
    
    # 3. 反归一化
    img = (img * std + mean) / 255.0
    img = np.clip(img, 0, 1).astype(np.float32)
    
    # 4. RGB -> BGR (OpenCV格式)
    img = img[..., ::-1] 
    return img

# ==============================================================================
# 5. 主程序
# ==============================================================================
def main():
    print("1. Loading Model...")
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    print(model)
    model.eval()

    # --- 步骤 A: 推理获取检测框 (原图坐标) ---
    print("2. Running Inference...")
    results = inference_detector(model, IMG_PATH)
    pred = results.pred_instances
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()
    bboxes_orig = pred.bboxes.cpu().numpy() # 对应原图 1920x1080
    
    valid_indices = scores > CONF_THR
    if not valid_indices.any():
        print("No objects detected.")
        return
    unique_labels = np.unique(labels[valid_indices])
    print(f"   -> Detected Classes: {unique_labels}")

    # --- 步骤 B: 构造输入 Tensor (512x512) ---
    cfg = model.cfg
    test_pipeline = Compose(cfg.test_pipeline)
    data = {'img_path': IMG_PATH}
    data = test_pipeline(data)
    
    # 获取原图尺寸，用于后续画框坐标映射
    ori_shape = data['data_samples'].ori_shape
    ori_h, ori_w = ori_shape[:2]
    
    # 打包 Batch
    data = pseudo_collate([data])
    
    with torch.no_grad():
        if DEVICE != 'cpu':
            data['inputs'] = [x.to(DEVICE) for x in data['inputs']]
            data['data_samples'] = [x.to(DEVICE) for x in data['data_samples']]
        
        # input_tensor 是 [1, 3, 512, 512]
        processed_data = model.data_preprocessor(data, training=False)
        input_tensor = processed_data['inputs']
        input_tensor.requires_grad = True

    # 获取均值方差
    if hasattr(model, 'data_preprocessor'):
        mean = model.data_preprocessor.mean.cpu().numpy()
        std = model.data_preprocessor.std.cpu().numpy()
    else:
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]

    # 还原底图 (512x512, 模型真正看到的图)
    bg_img = denormalize_image(input_tensor[0], mean, std)

    # --- 步骤 C: 计算 Grad-CAM++ (FPN融合) ---
    print("3. Computing Grad-CAM++ (FPN P3-P5)...")
    wrapper_model = MMDetGradCAMWrapper(model)
    
    # [关键设置] 选择 FPN 层
    # P3(0), P4(1), P5(2) 是最佳组合：既有细节，又有语义，且避免了 P7 的大方块
    # target_layers = [model.neck.fpn_convs[i].conv for i in [ 0,1,2,3,4]]
    target_layers = [model.bbox_head.inter_convs[i].conv for i in [ 0,1,2,3,4]]
    
    cam = GradCAMPlusPlus(model=wrapper_model, target_layers=target_layers)

    targets = []
    for cls_id in unique_labels:
        targets.append(ConfidenceTarget(category_index=cls_id, score_thr=CONF_THR))
    
    # 计算热力图
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # --- 步骤 D: 可视化 (所见即所得) ---
    # 1. Resize: 将热力图缩放到 512x512 (使用双三次插值平滑)
    final_heatmap = cv2.resize(grayscale_cam, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
    
    # 2. 叠加: show_cam_on_image 需要 RGB 输入
    bg_img_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    visualization = show_cam_on_image(bg_img_rgb, final_heatmap, use_rgb=True)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    # --- 步骤 E: 坐标映射与画框 ---
    # 将原图坐标映射到 512x512
    scale_x = INPUT_SIZE[0] / ori_w
    scale_y = INPUT_SIZE[1] / ori_h
    
    # for i in range(len(scores)):
    #     if scores[i] > CONF_THR:
    #         x1, y1, x2, y2 = bboxes_orig[i]
            
    #         # 手动缩放坐标
    #         x1 = int(x1 * scale_x)
    #         y1 = int(y1 * scale_y)
    #         x2 = int(x2 * scale_x)
    #         y2 = int(y2 * scale_y)
            
    #         # 画白色细框
    #         cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 255, 255), 1)
    #         # 标注分数 (可选)
    #         # cv2.putText(visualization, f'{scores[i]:.2f}', (x1, y1-5), 
    #         #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    save_path = 'gradcam_mmdet_final.jpg'
    cv2.imwrite(save_path, visualization)
    print(f"Done! Result saved to: {save_path}")
    print("提示：输出图片为 512x512，物体有拉伸变形，这符合 config 设置，保证了热力图绝对对齐。")

if __name__ == '__main__':
    main()