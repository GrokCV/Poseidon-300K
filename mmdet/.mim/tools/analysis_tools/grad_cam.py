import torch
import cv2
import numpy as np
from mmengine.dataset import Compose
from mmdet.apis import init_detector, inference_detector
# 使用 GradCAMPlusPlus，定位更准
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==============================================================================
# 1. 配置区域 (修改这里即可切换层级)
# ==============================================================================
# 0 -> P3 (Stride 8):  看细节、小物体 (海胆、海星)
# 1 -> P4 (Stride 16): 看中等物体
# 2 -> P5 (Stride 32): 看大物体
# 3 -> P6 (Stride 64): 看超大物体
# 4 -> P7 (Stride 128): 看全局、整体轮廓 (海龟)
LAYER_INDEX = 2  # <--- 修改这个数字来观察不同层

CONF_THR = 0.3   # 置信度阈值
CONFIG_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251028_145925/vis_data/config.py'
CHECKPOINT_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251028_145925/epoch_12.pth'
IMG_PATH = 'image/60046.jpg'
DEVICE = 'cuda:0'

# ==============================================================================
# 2. Wrapper
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
        for score in cls_scores:
            score = score.permute(0, 2, 3, 1) 
            score = score.reshape(score.size(0), -1, score.size(-1))
            flat_cls_scores.append(score)
        return torch.cat(flat_cls_scores, dim=1)

# ==============================================================================
# 3. Target
# ==============================================================================
class ConfidenceTarget:
    def __init__(self, category_index, score_thr=0.3):
        self.category_index = category_index
        self.score_thr = score_thr

    def __call__(self, model_output):
        if model_output.ndim == 3:
            target_scores = model_output[0, :, self.category_index]
        else:
            target_scores = model_output[:, self.category_index]
        mask = target_scores > self.score_thr
        if mask.sum() == 0: return target_scores.sum() * 0.0
        return target_scores[mask].sum()

# ==============================================================================
# 4. 后处理 (裁剪 + 平滑)
# ==============================================================================
def process_heatmap(heatmap, valid_shape, origin_shape):
    h_valid, w_valid = valid_shape[:2]
    h_orig, w_orig = origin_shape[:2]
    
    # 1. 移除 Padding
    heatmap = heatmap[:h_valid, :w_valid]
    
    # 2. 归一化
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
        
    # 3. Resize (使用 Bicubic 插值，这对低分辨率的 P6/P7 尤其重要，防止马赛克)
    heatmap = cv2.resize(heatmap, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    return heatmap

# ==============================================================================
# 5. 主程序
# ==============================================================================
def main():
    print(f"1. Loading Model...")
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    model.eval()

    print(f"2. Detecting...")
    results = inference_detector(model, IMG_PATH)
    pred = results.pred_instances
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()
    
    valid_indices = scores > CONF_THR
    if not valid_indices.any():
        print("No objects detected.")
        return
    unique_labels = np.unique(labels[valid_indices])
    print(f"   -> Classes: {unique_labels}")

    # --- Preprocess ---
    cfg = model.cfg
    test_pipeline = Compose(cfg.test_pipeline)
    data = test_pipeline({'img_path': IMG_PATH})
    ori_shape = data['data_samples'].ori_shape
    valid_shape = data['data_samples'].img_shape
    
    data['inputs'] = [data['inputs']]
    data['data_samples'] = [data['data_samples']]
    if DEVICE != 'cpu':
        data['inputs'] = [x.to(DEVICE) for x in data['inputs']]
        data['data_samples'] = [x.to(DEVICE) for x in data['data_samples']]
    
    processed_data = model.data_preprocessor(data, training=False)
    input_tensor = processed_data['inputs']
    input_tensor.requires_grad = True

    # --- Grad-CAM ---
    print(f"3. Computing Grad-CAM++ for FPN Layer Index: {LAYER_INDEX}...")
    wrapper_model = MMDetGradCAMWrapper(model)
    
    # [核心] 只选择用户指定的那一层
    # model.neck.fpn_convs 是一个 List，包含 5 个卷积层
    target_layers = [model.neck.fpn_convs[LAYER_INDEX].conv]
    
    cam = GradCAMPlusPlus(model=wrapper_model, target_layers=target_layers)

    targets = []
    for cls_id in unique_labels:
        targets.append(ConfidenceTarget(category_index=cls_id, score_thr=CONF_THR))
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # --- Visualize ---
    final_heatmap = process_heatmap(grayscale_cam, valid_shape, ori_shape)

    img = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    visualization = show_cam_on_image(img_rgb, final_heatmap, use_rgb=True)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    bboxes = pred.bboxes.cpu().numpy()
    for i in range(len(scores)):
        if scores[i] > CONF_THR:
            x1, y1, x2, y2 = bboxes[i].astype(int)
            cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 255, 255), 1)
            # cv2.putText(visualization, f'{scores[i]:.2f}', (x1, y1-5), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    save_path = f'gradcam_layer_P{LAYER_INDEX + 3}.jpg'
    cv2.imwrite(save_path, visualization)
    print(f"Done! Result saved to: {save_path}")

if __name__ == '__main__':
    main()

# import torch
# import mmcv
# import numpy as np
# import cv2
# from mmdet.apis import inference_detector
# from mmdet.apis import init_detector
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# #from pytorch_grad_cam.utils.model_targets import BaseCAMTarget
# from mmengine.dataset import Compose
# import cv2
# import numpy as np
# import torch
# from mmdet.apis import init_detector, inference_detector
# import torch
# import torch
# import cv2
# import numpy as np
# import mmcv
# from mmengine.dataset import Compose
# from mmdet.apis import init_detector, inference_detector
# from pytorch_grad_cam import EigenCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

# # ---------------------------------------------------
# # 1. 修正后的 Wrapper
# # ---------------------------------------------------
# class MMDetFeatureWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, x):
#         # x: [B, C, H, W] 归一化后的 Tensor
#         feats = self.model.extract_feat(x)
        
#         # [关键修改]
#         # extract_feat 返回的是 tuple (feat_lvl0, feat_lvl1, ...)。
#         # pytorch_grad_cam 期望 forward 返回一个 Tensor。
#         # 这里我们返回 tuple 中的第一个 tensor (P3层)，仅仅是为了让库的流程跑通。
#         # EigenCAM 计算热力图时使用的是 hook 抓取的 target_layers 的特征，
#         # 所以这里的返回值不会影响热力图的实际生成。
#         return feats[0]

# # ---------------------------------------------------
# # 2. 配置与初始化
# # ---------------------------------------------------
# config_file = 'work_dirs/tood_r50_fpn_1x_poseidon/20251215_183547/vis_data/config.py'
# checkpoint_file = 'work_dirs/tood_r50_fpn_1x_poseidon/20251215_183547/epoch_12.pth'
# img_path = '14581.jpg'
# device = 'cuda:0'

# # 初始化模型
# model = init_detector(config_file, checkpoint_file, device=device)
# model.eval()

# # ---------------------------------------------------
# # 3. 数据预处理 (解决尺寸不匹配问题)
# # ---------------------------------------------------
# # 使用配置文件的 pipeline 确保 Resize 和 Pad 正确
# cfg = model.cfg
# test_pipeline = Compose(cfg.test_pipeline)

# # 准备数据
# data = {'img_path': img_path}
# data = test_pipeline(data)

# # 处理 Batch 维度并送入 GPU
# data['inputs'] = [data['inputs']]
# data['data_samples'] = [data['data_samples']]

# with torch.no_grad():
#     if device != 'cpu':
#         data['inputs'] = [x.to(device) for x in data['inputs']]
#         data['data_samples'] = [x.to(device) for x in data['data_samples']]
    
#     # 这一步进行 Normalization 和 Pad (至 32 的倍数)
#     # 解决了 "size mismatch" 错误
#     processed_data = model.data_preprocessor(data, training=False)
#     input_tensor = processed_data['inputs'] # [1, 3, H, W]

# # ---------------------------------------------------
# # 4. 定义 EigenCAM
# # ---------------------------------------------------
# wrapper_model = MMDetFeatureWrapper(model)

# # 选择 Target Layer
# # 这里选择 FPN 的 P4 层 (fpn_convs[1])，通常对中等物体效果好
# # 你可以尝试改为 fpn_convs[0] (P3, 小物体) 或 fpn_convs[2] (P5, 大物体)
# target_layers = [model.neck.fpn_convs[1].conv] 

# cam = EigenCAM(
#     model=wrapper_model,
#     target_layers=target_layers,
# )

# # ---------------------------------------------------
# # 5. 生成可视化
# # ---------------------------------------------------
# print(f"开始计算 EigenCAM, Input Shape: {input_tensor.shape}")

# # 生成热力图
# # EigenCAM 不需要 target category，它自动寻找主要特征
# grayscale_cam = cam(input_tensor=input_tensor)
# grayscale_cam = grayscale_cam[0, :] # [H, W]

# # ---------------------------------------------------
# # 6. 后处理与画框
# # ---------------------------------------------------
# # 读取原图
# img = cv2.imread(img_path)
# h, w = img.shape[:2]

# # 将热力图 Resize 回原图尺寸
# # 注意：input_tensor 经过了 pad，heatmap 边缘可能有黑边，
# # resize 回原图尺寸可以自动切除 pad 部分（如果 pad 在右下角）
# heatmap = cv2.resize(grayscale_cam, (w, h))

# # 叠加热力图
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
# visualization = show_cam_on_image(img_rgb, heatmap, use_rgb=True)
# visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

# # 使用 Inference 接口再次获取检测框用于绘制
# results = inference_detector(model, img_path)
# pred_instances = results.pred_instances
# bboxes = pred_instances.bboxes.cpu().numpy()
# scores = pred_instances.scores.cpu().numpy()
# labels = pred_instances.labels.cpu().numpy()

# # 绘制高分框
# for i in range(len(bboxes)):
#     if scores[i] > 0.3: # 阈值
#         x1, y1, x2, y2 = bboxes[i].astype(int)
#         # 画绿色框
#         cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         # 写分数
#         label_text = f'{scores[i]:.2f}'
#         cv2.putText(visualization, label_text, (x1, y1-5), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# output_file = 'eigencam_final.jpg'
# cv2.imwrite(output_file, visualization)
# print(f"可视化完成: {output_file}")
# model = init_detector(config_file, checkpoint_file, device=device)
# model.eval()
# cfg = model.cfg

# # -----------------------------
# # 构建 pipeline 生成 tensor
# # -----------------------------
# test_pipeline_cfg = get_test_pipeline_cfg(cfg)
# test_pipeline = Compose(test_pipeline_cfg)
# data = {"img_path": img_path}
# data = test_pipeline(data)
# img_tensor = data["inputs"].float().unsqueeze(0).to(device)
#   # [1,C,H,W]

# # -----------------------------
# # 定义 Grad-CAM 类
# # -----------------------------
# class GradCAM:
#     def __init__(self, model, target_layer, device='cuda:0'):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         self.hook_handles = []
#         self.device = device
#         self._register_hooks()

#     def _register_hooks(self):
#         # forward hook
#         def forward_hook(module, input, output):
#             self.activations = output.detach()

#         # backward hook
#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0].detach()

#         handle_fw = self.target_layer.register_forward_hook(forward_hook)
#         handle_bw = self.target_layer.register_backward_hook(backward_hook)
#         self.hook_handles.extend([handle_fw, handle_bw])
#     # def _register_hooks(self):
#     #     def forward_hook(module, input, output):
#     #         if isinstance(output, (tuple, list)):
#     #             output = output[0]
#     #         output.requires_grad_(True)  # 确保可以求梯度
#     #         output.retain_grad()
#     #         self.activations = output

#     #     def backward_hook(module, grad_input, grad_output):
#     #         self.gradients = grad_output[0]

#     #     handle_fw = self.target_layer.register_forward_hook(forward_hook)
#     #     handle_bw = self.target_layer.register_full_backward_hook(backward_hook)
#     #     self.hook_handles.extend([handle_fw, handle_bw])


#     def remove_hooks(self):
#         for handle in self.hook_handles:
#             handle.remove()

#     def generate_cam(self, img_tensor):
#         img_tensor = img_tensor.float().to(self.device)
#         img_tensor.requires_grad_(True)

#         with torch.enable_grad():
#             _ = self.model.forward(img_tensor)

#         if self.activations is None:
#             raise RuntimeError("target_layer 未产生激活，请检查 target_layer 是否正确")

#         if not self.activations.requires_grad:
#             self.activations.requires_grad_(True)
#         self.activations.retain_grad()

#         score = self.activations.mean()
#         self.model.zero_grad()
#         score.backward(retain_graph=True)

#         weights = torch.mean(self.activations.grad, dim=(2,3), keepdim=True)
#         cam = torch.sum(weights * self.activations, dim=1)[0]
#         cam = torch.relu(cam)
#         cam -= cam.min()
#         cam /= (cam.max() + 1e-8)

#         return cam.detach().cpu().numpy()  # <-- 关键修改


# # -----------------------------
# # 选择 target layer（FPN 输出最后一层 conv 或 ViT adapter 输出）
# # -----------------------------
# target_layer = model.neck.fpn_convs[1].conv
# # 对 ViT adapter，可以修改为：
# # target_layer = model.backbone.blocks[-1].mlp[0]
# grad_cam = GradCAM(model, target_layer, device=device)

# # -----------------------------
# # 推理得到 DetDataSample
# # -----------------------------
# results = inference_detector(model, img_path)
# if not isinstance(results, list):
#     results = [results]

# # 读取原图用于叠加
# img = cv2.imread(img_path)

# # -----------------------------
# # 对每个预测目标生成 Grad-CAM
# # -----------------------------
# for sample_idx, sample in enumerate(results):
#     det_instances = sample.pred_instances
#     bboxes = det_instances.bboxes.cpu().numpy()
#     scores = det_instances.scores.cpu().numpy()
#     labels = det_instances.labels.cpu().numpy()

#     for i in range(len(bboxes)):
#         if scores[i] < 0.1:  # 过滤低分
#             continue
#         x1, y1, x2, y2 = bboxes[i].astype(int)
#         cls_idx = labels[i]

#         # 生成 Grad-CAM
#         cam = grad_cam.generate_cam(img_tensor)

#         # resize cam 到原图大小
#         heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
#         heatmap = np.uint8(255 * heatmap)
#         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#         # 叠加 bbox
#         overlay = heatmap * 0.6 + img
#         # overlay = np.clip(overlay, 0, 255).astype(np.uint8)
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
#         cv2.putText(overlay, f'class:{cls_idx} score:{scores[i]:.2f}', 
#                     (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

#         # 保存结果
#         cv2.imwrite(f'gradcam_sample{sample_idx}_cls{cls_idx}_obj{i}.jpg', overlay)

# grad_cam.remove_hooks()
# print("Grad-CAM 可视化完成，结果保存为 gradcam_sample*_cls*_obj*.jpg")
# 定义 Grad-CAM 函数
# -----------------------------
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         self.hook_handles = []

#         self._register_hooks()

#     def _register_hooks(self):
#         def forward_hook(module, input, output):
#             self.activations = output.detach()

#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0].detach()

#         handle_fw = self.target_layer.register_forward_hook(forward_hook)
#         handle_bw = self.target_layer.register_backward_hook(backward_hook)
#         self.hook_handles.extend([handle_fw, handle_bw])

#     def remove_hooks(self):
#         for handle in self.hook_handles:
#             handle.remove()

#     def __call__(self, input_tensor, class_idx=None):
#         #output = self.model(return_loss=False, imgs=[input_tensor])[0]
#         # 输入是 numpy 图片或者 tensor
# # 如果是 tensor，需要先转成 [B,C,H,W]
#         img_tensor = input_tensor.to(model.device)
#         output = inference_detector(model, img_tensor)  #
#         # 如果是检测任务，可以选择某个类别的输出
#         if isinstance(output, list):
#             score = output[0]['scores'][0]  # 选择最高分目标
#             score.backward(retain_graph=True)
#         else:
#             output[0][class_idx].backward(retain_graph=True)

#         # 计算 Grad-CAM
#         weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
#         cam = torch.sum(weights * self.activations, dim=1)[0]
#         cam = torch.relu(cam)
#         cam -= cam.min()
#         cam /= cam.max()
#         cam = cam.cpu().numpy()
#         return cam

# # -----------------------------
# # 读入图片
# # -----------------------------
# img = cv2.imread(IMG_PATH)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float()

# # -----------------------------
# # 选择可视化层（FPN 最后一层 conv）
# # -----------------------------
# target_layer = model.neck.fpn_convs[-1].conv  # 根据你的模型修改
# grad_cam = GradCAM(model, target_layer)

# # -----------------------------
# # 生成热力图
# # -----------------------------
# cam = grad_cam(img_tensor)
# heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# superimposed_img = heatmap * 0.4 + img

# # -----------------------------
# # 保存结果
# # -----------------------------
# cv2.imwrite('gradcam_result.jpg', superimposed_img)
# print("Grad-CAM 可视化完成，保存为 gradcam_result.jpg")
# grad_cam.remove_hooks()