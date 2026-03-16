import bisect
import copy
import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmengine.dataset import Compose, pseudo_collate
# from mmengine.dataset import Compose, collate_utils
from mmengine.model.utils import revert_sync_batchnorm
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample

try:
    from pytorch_grad_cam import AblationCAM, AblationLayer, ActivationsAndGradients
    from pytorch_grad_cam.base_cam import BaseCAM
    from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
    from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

def reshape_transform(feats, max_shape=(20, 20), is_need_grad=False):
    """适配 3.x 的特征图重塑函数"""
    if len(max_shape) == 1:
        max_shape = max_shape * 2

    if isinstance(feats, torch.Tensor):
        feats = [feats]
    
    # 3.x 中提取的 features 通常是 tuple
    max_h = max([im.shape[-2] for im in feats])
    max_w = max([im.shape[-1] for im in feats])
    
    if -1 in max_shape:
        max_shape = (max_h, max_w)
    else:
        max_shape = (min(max_h, max_shape[0]), min(max_w, max_shape[1]))

    activations = []
    for feat in feats:
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(feat), max_shape, mode='bilinear', align_corners=False))

    return torch.cat(activations, axis=1)

class DetCAMModel(nn.Module):
    """封装 MMDet 3.x 模型以适配 pytorch_grad_cam"""

    def __init__(self, cfg, checkpoint, score_thr, device='cuda:0'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.score_thr = score_thr
        # 使用 3.x 的 api 初始化模型
        self.detector = init_detector(cfg, checkpoint, device=device)
        self.detector = revert_sync_batchnorm(self.detector)
        
        if hasattr(self.detector, 'bbox_head'):
            self.detector.bbox_head.epoch = 20
        self.return_loss = False
        self.input_data = None

    def set_return_loss(self, return_loss):
        self.return_loss = return_loss

    def set_input_data(self, img, bboxes=None, labels=None):
        """处理 3.x 的数据流水线"""
        cfg = copy.deepcopy(self.cfg)
        # 3.x 标准推理 Pipeline
        pipeline = cfg.test_dataloader.dataset.pipeline
        # 过滤掉不需要的转换，保留必要的 LoadImageFromNDArray
        if pipeline[0].type == 'LoadImageFromFile':
            pipeline[0].type = 'LoadImageFromNDArray'
        
        test_pipeline = Compose(pipeline)
        
        data_dict = dict(img=img)
        if self.return_loss:
            # 这里的逻辑根据具体 Loss 需求可调，通常 CAM 关注正向推理
            data_dict['gt_bboxes'] = bboxes
            data_dict['gt_labels'] = labels.astype(np.int64)

        data = test_pipeline(data_dict)
        # 3.x 的 collate 略有不同
        # data = collate_utils.pseudo_collate([data])
        data = pseudo_collate([data])
        
        # 将数据移至设备
        if 'cuda' in str(self.device):
            # data['inputs'] = data['inputs'].to(self.device)
            if isinstance(data['inputs'], list):
                data['inputs'] = [img.to(self.device) for img in data['inputs']]
            else:
                data['inputs'] = data['inputs'].to(self.device)
            if 'data_samples' in data:
                for i in range(len(data['data_samples'])):
                    data['data_samples'][i] = data['data_samples'][i].to(self.device)
        
        self.input_data = data

    def forward(self, *args, **kwargs):
        assert self.input_data is not None
        
        # 1. 数据预处理
        data = self.detector.data_preprocessor(self.input_data, training=self.return_loss)
        inputs = data['inputs']
        data_samples = data['data_samples']

        # 2. 【TOOD 特色逻辑】注入 Epoch 属性
        for m in self.detector.modules():
            if hasattr(m, 'initial_epoch'):
                if m.initial_epoch is None:
                    m.initial_epoch = 4
                m.epoch = m.initial_epoch + 1
            if 'TOODHead' in m.__class__.__name__:
                setattr(m, 'epoch', 20)
                if getattr(m, 'initial_epoch', None) is None:
                    setattr(m, 'initial_epoch', 4)

        # 3. 执行转发
        if self.return_loss:
            # 计算模型的真实损失
            losses = self.detector(
                inputs, 
                data_samples=data_samples, 
                mode='loss')
            
            # 将字典中的所有 loss 项求和
            total_loss = 0
            for v in losses.values():
                if isinstance(v, (list, tuple)):
                    total_loss += sum(_loss for _loss in v)
                else:
                    total_loss += v
            
            # 【关键修复】: 不要直接返回 total_loss (0-d tensor)
            # 1. 增加维度变为 (1,) 2. 包装成 list 以适配 grad-cam 内部迭代
            return [total_loss.reshape(1)]
            
        else:
            # 预测模式逻辑
            with torch.no_grad():
                results = self.detector(
                    inputs, 
                    data_samples=data_samples, 
                    mode='predict')
                
                res = results[0]
                pred_instances = res.pred_instances
                
                bboxes = pred_instances.bboxes.detach().cpu().numpy()
                scores = pred_instances.scores.detach().cpu().numpy()
                labels = pred_instances.labels.detach().cpu().numpy()
                
                bboxes_with_scores = np.concatenate([bboxes, scores[:, None]], axis=1)
                
                masks = None
                if 'masks' in pred_instances:
                    masks = pred_instances.masks.detach().cpu().numpy()

                if self.score_thr > 0:
                    inds = scores > self.score_thr
                    bboxes_with_scores = bboxes_with_scores[inds]
                    labels = labels[inds]
                    if masks is not None:
                        masks = masks[inds]

                # 预测模式返回 2.x 兼容格式
                return [{'bboxes': bboxes_with_scores, 'labels': labels, 'segms': masks}]
    # def forward(self, *args, **kwargs):
    #     """3.x 模型调用方式：mode='predict' 或 'loss'"""
    #     assert self.input_data is not None
        
    #     if self.return_loss:
    #         # 模式切换为 loss
    #         losses = self.detector(
    #             self.input_data['inputs'], 
    #             data_samples=self.input_data['data_samples'], 
    #             mode='loss')
    #         return [losses]
    #     else:
    #         # 预测模式
    #         with torch.no_grad():
    #             results = self.detector(
    #                 self.input_data['inputs'], 
    #                 data_samples=self.input_data['data_samples'], 
    #                 mode='predict')
                
    #             # 3.x 返回的是 DetDataSample
    #             res = results[0]
    #             pred_instances = res.pred_instances
                
    #             # 过滤低分框
    #             bboxes = pred_instances.bboxes.detach().cpu().numpy()
    #             scores = pred_instances.scores.detach().cpu().numpy()
    #             labels = pred_instances.labels.detach().cpu().numpy()
                
    #             # 拼接 score 到 bboxes 后面，保持与 2.x 逻辑一致 [x1, y1, x2, y2, score]
    #             bboxes_with_scores = np.concatenate([bboxes, scores[:, None]], axis=1)
                
    #             masks = None
    #             if 'masks' in pred_instances:
    #                 masks = pred_instances.masks.detach().cpu().numpy()

    #             if self.score_thr > 0:
    #                 inds = scores > self.score_thr
    #                 bboxes_with_scores = bboxes_with_scores[inds]
    #                 labels = labels[inds]
    #                 if masks is not None:
    #                     masks = masks[inds]

    #             return [{'bboxes': bboxes_with_scores, 'labels': labels, 'segms': masks}]

# DetAblationLayer, DetCAMVisualizer, EigenCAM, FeatmapAM 等类基本逻辑不变
class DetAblationLayer(AblationLayer):

    def __init__(self):
        super(DetAblationLayer, self).__init__()
        self.activations = None

    def set_next_batch(self, input_batch_index, activations,
                       num_channels_to_ablate):
        """Extract the next batch member from activations, and repeat it
        num_channels_to_ablate times."""
        if isinstance(activations, torch.Tensor):
            return super(DetAblationLayer,
                         self).set_next_batch(input_batch_index, activations,
                                              num_channels_to_ablate)

        self.activations = []
        for activation in activations:
            activation = activation[
                input_batch_index, :, :, :].clone().unsqueeze(0)
            self.activations.append(
                activation.repeat(num_channels_to_ablate, 1, 1, 1))

    def __call__(self, x):
        """Go over the activation indices to be ablated, stored in
        self.indices.
        Map between every activation index to the tensor in the Ordered Dict
        from the FPN layer.
        """
        result = self.activations

        if isinstance(result, torch.Tensor):
            return super(DetAblationLayer, self).__call__(x)

        channel_cumsum = np.cumsum([r.shape[1] for r in result])
        num_channels_to_ablate = result[0].size(0)  # batch
        for i in range(num_channels_to_ablate):
            pyramid_layer = bisect.bisect_right(channel_cumsum,
                                                self.indices[i])
            if pyramid_layer > 0:
                index_in_pyramid_layer = self.indices[i] - channel_cumsum[
                    pyramid_layer - 1]
            else:
                index_in_pyramid_layer = self.indices[i]
            result[pyramid_layer][i, index_in_pyramid_layer, :, :] = -1000
        return result


class DetCAMVisualizer:
    """mmdet cam visualization class.
    Args:
        method:  CAM method. Currently supports
           `ablationcam`,`eigencam` and `featmapam`.
        model (nn.Module): MMDet model.
        target_layers (list[torch.nn.Module]): The target layers
            you want to visualize.
        reshape_transform (Callable, optional): Function of Reshape
            and aggregate feature maps. Defaults to None.
    """

    def __init__(self,
                 method_class,
                 model,
                 target_layers,
                 reshape_transform=None,
                 is_need_grad=False,
                 extra_params=None):
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.is_need_grad = is_need_grad

        if method_class.__name__ == 'AblationCAM':
            batch_size = extra_params.get('batch_size', 1)
            ratio_channels_to_ablate = extra_params.get(
                'ratio_channels_to_ablate', 1.)
            self.cam = AblationCAM(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
                batch_size=batch_size,
                ablation_layer=extra_params['ablation_layer'],
                ratio_channels_to_ablate=ratio_channels_to_ablate)
        else:
            self.cam = method_class(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
            )
            if self.is_need_grad:
                self.cam.activations_and_grads.release()

        # self.classes = model.detector.CLASSES
        # 优先获取 CLASSES 属性，如果没有（3.x环境），则从 dataset_meta 中获取
        # 检查包装器内部的 detector 是否有元数据
        if hasattr(model.detector, 'CLASSES'):
            self.classes = model.detector.CLASSES
        elif hasattr(model.detector, 'dataset_meta'):
            self.classes = model.detector.dataset_meta.get('classes', [])
        else:
            self.classes = []
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def switch_activations_and_grads(self, model):
        self.cam.model = model

        if self.is_need_grad is True:
            self.cam.activations_and_grads = ActivationsAndGradients(
                model, self.target_layers, self.reshape_transform)
            self.is_need_grad = False
        else:
            self.cam.activations_and_grads.release()
            self.is_need_grad = True

    def __call__(self, img, targets, aug_smooth=False, eigen_smooth=False):
        img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)
        return self.cam(img, targets, aug_smooth, eigen_smooth)[0, :]

    def show_cam(self,
                 image,
                 boxes,
                 labels,
                 grayscale_cam,
                 with_norm_in_bboxes=False):
        """Normalize the CAM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""
        if with_norm_in_bboxes is True:
            boxes = boxes.astype(np.int32)
            renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
            images = []
            for x1, y1, x2, y2 in boxes:
                img = renormalized_cam * 0
                img[y1:y2,
                    x1:x2] = scale_cam_image(grayscale_cam[y1:y2,
                                                           x1:x2].copy())
                images.append(img)

            renormalized_cam = np.max(np.float32(images), axis=0)
            renormalized_cam = scale_cam_image(renormalized_cam)
        else:
            renormalized_cam = grayscale_cam

        cam_image_renormalized = show_cam_on_image(
            image / 255, renormalized_cam, use_rgb=False)

        image_with_bounding_boxes = self._draw_boxes(boxes, labels,
                                                     cam_image_renormalized)
        return image_with_bounding_boxes

    def _draw_boxes(self, boxes, labels, image):
        for i, box in enumerate(boxes):
            label = labels[i]
            color = self.COLORS[label]
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color, 2)
            cv2.putText(
                image,
                self.classes[label], (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA)
        return image
# 但在 DetBoxScoreTarget 中需要针对 3.x 的结果字典进行微调

class DetBoxScoreTarget:
    def __init__(self, bboxes, labels, segms=None, match_iou_thr=0.5, device='cuda:0'):
        self.focal_bboxes = torch.from_numpy(bboxes).to(device)
        self.focal_labels = labels
        self.focal_segms = segms
        self.match_iou_thr = match_iou_thr
        self.device = device

    def __call__(self, results):
        output = torch.tensor([0.], device=self.device)
        
        # 处理 Loss 情况 (Grad-based)
        if isinstance(results[0], dict) and 'loss_cls' in results[0]:
            for loss_value in results[0].values():
                if isinstance(loss_value, torch.Tensor):
                    output += loss_value.sum()
                elif isinstance(loss_value, list):
                    output += sum([l.sum() for l in loss_value])
            return output

        # 处理预测情况 (Grad-free)
        res = results[0]
        if len(res['bboxes']) == 0:
            return output

        pred_bboxes = torch.from_numpy(res['bboxes']).to(self.device)
        pred_labels = res['labels']
        pred_segms = res['segms']

        for focal_box, focal_label in zip(self.focal_bboxes, self.focal_labels):
            import torchvision
            ious = torchvision.ops.box_iou(focal_box[None, :4], pred_bboxes[:, :4])
            index = ious.argmax()
            
            if ious[0, index] > self.match_iou_thr and pred_labels[index] == focal_label:
                # 分数 = IoU + 分类得分
                score = ious[0, index] + pred_bboxes[index, 4]
                output += score
        return output
class EigenCAM(BaseCAM):

    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None):
        super(EigenCAM, self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            uses_gradients=False)

    def get_cam_image(self, input_tensor, target_layer, target_category,
                      activations, grads, eigen_smooth):
        return get_2d_projection(activations)


class FeatmapAM(EigenCAM):
    """Visualize Feature Maps.
    Visualize the (B,C,H,W) feature map averaged over the channel dimension.
    """

    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None):
        super(FeatmapAM, self).__init__(model, target_layers, use_cuda,
                                        reshape_transform)

    def get_cam_image(self, input_tensor, target_layer, target_category,
                      activations, grads, eigen_smooth):
        return np.mean(activations, axis=1)