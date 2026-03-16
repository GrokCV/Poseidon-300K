# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod
from inspect import signature
from typing import List, Optional, Tuple

import torch
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, constant_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmdet.utils import InstanceList, OptMultiConfig
from ..test_time_augs import merge_aug_results
from ..utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads.

    1. The ``init_weights`` method is used to initialize densehead's
    model parameters. After detector initialization, ``init_weights``
    is triggered when ``detector.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of densehead,
    which includes two steps: (1) the densehead model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.
    计算 DenseHead 的损失，包括两个步骤：(1) DenseHead 模型执行前向传播以获取特征图
      (2) 基于特征图调用 loss_by_feat 方法来计算损失。

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict detection results,
    which includes two steps: (1) the densehead model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict detection results including
    post-processing.预测检测结果，包括两个步骤：(1) DenseHead 模型执行前向传播以获取特征图 
    (2) 基于特征图调用 predict_by_feat 方法来预测检测结果，包括后处理。

    .. code:: text

    predict(): forward() -> predict_by_feat()

    4. The ``loss_and_predict`` method is used to return loss and detection
    results at the same time. It will call densehead's ``forward``,
    ``loss_by_feat`` and ``predict_by_feat`` methods in order.  If one-stage is
    used as RPN, the densehead needs to return both losses and predictions.
    This predictions is used as the proposal of roihead.
同时返回损失和检测结果。它将依次调用 DenseHead 的 forward、loss_by_feat 和 predict_by_feat 方法。
如果一阶段检测器用作 RPN（Region Proposal Network），DenseHead 需要同时返回损失和预测结果。这些预测结果将用作 RoIHead 的提议。
    .. code:: text

    loss_and_predict(): forward() -> loss_by_feat() -> predict_by_feat()
    """

    def __init__(self, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # `_raw_positive_infos` will be used in `get_positive_infos`, which
        # can get positive information.
        self._raw_positive_infos = dict()  #用于存储正样本的信息

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():  #遍历当前模块的所有子模块。
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'): #检查子模块是否有 conv_offset 属性（是可变形卷积（Deformable Convolution）中的一个偏移量参数。）
                constant_init(m.conv_offset, 0)  #constant_init 是一个辅助函数，用于将张量初始化为常数值。

    def get_positive_infos(self) -> InstanceList:
        """Get positive information from sampling results.

        Returns:
            list[:obj:`InstanceData`]: Positive information of each image,
            usually including positive bboxes, positive labels, positive
            priors, etc.
        """
        if len(self._raw_positive_infos) == 0:
            return None

        sampling_results = self._raw_positive_infos.get(
            'sampling_results', None)  #从 _raw_positive_infos 字典中获取 sampling_results
        assert sampling_results is not None
        positive_infos = []
        for sampling_result in enumerate(sampling_results):
            pos_info = InstanceData()  #InstanceData 对象 pos_info的属性
            pos_info.bboxes = sampling_result.pos_gt_bboxes  #正样本边界框
            pos_info.labels = sampling_result.pos_gt_labels
            pos_info.priors = sampling_result.pos_priors  #正样本先验眶
            pos_info.pos_assigned_gt_inds = \
                sampling_result.pos_assigned_gt_inds  #正样本分配的真实边界框的真实索引
            pos_info.pos_inds = sampling_result.pos_inds  #正样本索引
            positive_infos.append(pos_info)
        return positive_infos  #返回包含每张图像的正样本信息的列表 positive_info

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    @abstractmethod
    def loss_by_feat(self, **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head."""
        pass

    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.一个可选的 ConfigDict，用于测试或后处理配置。如果为 None，则使用 test_cfg。默认为 None。

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        outputs = unpack_gt_instances(batch_data_samples)  #通过辅助函数，用于从 batch_data_samples 中提取真实标注信息。
        (batch_gt_instances, batch_gt_instances_ignore,  #batch_gt_instances：包含每个图像的真实边界框和类别标签。batch_gt_instances_ignore：包含每个图像的忽略边界框。
         batch_img_metas) = outputs

        outs = self(x)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)  #把预测信息和真实信息传递给loss_by_feat

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)
        return losses, predictions

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
                每个数据样本通常包含图像的元信息（如 img_shape、scale_factor 等）和
                真实标注信息（如 gt_instance、gt_panoptic_seg 和 gt_sem_seg）。
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        #提取元信息
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
#前向传播
        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]  #获取每个维度的高宽
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)  #【(num_priors * num_classes, H, W).*5】-> (num_priors * num_classes, H, W)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape从所有尺度级别提取的分类分数
                (num_priors * num_classes, H, W).num_priors是每个点先眼眶的数量
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).从所有尺度级别提取的边界框预测（偏移量或能量）
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape从所有尺度级别提取的分数因子
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is特征金字塔中每个级别的先验框
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if score_factor_list[0] is None:  #检查是否使用分数因子
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True
#获取一些配置和图像元信息
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)  #表示在 NMS 前保留的边界框数量

        mlvl_bbox_preds = []  #用于存储每个尺度级别的边界框预测。
        mlvl_valid_priors = []  #用于存储每个尺度级别的有效先验框。
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []  #存储每个尺度级别的分数因子
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]  #高宽相等
#1.调整维度信息
            dim = self.bbox_coder.encode_size  #4
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)  #形状从 (C,H,W) 转换为 (H×W×num priors,4)
            #处理分数因子
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,  #对分数因子应用 Sigmoid 激活函数。
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels) #将 cls_score 的形状从 (C,H,W) 转换为 (H×W×num priors,num c lasses)。

            # the `custom_cls_channels` parameter is derived from
            # CrossEntropyCustomLoss and FocalCustomLoss, and is currently used
            # in v3det.
#2.计算分类分数
            if getattr(self.loss_cls, 'custom_cls_channels', False):
                scores = self.loss_cls.get_activation(cls_score)
            elif self.use_sigmoid_cls:  #如果使用 Sigmoid 激活函数，对每个类别分数应用 Sigmoid。
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class对每个类别分数应用 Softmax，并去掉背景类别。
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)  #score_thr：分数阈值，用于筛选置信度分数。
#根据分数阈值筛选分数达标的results
            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,   #使用 filter_scores_and_topk 函数筛选分数高于阈值的候选框，并保留前nms_pre个候选框。
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results  #返回筛选后的分数、标签、保留的索引和筛选后的结果。
#根据上面两行代码，更新边界框预测和先验框
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]
#收集所有尺度级别的结果
            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)  #针对先验框的cat
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,   #边界框、置信度、标签等信息
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.测试或后处理配置
            rescale (bool): If True, return boxes in original image space.
                Default to False.表示是否将边界框调整到原始图像尺度。默认为 False。
            with_nms (bool): If True, do nms before return boxes.
                Default to True.表示是否应用非极大值抑制（NMS）。默认为 True
            img_meta (dict, optional): Image meta info. Defaults to None.图像元信息，类型为字典

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]  #从图像元信息中获得图像的缩放因子 scale_factor
            results.bboxes = scale_boxes(results.bboxes, scale_factor)  #使用 scale_boxes 函数将边界框的坐标调整到原始图像尺度。

        if hasattr(results, 'score_factors'):
            # TODO: Add sqrt operation in order to be consistent with
            # #如果 results 中包含 score_factors 属性，将其取出并与 results.scores 相乘，更新置信度分数。
            #  the paper.score_factors 通常用于调整置信度分数，例如在某些目标检测算法中，可能会对置信度分数进行加权。
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)  #使用 get_box_tensor 函数将边界框转换为张量。
            #调用 batched_nms 函数进行批量 NMS，返回保留的边界框和对应的索引
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs] #更新 results，保留通过 NMS 的边界框。
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]  #更新置信度分数，因为某些 NMS 实现可能会重新计算分数。
            results = results[:cfg.max_per_img]  #根据配置中的 max_per_img 参数，限制每张图像的检测结果数量。

        return results

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation.

        Args:
            aug_batch_feats (list[tuple[Tensor]]): The outer list
                indicates test-time augmentations and inner tuple
                indicate the multi-level feats from包含不同测试增强下的多尺度特征
                FPN, each Tensor should have a shape (B, C, H, W),
            aug_batch_img_metas (list[list[dict]]): Meta information
                of images under the different test-time augs包含不同测试增强下的图像元信息
                (multiscale, flip, etc.). The outer list indicate
                the
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            with_ori_nms (bool): Whether execute the nms in original head.
                Defaults to False. It will be `True` when the head is
                adopted as `rpn_head`.

        Returns:
            list(obj:`InstanceData`): Detection results of the
            input images. Each item usually contains\
            following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance,)
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances,).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # TODO: remove this for detr and deformdetr
        sig_of_get_results = signature(self.get_results)
        get_results_args = [
            p.name for p in sig_of_get_results.parameters.values()
        ]
        get_results_single_sig = signature(self._get_results_single)
        get_results_single_sig_args = [
            p.name for p in get_results_single_sig.parameters.values()
        ]
        assert ('with_nms' in get_results_args) and \
               ('with_nms' in get_results_single_sig_args), \
               f'{self.__class__.__name__}' \
               'does not support test-time augmentation '

        num_imgs = len(aug_batch_img_metas[0])
        aug_batch_results = []
        for x, img_metas in zip(aug_batch_feats, aug_batch_img_metas):
            outs = self.forward(x)
            batch_instance_results = self.get_results(
                *outs,
                img_metas=img_metas,
                cfg=self.test_cfg,
                rescale=False,
                with_nms=with_ori_nms,
                **kwargs)
            aug_batch_results.append(batch_instance_results)

        # after merging, bboxes will be rescaled to the original image
        batch_results = merge_aug_results(aug_batch_results,
                                          aug_batch_img_metas)

        final_results = []
        for img_id in range(num_imgs):
            results = batch_results[img_id]
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                                results.labels,
                                                self.test_cfg.nms)
            results = results[keep_idxs]
            # some nms operation may reweight the score such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:self.test_cfg.max_per_img]
            if rescale:
                # all results have been mapped to the original scale
                # in `merge_aug_results`, so just pass
                pass
            else:
                # map to the first aug image scale
                scale_factor = results.bboxes.new_tensor(
                    aug_batch_img_metas[0][img_id]['scale_factor'])
                results.bboxes = \
                    results.bboxes * scale_factor

            final_results.append(results)

        return final_results
