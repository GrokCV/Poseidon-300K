# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.layers import NormedConv2d
from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from ..utils import multi_apply
from .anchor_free_head import AnchorFreeHead

INF = 1e8


@MODELS.register_module()
class FCOSHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions. FCOS head不使用锚框。相反，在每个像素上预测边界框，\
    并使用中心度（centerness）度量来抑制低质量的预测。
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.
    这里的 `norm_on_bbox`、`centerness_on_reg`、`dcn_on_last_conv` \
    是官方代码库中使用的训练技巧，这些技巧可以显著提高 mAP，最高可达 4.9。
    详情请参见 https://github.com/tianzhi0549/FCOS。
    Args:
        num_classes (int): Number of categories excluding the background
            category.不包括背景类别的类别数量。
         in_channels (int): 输入特征图中的通道数量。
        strides (Sequence[int] 或 Sequence[Tuple[int, int]]): 多个特征层中点的步幅。默认值为 (4, 8, 16, 32, 64)。
        regress_ranges (Sequence[Tuple[int, int]]): 多级点的回归范围。规定了每层特征所预测的物体尺度范围，只有某一层对应尺寸范围内的目标才会被该层标记为正样本;
        center_sampling (bool): 如果为 True，使用中心采样。默认值为 False。是否只将物体中央区域的 Keypoint 设置为 GT,
        center_sample_radius (float): 中心采样的半径。默认值为 1.5。
         norm_on_bbox (bool): 如果为 True，则用 FPN 步幅对回归目标进行归一化。默认值为 False。
        centerness_on_reg (bool): 如果为 True，在回归分支上计算中心度。\
        	详情请参见 \这个变量用于控制 centerness branch 是用 classification branch 还是regression branch 的特征来预测。
        	https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042。\
        	默认值为 False。
        conv_bias (bool 或 str): 如果设置为 `auto`，将由 `norm_cfg` 决定。\
        	如果 `norm_cfg` 为 None，则卷积的偏差设置为 True，\
        	否则为 False。默认值为 "auto"。
        loss_cls (:obj:`ConfigDict` 或 dict): 分类损失的配置。
        loss_bbox (:obj:`ConfigDict` 或 dict): 定位损失的配置。
        loss_centerness (:obj:`ConfigDict` 或 dict): 中心度损失的配置。
        norm_cfg (:obj:`ConfigDict` 或 dict): 用于构建和配置归一化层的字典。默认值为\
        	``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``。
        cls_predictor_cfg (:obj:`ConfigDict` 或 dict): 用于构建和配置\
        	`conv_cls` 的字典。默认值为 None。
        init_cfg (:obj:`ConfigDict` 或 dict 或 list[:obj:`ConfigDict` 或 dict]):\
        	初始化配置字典。
    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                                              (256, 512), (512, INF)),
                 center_sampling: bool = False,
                 center_sample_radius: float = 1.5,
                 norm_on_bbox: bool = False,
                 centerness_on_reg: bool = False,
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 cls_predictor_cfg=None,
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.cls_predictor_cfg = cls_predictor_cfg
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = MODELS.build(loss_centerness)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)#输出通道为1，初始化中心度预测层
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])  #初始化每个特征层的尺度调整模块
         # self.strides 指的是特征图下采样的步长，每个步长对应一个 Scale 层。
        if self.cls_predictor_cfg is not None:
            self.cls_predictor_cfg.pop('type')
            self.conv_cls = NormedConv2d(
                self.feat_channels,
                self.cls_out_channels,
                1,
                padding=0,
                **self.cls_predictor_cfg)

    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """将来自上游网络的特征进行前向传递

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:一个包含各层级输出的元组。

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.每个尺度级别的框得分
            - bbox_preds (list[Tensor]): 每个尺度级别的框能量/偏移量, \
            	每个是一个4D张量，通道数为num_points * 4。 \
            - centernesses (list[Tensor]): 每个尺度级别的中心点度, \
            	每个是一个4D张量，通道数为num_points * 1。
                将5个feature map 依次 输入forward_single
        input: 
            [(N, 256, x, y)*5]   维度1:图片数量n; 维度2:通道数c; 维度3:高h; 维度4:宽w
        output:
            cls_score = [(N, 80, h, w)*5]
            bbox_pred = [(N, 4, h, w)*5],
            conv_centerness = [(N, 1, h, w)*5]
        """
        return multi_apply(self.forward_single, x, self.scales, self.strides)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward features of a single scale level.对单尺度级别的特征进行前向传递。

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj:`mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.可学习的缩放模块，用于调整边界框预测的尺寸。
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness
            predictions of input feature maps. 输入特征图的每个类别的得分、边界框预测和中心点度预测。
            对某一level的feature map进行计算
        input: 
            x:([N, 256, h, w]*5) 每个Level的feature map, 对应stride
            scale: ModuleList(Scale()*5) 可学习的参数用于调整预测的BBOX
            stride:[8, 16, 32, 64, 128];  
        output:
            cls_score = [N, 80, h, w]
            bbox_pred = [N, 4, h, w]
            conv_centerness = [N, 1, h, w]
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
          # STEP 3：对Bbox的预测结果进行优化
        # Tricks 1:这里 scale 是一个 nn.Parameter(),初始化为 1，\
        # 每一个特征图尺度都有一个 scale
        # 就是防止网络每一层需要回归的范围变化太大，而检测头的卷积是共享的，\
        # 因此让网络自己学习一个参数，避免FP16时溢出
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)orch.clamp 函数的作用是限幅，将 input 的值限制在 [min, max] 之间，并返回结果，bbox_pred.clamp(min=0)
            #其实就是 ReLU，但因为某些原因，不能再用 F.relu 了，所以用 bbox_pred.clamp(min=0) 来代替。要限制成大于等于 0 是很容易理解的，因为距离 tlbr 一定都是非负的
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride  #不在训练时，恢复图像尺寸
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): 每个尺度级别的框得分，\
            	每个是一个4D张量，通道数为num_points * num_classes
            bbox_preds (list[Tensor]): 每个尺度级别的框能量/偏移量，\
            	每个是一个4D张量，通道数为num_points * 4.
            centernesses (list[Tensor]): 每个尺度级别的中心点度，\
            	每个是一个4D张量，通道数为num_points * 1.
            batch_gt_instances (list[:obj:`InstanceData`]): gt_instance的batch。\
            	通常包括bboxes（边界框）和labels（标签）属性。 
            batch_img_metas (list[dict]):  每张图像的元信息，\
            	例如图像大小、缩放因子等。
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. 被忽略的真实实例批次。\
                包括在训练和测试期间被忽略的bboxes属性数据。默认为None。

        Returns:
            dict[str, Tensor]: 损失组件的字典。
        """
        #此处长度是经过FPN的特征层数
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores] 
        #torch.Size([batch_size, num_points * num_classes, height, width])去最后俩个维度
        '''featmap_sizes: 这是一个包含多个特征图尺寸的列表，每个尺寸通常以\
        (height, width) 的元组形式表示。这个参数用于指定每个特征图级别的大小，\
        以便生成对应位置的网格点。'''
        #1.生成所有特征层的先验点
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype, #取了FPN的五个输出中的第一个输出
            device=bbox_preds[0].device)
        #all_level_points是一个 list[Tensor],每个张量的形状为 (height * width, 2) (with_stride默认False)
        '''prior_generator.grid_priors
        
        继承自mmdet/models/dense_heads/anchor_free_head.py的AnchorFreeHead类里的\
        self.prior_generator = MlvlPointGenerator(strides)\
        源于mmdet/models/task_modules/prior_generators/point_generator.py\
        的MlvlPointGenerator类\
        
        MlvlPointGenerator是一个用于在 2D 点检测器中生成多级特征图的标准点生成器。\
        它的主要功能是根据给定的特征图尺寸和步幅（stride）生成网格点，\
        这些点用于后续的目标检测任务。\
        
        主要参数 strides: 包含每个特征图级别的步幅，\
        可以是单个整数（统一步幅）或元组（宽度和高度的步幅）。\
        offset: 点的偏移量，默认值为 0.5，表示生成点的位置相对于特征图的中心。
     
        grid_priors: 生成多个特征级别的网格点。\
        它会调用 single_level_grid_priors 为每个特征图级别生成点。
        '''
#2.得到每一张图片中所有anchor point分配的标签（正负样本）
        labels, bbox_targets = self.get_targets(all_level_points,
                                                batch_gt_instances)
#3.一个batch中的图片数量
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness做一个展平处理
        flatten_cls_scores = [  # # flatten_cls_scores = [(h*w*n, 80)*5]
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [     # flatten_bbox_preds = [(h*w*n, 4)*5]
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [     # # flatten_bbox_preds = [(h*w*n)*5]
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores) # (5 * (h*w*n), 80)，默认在dim=0的维度进行cat
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels) #维度可能是 (5 * (h*w*n), )
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds通过重复先验框点以匹配每个图像的数量。
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
#4.
        losses = dict()

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)  #通过nonzero()找出索引
        #gt的正样本在0-num_classes-1之间的正样本索引
        num_pos = torch.tensor(  #计算正样本的数量
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)#使用 reduce_mean 确保在多 GPU 训练时能正确汇总。
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)
 # 计算分类损失，使用正样本数量作为平均因子。
        # FocalLoss
        if getattr(self.loss_cls, 'custom_accuracy', False):  #检查 loss_cls 是否有自定义的准确率计算方法
            acc = self.loss_cls.get_accuracy(flatten_cls_scores,
                                             flatten_labels)
            losses.update(acc)

        pos_bbox_preds = flatten_bbox_preds[pos_inds] #是落在 GT BBox 中心区域内部的那些特征点预测出的 pred bbox，shape 为 (num_pos, 4)
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
 # 计算用于中心点度损失的归一化因子，防止为 0 的情况。
        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]  #正样本点，落在GT BBox 中心区域内部的那些特征点
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds) #就是把ltrb解码成x1 y1 x2 y2
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        losses['loss_cls'] = loss_cls
        losses['loss_bbox'] = loss_bbox
        losses['loss_centerness'] = loss_centerness

        return losses

    def get_targets(
            self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """为多张图像中的点计算回归、分类和中心点度目标。

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. 通常包括bboxes（边界框）和labels（标签）属性。

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
            level.
        """
        assert len(points) == len(self.regress_ranges)
         # 确保输入的点数和回归范围的数量一致。
          # tensor1,  # 对应第一级特征图，形状为 (h1*w1, 2)
        # tensor2,  # 对应第二级特征图，形状为 (h2*w2, 2)
        # tensor3   # 对应第三级特征图，形状为 (h3*w3, 2)
        # ...
        # ]
        # 2 表示点的坐标 (x, y)。
        # regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),\
        # (256, 512), (512, INF))
        #1.获取特征图level，此处经过fpn后是5
        num_levels = len(points) #5
        # expand regress ranges to align with points
        #2.根据anchor point 所在level生成用于确定其回归的最大和最小范围
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # 创建一个新张量，以匹配每个级别的点，便于后续操作。
        # 相当于对应的每个级别(num_points, 2)个对应的regress_ranges
        # [
        #   [[-1, 64],
        #	[-1, 64],
        #	...],  # 第一级别，形状为 (num_points1, 2)
        #	[[64, 128],
        #	[64, 128],
        #	[64, 128],
        #	...]   # 第二级别，形状为 (num_points2, 2)
        #	...
        # ]


        # concat all levels points and regress ranges
        #3.将这张图片上的所有 anchor point 和他们回归的范围 concat 起来,\类似于上面的flatten这样点和点的回归范围就可以一一对应起来了
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        # 分别连接所有级别的回归范围和点，形成统一的张量。
        # concat_regress_ranges = [
        #     [-1, 64],
        #     [-1, 64],
        #     ...,
        #     [64, 128],
        #     [64, 128],
        #     [64, 128],
        #	  ...
        # ] #类似于这样
        concat_points = torch.cat(points, dim=0)

        # 4.the number of points per img, per lvl一张图片上每一level特征图的 anchor point 个数
        num_points = [center.size(0) for center in points]

        # 5.get labels and bbox_targets of each image得到每个anchor point的gt label（被分配的标签）和bbox
        labels_list, bbox_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # 6.split to per img, per level
        #labels.split(num_points, 0) 返回的是一个 tuple(Tensor), 每个 Tensor 代表一幅图像的每个层级的labels, 每个 Tensor 的大小为 (H_i*W_i,), i 表示 FPN 的第 i 层
        #就是根据level的每层的point数量进行划分，划分成多个tensor
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # 7.concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(  #把每个level 的标签拼在一起，就是第一层FPN的labe全放一起，第二层的全放一起。。。
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:  #如果 self.norm_on_bbox 为真，将边界框目标除以相应的步幅，以进行标准化
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        #1.获取point数量、qt的bbox和label信息
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
#如果没有真实目标，返回每个点的标签为背景类，并返回零边界框。
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))
 # STEP 2: 计算出该图片中每一个 gt_box 的面积。每个边界框的四个坐标（x_min, y_min, x_max, y_max）。
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
         # 将面积扩展到与每个点的数量一致，形成一个 (num_points, num_gts) 的张量。
         #之所以这样扩展是因为后面每一个 anchor 都要\
        # 和每一个 gt_box 进行计算，看看匹配哪一个 gt_box，这样子的话方便很多
        '''假设有 3 个真实边界框（num_gts = 3），则 areas 的初始值可能如下：\
        areas = torch.tensor([200.0, 150.0, 300.0])  # 形状为 (3,)\
        当执行 areas[None] 时，areas 的形状会变成 (1, 3)：\
        areas = areas[None]  # 形状变为 (1, 3)\
        # 现在的值为：\
        # tensor([[200.0, 150.0, 300.0]])\
        执行 areas.repeat(num_points, 1)\
        假设 num_points = 5，则执行 repeat 后，areas 的形状会变为 (5, 3)：\
        num_points = 5\
        areas = areas.repeat(num_points, 1)  # 形状变为 (5, 3)\
        # 现在的值为：\
        # tensor([[200.0, 150.0, 300.0],\
        #         [200.0, 150.0, 300.0],\
        #         [200.0, 150.0, 300.0],\
        #         [200.0, 150.0, 300.0],\
        #         [200.0, 150.0, 300.0]])\
        总结 在 areas 经过这两步操作后，它从 (3,) 变成了 (1, 3)，再变成了 (5, 3)。\
        这样做的目的是为了在后续计算中，能够为每个点对应多个真实边界框的面积信息。
        '''
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
         # 形状从(concat_num_points, 2)变为 (concat_num_points, 1, 2)，\
        # 再变为(concat_num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
#每个点距离gt的ltrb
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)  #(num_points, num_gts，4)
#只有当 anchor point 在 gt_box 的中间一小部分的时候才算作正样本，\
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
             # 初始化一个与 gt_bboxes 相同形状的张量，用于存储中心边界框的坐标。
       
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride  
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)
#上面就是计算中心区域的范围得到center_gts
            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys #计算每个point点到中心区域的距离
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0   #确定正样本点
            # .min(-1)操作，是在沿着最后一个维度（即坐标维度）寻找最小值。
            # 即距离的最小值部分，得到一个形状为 (num_points,) 的 tensor，\
            # 表示每个点到中心边界框四个边的最小距离。
            # 检查每个点是否在中心边界框内，如果最小距离大于0，表示该点在中心边界框内。
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location计算每个点的最大回归距离，并检查是否在回归范围内。
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF   #将不与这个 anchor point 匹配的 gt_box 的 area 置为无穷大，\
        areas[inside_regress_range == 0] = INF
        #找到每一个 anchor 匹配的 gt_box (面积最小的那个)的索引
        min_area, min_area_inds = areas.min(dim=1)
# 确定每一个anchor的gt bbox的类别标签,正样本类别/负样本
        labels = gt_labels[min_area_inds]
        # # 负样本的标签为背景
        labels[min_area == INF] = self.num_classes  # set as BG
        # STEP 11: 确定每一个 anchor的gt bbox
        # [range(num_points), min_area_inds] 这个写法很妙，\
        # 将 (num_points, num_gts, 4)  -> (num_points, 4)
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets  # # 返回每个点的标签和边界框目标。

    def centerness_target(self, pos_bbox_targets: Tensor) -> Tensor:
        """Compute centerness targets.
正样本点的gt的ltrb
        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
