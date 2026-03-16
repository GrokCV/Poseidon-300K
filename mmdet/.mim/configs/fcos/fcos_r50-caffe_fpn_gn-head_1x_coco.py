_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]#1*调度意味着在模型训练一个 epoch 之后，学习率会减小 10 倍
#default_runtime.py：该文件包含默认的运行时配置，例如要使用的 GPU 数量、要训练模型的 epoch 数量以及用于保存检查点和日志的输出目录。
# model settings
custom_imports = dict(
    imports=['mmdet.models.utils.soft_hist'],
    allow_failed_imports=False
)

model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),  #图形的长和宽被pad到32的倍数
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4, #骨干网路输出4个feature map
        out_indices=(0, 1, 2, 3),  #骨干网路中输出feature map的stages的索引
        frozen_stages=1,  #冻结骨干网络中前 1 个 (第0，1层)stage 的权重，不参与训练。
# 通常会选择冻结骨干网络中的一部分或全部层的权重，不参与微调过程，只更新分类器或检测头等任务特定的权重。这是因为骨干网络中的低层特征对于各种任务都是通用的，而高层特征则更加具有任务特异性
        norm_cfg=dict(type='BN', requires_grad=False),#归一化层的权重不参与训练
        norm_eval=True,
        style='caffe',
        init_cfg=dict(  #权重初始化方式为与训练，使用模型来自open-mmlab上的库中的模型
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
   
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  #表示resnet中2，3，4，5个阶段的输出通道数
        out_channels=256, #特征金字塔网络输出的特征图通道为256
        start_level=1, #骨干网路输出的第二个特征图开始进行处理
        add_extra_convs='on_output',  # use P5 ，在特征金字塔网络的最后一层输出特征图上添加额外的卷积层，p5层
        num_outs=5,#特征金字塔网络输出的特征图个数为5
        relu_before_extra_convs=True),  #额外的卷积层之前添加relu激活函数
    bbox_head=dict(
        type='FCOSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,   # 4 个卷积层来进行特征提取和处理
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],   #特征图的步长，下采样倍数分别为8、16、32、64、128
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),   #分类损失在总损失中的权重，1表示分类和回归损失权重相等
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),  #回归损失
        loss_centerness=dict(  #中心度损失
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000, #进行nms'之前保留的预测狂的数量，这里是1000
        min_bbox_size=0, # 预测框的最小尺寸，小于该尺寸的预测框将会被过滤掉，
        score_thr=0.05,  #低于该阈值的预测框将会被过滤掉，
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))  #每张图片最多保留100给预测狂

# learning rate
param_scheduler = [
    dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    # ConstantLR 是一种常数学习率调度器，用于在训练过程中保持恒定的学习率，即不随着训练进程而动态调整学习率。前期以便让模型更好地学习数据集的特征
    ## factor 表示初始学习率相对于默认学习率的比例，这里设置为 1/3。
    #by_epoch 表示是否按照 epoch 进行调整，这里设置为 False，表示按照 iteration 进行调整。
    dict(
        type='MultiStepLR',  #表示多步骤学习率调度器
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],  #表示学习率体调整的阶段点，表示在第 8 和 11 个 epoch 时调整学习率
        gamma=0.1)   # 每次调整后学习率将缩小为原来的 0.1
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.), #分别设置了偏置项的学习率倍数和正则化系数
    clip_grad=dict(max_norm=35, norm_type=2)) #max_norm 表示梯度的最大范数，norm_type 表示计算范数的方式为l2
 #用于控制梯度的大小，以避免梯度爆炸或消失问题