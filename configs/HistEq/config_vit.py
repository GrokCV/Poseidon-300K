METAINFO = dict(
    classes=(
        'holothurian',
        'echinus',
        'scallop',
        'starfish',
        'fish',
        'diver',
        'cuttlefish',
        'turtle',
        'jellyfish',
        'crab',
        'shrimp',
        'plastic trash',
        'rov',
        'fabric trash',
        'fishing trash',
        'metal trash',
        'paper trash',
        'rubber trash',
        'wood trash',
    ),
    palette=[
        (
            220,
            20,
            60,
        ),
        (
            119,
            11,
            32,
        ),
        (
            0,
            0,
            142,
        ),
        (
            0,
            0,
            230,
        ),
        (
            106,
            0,
            228,
        ),
        (
            0,
            60,
            100,
        ),
        (
            0,
            80,
            100,
        ),
        (
            0,
            0,
            70,
        ),
        (
            0,
            0,
            192,
        ),
        (
            250,
            170,
            30,
        ),
        (
            100,
            170,
            30,
        ),
        (
            220,
            220,
            0,
        ),
        (
            175,
            116,
            175,
        ),
        (
            250,
            0,
            30,
        ),
        (
            165,
            42,
            42,
        ),
        (
            255,
            77,
            255,
        ),
        (
            0,
            226,
            252,
        ),
        (
            182,
            182,
            255,
        ),
        (
            0,
            82,
            0,
        ),
    ])
auto_resume = False
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet.models.backbones.vit_adapter',
        'mmdet.mmcv_custom.layer_decay_optimizer_constructor_intertvit_adp',
    ])
data_root = 'datasets/Poseidon-300K/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(_scope_='mmdet', interval=1, type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
device = 'cuda'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'work_dirs/tood_r50_fpn_1x_poseidon/20260127_160415/epoch_12.pth'
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        cffn_ratio=0.25,
        deform_ratio=0.25,
        depth=24,
        drop_path_rate=0.1,
        embed_dim=1024,
        freeze_vit=False,
        img_size=512,
        init_values=0.1,
        interaction_indexes=[
            [
                0,
                7,
            ],
            [
                8,
                11,
            ],
            [
                12,
                15,
            ],
            [
                16,
                23,
            ],
        ],
        layerscale_force_fp32=False,
        mlp_ratio=4.0,
        norm_type='layer_norm',
        num_heads=16,
        only_feat_out=True,
        patch_size=16,
        pretrain_size=448,
        pretrained='pretrained/model.safetensors',
        pretrained_type='full',
        qk_normalization=False,
        qkv_bias=True,
        type='InternViTAdapter',
        use_final_norm=True,
        use_flash_attn=False,
        with_cp=True,
        with_fpn=False),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=8,
            ratios=[
                1.0,
            ],
            scales_per_octave=1,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        anchor_type='anchor_free',
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        initial_loss_cls=dict(
            activated=True,
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            activated=True,
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        num_classes=19,
        stacked_convs=6,
        type='TOODHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    he_module=dict(
        alpha_init=0.25,
        detach_hist=True,
        downsample=None,
        group=1,
        num_bins=512,
        type='LearnableHistEq'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            1024,
            1024,
            1024,
            1024,
        ],
        num_outs=5,
        out_channels=256,
        start_level=0,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        alpha=1,
        assigner=dict(topk=13, type='TaskAlignedAssigner'),
        beta=6,
        debug=False,
        initial_assigner=dict(topk=9, type='ATSSAssigner'),
        initial_epoch=4,
        pos_weight=-1),
    type='TOOD')
optim_wrapper = dict(
    constructor='InternViTAdapterLayerDecayOptimizerConstructor',
    dtype='float16',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=2.5e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(layer_decay_rate=0.95, num_layers=24),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        _scope_='mmdet',
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        _scope_='mmdet',
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='annotations/COCO/test.json',
        data_prefix=dict(img='images/test/'),
        data_root='datasets/Poseidon-300K/',
        metainfo=dict(
            classes=(
                'holothurian',
                'echinus',
                'scallop',
                'starfish',
                'fish',
                'diver',
                'cuttlefish',
                'turtle',
                'jellyfish',
                'crab',
                'shrimp',
                'plastic trash',
                'rov',
                'fabric trash',
                'fishing trash',
                'metal trash',
                'paper trash',
                'rubber trash',
                'wood trash',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
                (
                    0,
                    60,
                    100,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    0,
                    0,
                    70,
                ),
                (
                    0,
                    0,
                    192,
                ),
                (
                    250,
                    170,
                    30,
                ),
                (
                    100,
                    170,
                    30,
                ),
                (
                    220,
                    220,
                    0,
                ),
                (
                    175,
                    116,
                    175,
                ),
                (
                    250,
                    0,
                    30,
                ),
                (
                    165,
                    42,
                    42,
                ),
                (
                    255,
                    77,
                    255,
                ),
                (
                    0,
                    226,
                    252,
                ),
                (
                    182,
                    182,
                    255,
                ),
                (
                    0,
                    82,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='datasets/Poseidon-300K/annotations/COCO/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    _scope_='mmdet', max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='annotations/COCO/train.json',
        backend_args=None,
        data_prefix=dict(img='images/train/'),
        data_root='datasets/Poseidon-300K/',
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        metainfo=dict(
            classes=(
                'holothurian',
                'echinus',
                'scallop',
                'starfish',
                'fish',
                'diver',
                'cuttlefish',
                'turtle',
                'jellyfish',
                'crab',
                'shrimp',
                'plastic trash',
                'rov',
                'fabric trash',
                'fishing trash',
                'metal trash',
                'paper trash',
                'rubber trash',
                'wood trash',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
                (
                    0,
                    60,
                    100,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    0,
                    0,
                    70,
                ),
                (
                    0,
                    0,
                    192,
                ),
                (
                    250,
                    170,
                    30,
                ),
                (
                    100,
                    170,
                    30,
                ),
                (
                    220,
                    220,
                    0,
                ),
                (
                    175,
                    116,
                    175,
                ),
                (
                    250,
                    0,
                    30,
                ),
                (
                    165,
                    42,
                    42,
                ),
                (
                    255,
                    77,
                    255,
                ),
                (
                    0,
                    226,
                    252,
                ),
                (
                    182,
                    182,
                    255,
                ),
                (
                    0,
                    82,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='annotations/COCO/val.json',
        backend_args=None,
        data_prefix=dict(img='images/val/'),
        data_root='datasets/Poseidon-300K/',
        metainfo=dict(
            classes=(
                'holothurian',
                'echinus',
                'scallop',
                'starfish',
                'fish',
                'diver',
                'cuttlefish',
                'turtle',
                'jellyfish',
                'crab',
                'shrimp',
                'plastic trash',
                'rov',
                'fabric trash',
                'fishing trash',
                'metal trash',
                'paper trash',
                'rubber trash',
                'wood trash',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
                (
                    0,
                    60,
                    100,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    0,
                    0,
                    70,
                ),
                (
                    0,
                    0,
                    192,
                ),
                (
                    250,
                    170,
                    30,
                ),
                (
                    100,
                    170,
                    30,
                ),
                (
                    220,
                    220,
                    0,
                ),
                (
                    175,
                    116,
                    175,
                ),
                (
                    250,
                    0,
                    30,
                ),
                (
                    165,
                    42,
                    42,
                ),
                (
                    255,
                    77,
                    255,
                ),
                (
                    0,
                    226,
                    252,
                ),
                (
                    182,
                    182,
                    255,
                ),
                (
                    0,
                    82,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='datasets/Poseidon-300K/annotations/COCO/val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/tood_r50_fpn_1x_poseidon'
