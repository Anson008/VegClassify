crop_size = (
    256,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        62.14484866,
        67.69474223,
        65.19883835,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        512,
    ),
    std=[
        39.26080611,
        38.97791043,
        40.7339263,
    ],
    type='SegDataPreProcessor')
data_root = '.\\data\\RN47963_C18_512X1024_Num100_voc'
dataset_type = 'SingleGreenDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_norm_cfg = dict(
    mean=[
        62.14484866,
        67.69474223,
        65.19883835,
    ],
    std=[
        39.26080611,
        38.97791043,
        40.7339263,
    ],
    to_rgb=True)
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=[
        dict(
            align_corners=False,
            channels=672,
            concat_input=False,
            dropout_ratio=-1,
            in_channels=[
                96,
                192,
                384,
            ],
            in_index=(
                1,
                2,
                3,
            ),
            input_transform='resize_concat',
            kernel_size=1,
            loss_decode=dict(
                loss_name='loss_dice',
                loss_weight=0.4,
                type='DiceLoss',
                use_sigmoid=True),
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=2,
            num_convs=1,
            type='FCNHead'),
        dict(
            align_corners=False,
            channels=576,
            concat_input=False,
            dropout_ratio=-1,
            in_channels=[
                192,
                384,
            ],
            in_index=(
                2,
                3,
            ),
            input_transform='resize_concat',
            kernel_size=1,
            loss_decode=dict(
                loss_name='loss_dice',
                loss_weight=0.4,
                type='DiceLoss',
                use_sigmoid=True),
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=2,
            num_convs=1,
            type='FCNHead'),
    ],
    backbone=dict(
        extra=dict(
            stage1=dict(
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_branches=1,
                num_channels=(64, ),
                num_modules=1),
            stage2=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                ),
                num_branches=2,
                num_channels=(
                    48,
                    96,
                ),
                num_modules=1),
            stage3=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                ),
                num_branches=3,
                num_channels=(
                    48,
                    96,
                    192,
                ),
                num_modules=4),
            stage4=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_branches=4,
                num_channels=(
                    48,
                    96,
                    192,
                    384,
                ),
                num_modules=3)),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        type='HRNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            62.14484866,
            67.69474223,
            65.19883835,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            512,
        ),
        std=[
            39.26080611,
            38.97791043,
            40.7339263,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=720,
        concat_input=False,
        dropout_ratio=-1,
        in_channels=[
            48,
            96,
            192,
            384,
        ],
        in_index=(
            0,
            1,
            2,
            3,
        ),
        input_transform='resize_concat',
        kernel_size=1,
        loss_decode=dict(
            loss_name='loss_dice',
            loss_weight=1.0,
            type='DiceLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        num_convs=1,
        type='FCNHead'),
    pretrained='open-mmlab://msra/hrnetv2_w48',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=80000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='splits\\val.txt',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        data_root='.\\data\\RN47963_C18_512X1024_Num100_voc',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='SingleGreenDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mDice',
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        512,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=1000, type='IterBasedTrainLoop', val_interval=100)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='splits\\train.txt',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        data_root='.\\data\\RN47963_C18_512X1024_Num100_voc',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    512,
                    1024,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    256,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='SingleGreenDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            512,
            1024,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        256,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='splits\\val.txt',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        data_root='.\\data\\RN47963_C18_512X1024_Num100_voc',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='SingleGreenDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mDice',
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/fcn_hr48_test_run'
