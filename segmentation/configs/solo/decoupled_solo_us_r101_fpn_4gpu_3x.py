# model settings
model = dict(
    type='SOLO',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3), # C2, C3, C4, C5
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    bbox_head=dict(
        type='DecoupledSOLOHead',
        num_classes=4,
        in_channels=256,
        stacked_convs=4,
        seg_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        with_deform=False,
        loss_ins=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
    ))
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.05,
    kernel='gaussian',  # gaussian/linear
    sigma=2.0,
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/public/ttzhang9/dataset/us2/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[(852, 512), (852, 480), (852, 448),
                    (852, 416), (852, 384), (852, 352)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(768, 448),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
	type='RepeatDataset',
	times=3,
	dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_test_v2.json',
        img_prefix=data_root + 'test_v2/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[20, 23])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 24
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/decoupled_solo_us_release_r101_fpn_4gpu_2x'
load_from = None
resume_from = None
workflow = [('train', 1)]
