_base_ = [
    'mmdet::_base_/default_runtime.py', 'mixpl_irdst_detection.py'
]

custom_imports = dict(
    imports=['projects.MixPL.mixpl'], allow_failed_imports=False)

detector = dict(
    type='FCOS',
    add_seg = True,
    add_enhance = True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675], 
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=True,
        pad_size_divisor=32), 
    backbone=dict(
        type='ResNetSeq',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpt/resnet50_msra-5891d200.pth')), # 
    neck=dict(
        type='FPNSeq',
        in_channels=[256, 512, 1024, 2048], # 
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_frame=5,
        num_outs=5,  # [1,256,64,64], [1,256,32,32], [1,256,16,16], [1,256,8,8], [1,256,4,4]
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='TOODHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8, 
            scales_per_octave=1,
            strides=[8,16,32,64,128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        # assigner=dict(type='TaskAlignedAssigner', topk=13),
        assigner = dict(type='DynamicSoftLabelAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    # bbox_head=dict(
    #     type='FCOSHead',
    #     num_classes=1,
    #     in_channels=256,
    #     stacked_convs=3,#4
    #     regress_ranges=((-1, 64), (64, 128),(128, 256), (256, 1e8)),
    #     feat_channels=256,
    #     strides=[8, 16, 32, 64],
    #     norm_on_bbox=True,
    #     centerness_on_reg=True,
    #     dcn_on_last_conv=False,
    #     center_sampling=True,
    #     conv_bias=True, 
    #     loss_cls=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True, 
    #         gamma=2.0, 
    #         alpha=0.25,
    #         loss_weight=1.0),
    #     loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
    #     loss_centerness=dict(
    #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=1,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65), #type=''
        max_per_img=100))

model = dict( 
    type='MixPL',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector['data_preprocessor']),
    semi_train_cfg=dict(
        least_num=1,
        cache_size=50, 
        # mixup=False,
        # mosaic=False,
        # mosaic_shape=[(400, 400), (800, 800)],
        # mosaic_weight=0.5,
        erase=False, 
        erase_patches=(1, 10),
        erase_ratio=(0, 0.02),
        erase_thr=0.7,
        cls_pseudo_thr=0.4,
        freeze_teacher=True, 
        sup_weight=1.0,
        unsup_weight=2.0,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# 10% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10.json'
unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10-unlabeled.json'
labeled_dataset.data_prefix = dict(img='')
unlabeled_dataset.data_prefix = dict(img='')

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(batch_size=4, source_ratio=[1, 3]),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=180000, val_interval=3000,val_begin = 2000) #,val_begin = 3000
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')
 
# learning rate policy 
param_scheduler = [
    dict(
       type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    # dict(type='CosineAnnealingLR',
    #      T_max=500,
    #      by_epoch=False,
    #     #  eta_min = 0,
    #      begin=500,
    #      end=10000)
]

# optimizer 
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    # optimizer=dict(
    #     type='AdamW',
    #     lr=0.0001,  # 0.0002 for DeformDETR
    #     weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2),
) 

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=1,save_best='student/coco/bbox_mAP_50',  # 按照该指标保存最优模型
        type='CheckpointHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),  
)
log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook', momentum=0.0002, gamma=4)]
resume=False
seed = 3407