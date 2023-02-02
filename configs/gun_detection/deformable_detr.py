# ++ two-stage Deformable DETR	

_base_ = ["../deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py"]

load_from = "https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth"

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(num_classes=2, as_two_stage=True))

# save best model
total_epochs = 36
evaluation = dict(
    metric = "bbox",
    interval = 1,
    save_best = 'bbox_h_mAP',
    rule = "greater"

)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

lr_config = dict(policy='step', step=[28])
runner = dict(type='EpochBasedRunner', max_epochs=36)


# learning policy
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)
