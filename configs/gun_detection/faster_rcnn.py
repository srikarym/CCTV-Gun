# Faster RCNN with Resnet 101
# https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r101_fpn_mstrain_3x_coco.py

_base_ = ['../faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py']

# Load pretrained COCO weights
load_from = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth"

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2)))

# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[28, 34])

lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)

############ runtime
# save best model
total_epochs = 36
evaluation = dict(
    metric = "bbox",
    interval = 1,
    save_best = 'bbox_h_mAP',
    rule = "greater",
    classwise=True
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = 8.
auto_scale_lr = dict(enable=True, base_batch_size=8)

workflow = [('train', 1), ('val', 1)]
############
# only save final model
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

