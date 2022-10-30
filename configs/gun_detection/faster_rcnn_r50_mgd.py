_base_ = ['../_base_/models/faster_rcnn_r50_fpn.py', 
        '../_base_/datasets/mgd.py',
        '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
        ]

# Load pretrained COCO weights
load_from = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

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
    step=[8,11])

log_config = dict(interval = 10)
total_epochs = 12
evaluation = dict(
    metric = "bbox",
    interval = 1
)