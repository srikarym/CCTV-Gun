# Swin - S

# _base_ = [
#     '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/gundet_runtime.py'
# ]
_base_ = [
    "../swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py"
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth'  # noqa

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
        mask_head = None),
    )


lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)
