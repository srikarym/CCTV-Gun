# Convnext Mask R-CNN


_base_ = [
    "../convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco.py"
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth'  # noqa

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
        mask_head = None),
    )


lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)

evaluation = dict(
    metric = "bbox",
    interval = 1,
    save_best = 'bbox_h_mAP',
    rule = "greater"
)
auto_scale_lr = dict(enable=True, base_batch_size=8)
