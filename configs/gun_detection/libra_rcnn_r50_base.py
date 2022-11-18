_base_ = ['../libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py', '../_base_/gundet_runtime.py']

# Load pretrained COCO weights
load_from = "https://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth"

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2)))


evaluation = dict(classwise=True)