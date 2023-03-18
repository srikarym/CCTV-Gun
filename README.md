## Introduction
This repo contains training and evaluation code of CCTV-GUN benchmark. It uses [mmdetection](https://mmdetection.readthedocs.io/en/latest/) to train object detection models.

## Data
We use images from three datasets : 

1. Monash Gun dataset (MGD) [^1]
2. US Real-time Gun detection dataset (USRT) [^2] 
3. UCF Crime scene dataset (UCF) [^3] 

Instructions on how to download these datasets can be found in [dataset_instructions.md](./dataset_instructions.md) .


## Training

All of the above datasets consists of two classes : Person (class 0) and Handgun (class 1). To train a detection model on this dataset, run
```bash
python tools/train.py --config <path/to/model/config.py> --dataset-config <path/to/dataset/config.py> <extra_args>
```

Model config files can be found [here](./configs/gun_detection/)

Dataset config files can be found [here](./configs/_base_/datasets/gun_detection/)

### Extra args
To adjust the training batch size
```
<base_command> --cfg-options data.samples_per_gpu=<batch-size>
```
Using [weights and biases](https://wandb.ai/) to log metrics:
After you create an account in wandb, change `entity` and `project` in [train.py](./tools/train.py) to your wandb username and project name. Then 
```
<base_command> --use-wandb --wandb-name <name-of-the-experiment>
```
### Examples:

Train a Swin-T on MGD
```bash
python tools/train.py --config configs/gun_detection/swin_transformer.py --dataset-config configs/_base_/datasets/gun_detection/mgd.py --cfg-options data.samples_per_gpu=6
```

Fine-tune MGD+USRT trained Deformable-DETR model on UCF
```bash
python tools/train.py --config configs/gun_detection/deformable_detr.py --dataset-config configs/_base_/datasets/gun_detection/ucf.py --cfg-options data.samples_per_gpu=6 --load-from <path/to/trained/model.pth>
```


## Testing
To evaluate a trained model, run
```bash
python tools/test.py <path/to/test/dataset/config.py> <path/to/trained/model> --work-dir <path/to/save/test/scores> --eval bbox
```

### Examples:

Evaluate a faster-rcnn **trained on MGD** on **USRT's test set**

```bash
python tools/test.py configs/gun_detection/faster_rcnn_r50_usrt.py work_dirs/faster_rcnn_r50_mgd/epoch_12.pth --work-dir work_dirs/faster_rcnn_r50_mgd/usrt/ --eval bbox
```

Evaluate a faster-rcnn **trained on USRT** on **MGD's test set** 

```bash
python tools/test.py configs/gun_detection/faster_rcnn_r50_mgd.py work_dirs/faster_rcnn_r50_usrt/epoch_12.pth --work-dir work_dirs/faster_rcnn_r50_usrt/mgd/ --eval bbox
```

To save the bounding box predictions on test set , add `--save-path <path/to/output/folder>` at the end.


## Citations

[^1]: Lim, JunYi, et al. "Deep multi-level feature pyramids: Application for non-canonical firearm detection in video surveillance." Engineering applications of artificial intelligence 97 (2021): 104094.

[^2]: González, Jose L. Salazar, et al. "Real-time gun detection in CCTV: An open problem." Neural networks 132 (2020): 297-308.

[^3]: Sultani, Waqas, Chen Chen, and Mubarak Shah. "Real-world anomaly detection in surveillance videos." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.