## Introduction
This repo uses [mmdetection](https://mmdetection.readthedocs.io/en/latest/) to train object detection models on handgun datasets.

## Data
We use three datasets in total - 

1. [Monash Gun dataset (MGD)](https://github.com/MarcusLimJunYi/Monash-Guns-Dataset)
2. [US Real-time Gun detection dataset](https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset)
3. UCF Crime scene dataset

Instructions on how to download these datasets will be uploaded soon.

## Usage

### Training

All of the above datasets consists of two classes : Person (class 0) and Handgun (class 1). To train a detection model on this dataset, run
```bash
python tools/train.py <path-to-config.py> <extra_args>
```
Example:
```bash
python tools/train.py configs/gun_detection/faster_rcnn_r50_mgd.py
```

#### Extra args
To adjust the training batch size
```
python tools/train.py <path-to-config> **--cfg-options data.samples_per_gpu=<batch-size>**
```
Using weights and biases to log metrics:
After you create an account in wandb, change `entity` in train.py to your wandb username. Then 
```
python tools/train.py <path-to-config> --use-wandb --wandb-name <name-of-the-experiment>
```

### Testing
To evaluate a trained model, run
```bash
python tools/test.py <path-to-test-dataset-config.py> <path-to-trained-model> --work-dir <path-to-save-test-scores> --eval bbox
```

Example:
Evaluate a faster-rcnn **trained on MGD** on **USRT's test set**

```bash
python tools/test.py configs/gun_detection/faster_rcnn_r50_usrt.py work_dirs/faster_rcnn_r50_mgd/epoch_12.pth --work-dir work_dirs/faster_rcnn_r50_mgd/usrt/ --eval bbox
```

Evaluate a faster-rcnn **trained on USRT** on **MGD's test set** 

```bash
python tools/test.py configs/gun_detection/faster_rcnn_r50_mgd.py work_dirs/faster_rcnn_r50_usrt/epoch_12.pth --work-dir work_dirs/faster_rcnn_r50_usrt/mgd/ --eval bbox
```

To save the bounding box predictions on test set , add `--save-path <path-to-output-folder>` at the end.