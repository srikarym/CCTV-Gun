## Introduction
This repo contains training and evaluation code of CCTV-GUN benchmark. It uses [mmdetection](https://mmdetection.readthedocs.io/en/latest/) to train object detection models.

## Data
We use images from three datasets : 

1. Monash Gun dataset (MGD) [^1]
2. US Real-time Gun detection dataset (USRT) [^2] 
3. UCF Crime scene dataset (UCF) [^3] 

Instructions on how to download these datasets can be found in [dataset_instructions.md](./dataset_instructions.md) .

## Setup

We perform two kinds of assessment : Intra-dataset and Cross-dataset (See paper for more details). We train five detection models : 
- Faster R-CNN [^4]
- Swin-T [^5]
- Deformable DETR [^6]
- DetectoRS [^7]
- ConvNeXt-T [^8]

## Training

All of the above datasets consists of two classes : Person (class 0) and Handgun (class 1). To train a detection model on this dataset, run
```bash
python tools/train.py --config <path/to/model/config.py> --dataset-config <path/to/dataset/config.py> <extra_args>
```

- Model config files [link](./configs/gun_detection/)

- Dataset config files [link](./configs/_base_/datasets/gun_detection/)

- Trained models link.

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

Train a Swin-T on MGD (Intra-dataset)
```bash
python tools/train.py --config configs/gun_detection/swin_transformer.py --dataset-config configs/_base_/datasets/gun_detection/mgd.py --cfg-options data.samples_per_gpu=6
```

Train a detectoRS model on MGD+USRT (Inter-dataset)
```bash
python tools/train.py --config configs/gun_detection/detectors.py --dataset-config configs/_base_/datasets/gun_detection/mgd_usrt.py --cfg-options data.samples_per_gpu=4
```

Fine-tune MGD+USRT trained Deformable-DETR model on UCF (Inter-dataset)
```bash
python tools/train.py --config configs/gun_detection/deformable_detr.py --dataset-config configs/_base_/datasets/gun_detection/ucf.py --cfg-options data.samples_per_gpu=6 --load-from <path/to/trained/model.pth>
```


## Testing
To evaluate a trained model, run
```bash
python tools/test.py --config <path/to/model/config.py> --dataset-config <path/to/dataset/config.py> --checkpoint <path/to/trained/model> --work-dir <path/to/save/test/scores> --eval bbox
```

### Examples:

Evaluate a faster-rcnn trained on MGD on MGD's test set (Intra-dataset)

```bash
python tools/test.py --config configs/gun_detection/faster_rcnn.py --dataset-config configs/_base_/datasets/gun_detection/mgd.py --checkpoint <path/to/mgd/trained/model.pth> --work-dir <path/to/save/test/scores> --eval bbox
```

Evaluate a ConvNeXt trained on MGD+USRT on the entirety of UCF (Inter-dataset) 

```bash
python tools/test.py --config configs/gun_detection/convnext.py --dataset-config configs/_base/datasets/gun_detection/ucf_test_full.py --checkpoint <path/to/mgd+usrt/trained/model.pth> --work-dir <path/to/save/test/scores> --eval bbox
```

To save the bounding box predictions on test set , add `--save-path <path/to/output/folder>` to the above command.


## References

[^1]: Lim, JunYi, et al. "Deep multi-level feature pyramids: Application for non-canonical firearm detection in video surveillance." Engineering applications of artificial intelligence 97 (2021): 104094.

[^2]: González, Jose L. Salazar, et al. "Real-time gun detection in CCTV: An open problem." Neural networks 132 (2020): 297-308.

[^3]: Sultani, Waqas, Chen Chen, and Mubarak Shah. "Real-world anomaly detection in surveillance videos." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[^4]: Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." Advances in neural information processing systems 28 (2015).

[^5]: Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[^6]: Zhu, Xizhou, et al. "Deformable detr: Deformable transformers for end-to-end object detection." arXiv preprint arXiv:2010.04159 (2020).

[^7]: Qiao, Siyuan, Liang-Chieh Chen, and Alan Yuille. "Detectors: Detecting objects with recursive feature pyramid and switchable atrous convolution." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

[^8]: Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.