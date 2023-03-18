# CCTV-GUN setup

In this file, we provide instructions on how to download and construct CCTV-GUN benchmark.

## Requirements
Some images in the original MGD source have byte code errors. Our script to fix them requires a couple of python packages. To install them, run

```
pip3 install -r requirements/dataset.txt
```

## Downloading images

We ask the user to download the images of MGD, USRT and UCF from original source first.


### MGD
- Download `MGD.rar` file from the original source [link](https://drive.google.com/file/d/12ly_8zSpuPTMoYU3Bw1zGkObPU_RmbK-/view) and extract into the current directory.
    - The extracted folder would have the following structure :
    ```
    MGD
    ├── annotations_cache
    ├── MGD2020
    │   ├── Annotations
    │   ├── ImageSets
    │   │   └── Main
    │   └── JPEGImages
    └── results
        ├── VOC2007
        │   └── Main
        └── VOC2020
            └── Main
    ```
- Run `python3 scripts/copy_images_mgd.py` to copy images used in CCTV-GUN

### USRT
- Download `weapons_images_2fps.zip` file from the original source [link](https://uses0-my.sharepoint.com/personal/jsalazar_us_es/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjsalazar%5Fus%5Fes%2FDocuments%2FShared%2FVICTORY%2FUS%2Fweapons%5Fimages%5F2fps%2Ezip&parent=%2Fpersonal%2Fjsalazar%5Fus%5Fes%2FDocuments%2FShared%2FVICTORY%2FUS&ga=1).
    - Run `unzip weapons_images_2fps.zip`
    - A folder called `Images` will be extracted into the current directory.
- Run `python3 scripts/copy_images_usrt.py` to copy images used in CCTV-GUN

### UCF
- Download `Anomaly-Videos-Part-3.zip` from the original source [link](https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0).
    - Run `unzip Anomaly-Videos-Part-3.zip` 
    - A folder called `Anomaly-Videos-Part-3` will be extracted into the current directory.
- Run `python3 scripts/copy_images_ucf.py` to copy images used in CCTV-GUN. Since UCF dataset has videos, we extract specific frames and save them as JPG images. The frame ids can be found in `data/ucf/frames.json`

After downloading and extracting all three datasets, the data folder should look like this
```
data
├── all_images #Contains images from all 3 sources
├── dataset_pairs
│   ├── mgd_usrt
│   │   ├── annotations_all.json
│   │   ├── annotations_test.json
│   │   ├── annotations_train.json
│   │   └── annotations_val.json
│   ├── ucf_mgd
│   │   ├── annotations_all.json
│   │   ├── annotations_test.json
│   │   ├── annotations_train.json
│   │   └── annotations_val.json
│   └── usrt_ucf
│       ├── annotations_all.json
│       ├── annotations_test.json
│       ├── annotations_train.json
│       └── annotations_val.json
├── mgd
│   ├── annotation_detection
│   │   ├── annotations_all.json
│   │   ├── annotations_test.json
│   │   ├── annotations_train.json
│   │   └── annotations_val.json
│   └── images 
├── ucf
│   ├── annotation_detection
│   │   ├── annotations_all.json
│   │   ├── annotations_test.json
│   │   ├── annotations_train.json
│   │   └── annotations_val.json
│   ├── frames.json
│   └── images
└── usrt
    ├── annotation_detection
    │   ├── annotations_all.json
    │   ├── annotations_test.json
    │   ├── annotations_train.json
    │   └── annotations_val.json
    └── images
```

## Annotations
We provide bounding box annotations of all images in CCTV-GUN dataset in `data/<split>/annotations_detection`. Train-val-test split can be found in the same folder. The annotations are in MS-COCO format, and also contain the person-handgun pair. 

To train models on pairs of datasets (Ex: MGD + USRT), the annotations provided in `data/dataset_pairs/<split>` can be used.

