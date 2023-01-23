# Use the entire dataset for testing. 


train_dataset_type = "GunDataset"
val_dataset_type = "GunDatasetHOI"
dataset_root = "data/mgd/"

# image norm calculated from mgd
img_norm_cfg = dict(
    mean=[121.112, 115.542, 96.106], std=[57.549, 52.405, 56.779], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu = 2,
    workers_per_gpu = 2,
    test = dict(
        type = val_dataset_type,
        data_root = dataset_root,
        ann_file = "annotation_detection/annotations_all.json",
        img_prefix = "images/",
        pipeline = test_pipeline
    ),
    val = dict(
        type = val_dataset_type,
        data_root = dataset_root,
        ann_file = "annotation_detection/annotations_all.json",
        img_prefix = "images/",
        pipeline = test_pipeline
    )
)