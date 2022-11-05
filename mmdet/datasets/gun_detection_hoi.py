from .builder import DATASETS
from .coco import CocoDataset
from mmdet.datasets.coco_hoi import CocoHOIDataset


@DATASETS.register_module()
class GunDatasetHOI(CocoHOIDataset):
    CLASSES = ('handgun', 'person')
