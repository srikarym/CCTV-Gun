from .builder import DATASETS
from .coco import CocoDataset
from mmdet.datasets.coco_hoi import CocoHOIDataset


@DATASETS.register_module()
class GunDatasetHOI(CocoHOIDataset):
    CLASSES = ('person', 'handgun')
    PALETTE = [(0, 255, 0), (255, 0, 0)]