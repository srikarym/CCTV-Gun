# MGD and USRT gun detection datasets

from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class GunDataset(CocoDataset):
    CLASSES = ('person', 'handgun')

    PALETTE = [(0, 255, 0), (0, 0, 255)]
