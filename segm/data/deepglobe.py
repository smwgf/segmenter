from pathlib import Path

from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_dir


DEEPGLOBE_CONFIG_PATH = Path(__file__).parent / "config" / "deepglobe.py"
DEEPGLOBE_CATS_PATH = Path(__file__).parent / "config" / "deepglobe.yml"


class DeepglobeDataset(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(
            image_size,
            crop_size,
            split,
            DEEPGLOBE_CONFIG_PATH,
            **kwargs,
        )
        self.names, self.colors = utils.dataset_cat_description(DEEPGLOBE_CATS_PATH)
        self.n_cls = 2
        self.ignore_label = 0
        self.reduce_zero_label = True

    def update_default_config(self, config):
        root_dir = dataset_dir()
        path = Path(root_dir) / "deepglobe"
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path
        elif self.split == "trainval":
            config.data.trainval.data_root = path
        elif self.split == "val":
            config.data.val.data_root = path
        elif self.split == "test":
            config.data.test.data_root = path
        config = super().update_default_config(config)
        return config

    def test_post_process(self, labels):
        return labels + 1
