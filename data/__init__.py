from .dataset import *
from .dataset_wiki import DatasetWikiDDP


def create_dataset(config):
    if config.data.dataset_name == "rocstory":
        return DatasetDDP
    if config.data.dataset_name == "wikipedia":
        return DatasetWikiDDP
