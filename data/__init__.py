from .dataset import *


def create_dataset(config):
    if config.data.dataset_name == "rocstory":
        return DatasetDDP
    if config.data.dataset_name == "wikipedia":
        return DatasetDDP
