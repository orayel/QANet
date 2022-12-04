import os
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog

COD10K_ROOT = '../datasets/COD10K'
NC4K_ROOT = '../datasets/NC4K'

COD10K_TRAIN_PATH = os.path.join(COD10K_ROOT, 'train')
COD10K_TRAIN_JSON = os.path.join(COD10K_ROOT, 'annotations/train.json')
COD10K_TEST_PATH = os.path.join(COD10K_ROOT, 'test')
COD10K_TEST_JSON = os.path.join(COD10K_ROOT, 'annotations/test.json')
NC4K_TEST_PATH = os.path.join(NC4K_ROOT, 'test')
NC4K_TEST_JSON = os.path.join(NC4K_ROOT, 'annotations/test.json')

CLASS_NAMES = ["foreground"]

PREDEFINED_SPLITS_DATASET = {
    "cod10k_train": (COD10K_TRAIN_PATH, COD10K_TRAIN_JSON),
    "cod10k_test": (COD10K_TEST_PATH, COD10K_TEST_JSON),
    "nc4k_test": (NC4K_TEST_PATH, NC4K_TEST_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)


def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")
