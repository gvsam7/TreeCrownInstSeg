import random
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt


def register_datasets():
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("my_dataset_train", {}, "Data/train/annotations.json", "Data/train")
    register_coco_instances("my_dataset_val", {}, "Data/val/annotations.json", "Data/val")
    register_coco_instances("my_dataset_test", {}, "Data/test/annotations.json", "Data/test")


def visualize_samples(dataset_name="my_dataset_train"):
    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, 2):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()
