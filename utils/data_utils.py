import random
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T


def register_datasets(data):
    if data == "data_rgb":
        # Register RGB datasets
        register_coco_instances("my_dataset_train", {}, "Data/train/annotations.json", "Data/train")
        register_coco_instances("my_dataset_val", {}, "Data/val/annotations.json", "Data/val")
        register_coco_instances("my_dataset_test", {}, "Data/test/annotations.json", "Data/test")
    elif data == "data_ndvi":
        # Register NDVI datasets
        register_coco_instances("my_dataset_train", {}, "Data_NDVI/train/annotations.json", "Data_NDVI/train")
        register_coco_instances("my_dataset_test", {}, "Data_NDVI/test/annotations.json", "Data_NDVI/test")
    else:
        raise ValueError(f"Unknown dataset type: {data}")

    print(f"Dataset: {data}")


def visualise_samples(dataset_name="my_dataset_test"):
    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, 2):
        img = cv2.imread(d["file_name"])
        visualiser = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualiser.draw_dataset_dict(d)
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()


# Data augmentations during training
class AugmentedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # Define custom augmentations
        augmentation = [
            T.RandomBrightness(0.8, 1.2),  # Adjust brightness
            T.RandomContrast(0.8, 1.2),    # Adjust contrast
            T.RandomSaturation(0.8, 1.2),  # Adjust saturation
            T.RandomFlip(horizontal=True, vertical=False),  # Random horizontal flip
            T.RandomRotation(angle=[-30, 30], expand=False),  # Random rotation within ±30°
            T.ResizeShortestEdge(short_edge_length=(640, 1024), max_size=1333),  # Resize
            T.RandomCrop("relative_range", (0.8, 0.8)),  # Random cropping
        ]

        # Use the augmented DatasetMapper
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentation)
        return build_detection_train_loader(cfg, mapper=mapper)
