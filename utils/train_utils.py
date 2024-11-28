import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog


def train_model(output_dir, num_classes, train_dataset, test_dataset, num_workers, ims_per_batch, max_iter, batch_size,
                base_lr, threshold):
    """
    Train a Detectron2 model with a given configuration.

    Args:
        output_dir (str): Directory to save trained models and logs.
        num_classes (int): Number of object classes in the dataset (excluding background).
        train_dataset (str): Registered name of the training dataset.
        test_dataset (str): Registered name of the test dataset.
        max_iter (int): Number of iterations for training.
        batch_size (int): Batch size per image.
        base_lr (float): Base learning rate.
    """
    print("Initialising training configuration...")

    # Register datasets in Detectron2
    try:
        # Make sure the datasets are registered first
        DatasetCatalog.get(train_dataset)
        DatasetCatalog.get(test_dataset)
    except Exception as e:
        print(
            f"Error: The dataset {train_dataset} or {test_dataset} is not registered. Please ensure datasets are "
            f"registered correctly.")
        raise

    # Configure Detectron2 model
    try:
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.OUTPUT_DIR = output_dir
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = (train_dataset,)
        cfg.DATASETS.TEST = (test_dataset,)
        cfg.DATALOADER.NUM_WORKERS = num_workers
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
        cfg.SOLVER.BASE_LR = base_lr
        cfg.SOLVER.MAX_ITER = max_iter
        cfg.SOLVER.STEPS = []
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set a custom testing threshold
        cfg.MODEL.ROI_MASK_HEAD.NAME = "ROI_MASK_HEAD"  # Ensure mask head is included

        # Create the output directory if it doesn't exist
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # Initialise trainer and start training
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        print(f"Training completed. Model and logs saved to {cfg.OUTPUT_DIR}.")

    except Exception as e:
        print(f"Error during training: {e}")
        raise
