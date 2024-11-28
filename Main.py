""""
Author: Georgios Voulgaris
Date: 22/11/2024
Description: This python script uses Detectron2 to train a Mask R-CNN to segment individual tree crowns.
"""

import os
from utils.data_utils import register_datasets, visualize_samples
from utils.train_utils import train_model
from utils.inference_utils import initialise_predictor, run_inference, export_results_to_csv
from utils.hyperparameters import arguments
import torch
from detectron2.data import MetadataCatalog


def main():
    args = arguments()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on the: {device}")

    # Step 1: Setup environment and datasets
    register_datasets()
    # visualize_samples()

    # Step 2: Train the model
    config_file = "Detectron2/config.yaml"
    train_model(
        output_dir="outputs/results",
        num_classes=args.num_classes,
        train_dataset="my_dataset_train",
        test_dataset="my_dataset_test",
        num_workers=args.num_workers,
        ims_per_batch=args.ims_per_batch,
        max_iter=args.max_iter,  # 1000,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        threshold=args.threshold,
    )

    # Step 3: Initialise the predictor
    cfg, predictor = initialise_predictor(config_file)

    # Step 4: Run inference on the test set
    test_images_dir = "Data/test"
    output_dir = "outputs/results"
    # metadata = cfg.DATASETS.TRAIN[0]
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Debugging to ensure metadata is correct
    print(f"Main Metadata type: {type(metadata)}")  # Debug line
    run_inference(test_images_dir, output_dir, predictor, metadata)

    # Step 5: Export results to a CSV
    output_csv_path = os.path.join(output_dir, "output_objects.csv")
    export_results_to_csv(test_images_dir, output_csv_path, predictor, metadata)


if __name__ == "__main__":
    main()
