""""
Author: Georgios Voulgaris
Date: 22/11/2024
Description: This python script uses Detectron2 to train a Mask R-CNN to segment individual tree crowns.
"""

import os
from utils.data_utils import register_datasets, visualize_samples
from utils.train_utils import train_model
from utils.inference_utils import initialise_predictor, run_inference, export_results_to_csv


def main():
    # Step 1: Setup environment and datasets
    register_datasets()
    # visualize_samples()

    # Step 2: Train the model
    config_file = "Detectron2/config.yaml"
    train_model(
        output_dir="outputs/results",
        num_classes=11,
        train_dataset="my_dataset_train",
        test_dataset="my_dataset_test",
        num_workers=2,
        ims_per_batch=2,
        max_iter=2,  # 1000,
        batch_size=256,
        base_lr=0.00025,
    )

    # Step 3: Initialise the predictor
    cfg, predictor = initialise_predictor(config_file)

    # Step 4: Run inference on the test set
    test_images_dir = "Data/test"
    output_dir = "outputs/results"
    metadata = cfg.DATASETS.TRAIN[0]
    run_inference(test_images_dir, output_dir, predictor, metadata)

    # Step 5: Export results to a CSV
    output_csv_path = os.path.join(output_dir, "output_objects.csv")
    export_results_to_csv(test_images_dir, output_csv_path, predictor)


if __name__ == "__main__":
    main()
