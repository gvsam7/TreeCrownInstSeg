""""
Author: Georgios Voulgaris
Date: 22/11/2024
Description: This python script uses Detectron2 to train a Mask R-CNN to segment individual tree crowns.
"""

import os
from utils.data_utils import register_datasets, visualize_samples
from utils.train_utils import train_model
from utils.inference_utils import run_inference, export_results


def main():
    # Step 1: Setup environment and datasets
    register_datasets()
    visualize_samples()

    # Step 2: Train the model
    config_file = "config.yaml"
    train_model()

    # Step 3: Run inference on the test set
    test_images_dir = "Data/test"
    output_dir = "outputs/results"
    run_inference(test_images_dir, config_file, output_dir)

    # Step 4: Export results to a CSV
    export_results(test_images_dir, output_dir, "outputs/results/output_objects.csv")


if __name__ == "__main__":
    main()
