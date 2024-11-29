""""
Author: Georgios Voulgaris
Date: 22/11/2024
Description: This python script uses Detectron2 to train a Mask R-CNN to segment individual tree crowns.
"""

import os
from utils.data_utils import register_datasets, visualize_samples
from utils.train_utils import train_model
from utils.test_utils import evaluate_model
from utils.inference_utils import initialise_predictor, run_inference, export_results_to_csv
from utils.hyperparameters import arguments
import torch
from detectron2.data import MetadataCatalog
import pandas as pd
import wandb


def main():
    args = arguments()

    # Initialize wandb with more detailed configuration
    wandb.init(
        project="InstSeg",
        config={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_classes": args.num_classes,
            "num_workers": args.num_workers,
            "ims_per_batch": args.ims_per_batch,
            "max_iter": args.max_iter,
            "batch_size": args.batch_size,
            "base_lr": args.base_lr,
            "threshold": args.threshold
        }
    )

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on the: {device}")

    # Step 1: Setup environment and datasets
    register_datasets()
    # visualize_samples()

    # Step 2: Train the model
    config_file = "Detectron2/config.yaml"
    train_model(
        device=device,
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

    # Step 4: Evaluate the model
    test_dataset = "my_dataset_test"  # Can be changed to any dataset name
    output_dir = "outputs/results"  # Define your output directory
    # Evaluate model and get results
    evaluation_results = evaluate_model(cfg, predictor, test_dataset, output_dir)
    # Log evaluation results (for example, using wandb)
    wandb.log({"evaluation_results": evaluation_results})

    # Step 5: Run inference on the test set
    test_images_dir = "Data/test"
    output_dir = "outputs/results"
    # metadata = cfg.DATASETS.TRAIN[0]
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Debugging to ensure metadata is correct
    print(f"Main Metadata type: {type(metadata)}")  # Debug line
    run_inference(test_images_dir, output_dir, predictor, metadata)

    # Step 6: Export results to a CSV
    output_csv_path = os.path.join(output_dir, "output_objects.csv")
    export_results_to_csv(test_images_dir, output_csv_path, predictor, metadata)

    # Wandb logging
    # Log the CSV file
    wandb.save(output_csv_path)

    # inference_results_dir = os.path.join(output_dir, "inference_results")
    # Create the output directory if it doesn't exist
    # os.makedirs(inference_results_dir, exist_ok=True)
    inference_images = []
    for img_file in os.listdir(output_dir):
        if img_file.endswith(('_result.png', '.jpg', '.jpeg')):
            inference_images.append(wandb.Image(
                os.path.join(output_dir, img_file),
                caption=f"Inference Result: {img_file}"
            ))

    # Log images to wandb
    wandb.log({"inference_results": inference_images})

    # Optional: Log some summary statistics from the CSV
    csv_df = pd.read_csv(output_csv_path)
    wandb.log({
        "total_detected_objects": len(csv_df),
        "unique_classes": csv_df['Category_ID'].nunique(),
        "avg_object_area": csv_df['Area'].mean(),
        "max_object_area": csv_df['Area'].max(),
        "min_object_area": csv_df['Area'].min()
    })


if __name__ == "__main__":
    main()
